from typing import Any, Optional, List

import torch
from genrl.data import DataManager
from genrl.logging_utils.global_defs import get_logger
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer.grpo_trainer import GRPOLanguageTrainerModule
from reasoning_gym.utils import SYSTEM_PROMPTS
from rgym_exp.src.utils.judge_client import JudgeClient
from rgym_exp.src.prg_module import PRGGameStatus

# Added imports for BitsAndBytes quantization
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except Exception:
    # If transformers not available at import time, user environment will raise when used.
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

PRG_SYSTEM_PROMPT = """Given a question, hints, and possible answers, your task is to answer the question by thinking step-by-step in a clear and specific manner for 1 line only.
Your answer MUST be one of the possible answers. Provide the answer in the following format:
<answer>answer here</answer>
Do not explain your reasoning inside the answer tags, provide only the final answer.
"""

PRG_SYSTEM_PROMPT_NO_THINKING = """Given a question, hints, and possible answers, your task is to answer the question.
Your answer MUST be one of the possible answers. Give your answer in the following format:
<answer>answer here</answer>
Do not explain your reasoning at all, provide only the final answer in the answer tag.
"""

class GRPOTrainerModule(GRPOLanguageTrainerModule, LoggerMixin):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method.
    Implements the TrainerModule interface defined in base_trainer.py.

    Supports optional loading of a quantized model via BitsAndBytes (8/4-bit).
    Pass kwargs like:
        use_bnb=True,
        bnb_4bit=True,
        bnb_quant_type="nf4",
        bnb_compute_dtype=torch.float16,
        device_map="auto",
        tokenizer_name_or_path=None
    Or pass `models` as [<model_name_or_path>] to let the trainer load the model.
    """

    def __init__(self, models: List[Any], **kwargs):
        """
        Initialize the GRPO trainer module.

        Args:
            models: List containing either a loaded model object or a string model path/name.
            **kwargs: Additional arguments for configuration.
        """
        super().__init__(models, **kwargs)
        judge_base_url = kwargs.get("judge_base_url", None)
        self.judge_client = JudgeClient(judge_base_url) if judge_base_url else None

        # BitsAndBytes / quantization config
        self.use_bnb = kwargs.get("use_bnb", True)  # default True; set False to disable
        self.bnb_4bit = kwargs.get("bnb_4bit", True)
        self.bnb_quant_type = kwargs.get("bnb_quant_type", "nf4")  # "nf4" or "fp4" etc.
        self.bnb_compute_dtype = kwargs.get("bnb_compute_dtype", torch.float16)
        self.device_map = kwargs.get("device_map", "auto")
        self.tokenizer_name_or_path = kwargs.get("tokenizer_name_or_path", None)

        # If models[0] is a string path, load tokenizer/model with bnb
        first = models[0] if models else None
        if isinstance(first, str):
            model_name = first
            tokenizer_name = self.tokenizer_name_or_path or model_name
            self.model, self.tokenizer = self._load_model_and_tokenizer(
                model_name,
                tokenizer_name,
            )
            # set processing_class tokenizer if needed (some frameworks expect .tokenizer attr)
            if hasattr(self, "processing_class") and getattr(self.processing_class, "tokenizer", None) is None:
                try:
                    self.processing_class.tokenizer = self.tokenizer
                except Exception:
                    pass
            # keep reference for name_or_path checks used elsewhere
            self.model.name_or_path = model_name
        else:
            # assume user passed already loaded model object and maybe tokenizer
            self.model = first
            # try to get tokenizer from kwargs
            self.tokenizer = kwargs.get("tokenizer", None)

        # configure generation kwargs safe defaults
        self.generation_kwargs = kwargs.get("generation_kwargs", {"max_new_tokens": 512, "do_sample": False})

    def _load_model_and_tokenizer(self, model_name: str, tokenizer_name: Optional[str] = None):
        """
        Load tokenizer + model with BitsAndBytes quantization if requested.
        Returns (model, tokenizer).
        """
        tokenizer_name = tokenizer_name or model_name
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers is required for BitsAndBytes loading. Install 'transformers' and 'bitsandbytes'.")

        # load tokenizer (fast)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        if not self.use_bnb:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.device_map)
            return model, tokenizer

        # Build BitsAndBytesConfig for 4-bit or 8-bit
        if self.bnb_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.bnb_quant_type,
                bnb_4bit_compute_dtype=self.bnb_compute_dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device_map,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
        else:
            # 8-bit loading (load_in_8bit)
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=self.bnb_compute_dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device_map,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )

        # move to device if device_map not used
        try:
            if isinstance(self.device_map, str) and self.device_map == "auto":
                pass  # device_map managed by accelerate
            else:
                # if single device specified
                device = torch.device(self.device_map if isinstance(self.device_map, str) else "cpu")
                model.to(device)
        except Exception:
            pass

        return model, tokenizer

    @torch.no_grad()
    def evaluate(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ):
        if not self.judge_client:
            return
            
        try:
            model_name = self.model.name_or_path
        except AttributeError:
            model_name = "none"

        # Request question from judge service
        result = self.judge_client.request_question(
            user_id=state.peer_id,
            round_number=state.round,
            model_name=model_name
        )
        
        if not result:
            return

        # Generate answer using the model
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPTS["default"]},
            {"role": "user", "content": result["question"]},
        ]
        input_ids = self.processing_class.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # ensure device
        device = self.model.device if hasattr(self.model, "device") else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        input_ids = input_ids.to(device)

        # If tokenizer is available for decode
        gen_kwargs = dict(self.generation_kwargs)
        gen_kwargs.setdefault("max_new_tokens", 512)

        # Use autocast for fp16 compute when available & desired
        use_autocast = (torch.cuda.is_available() and self.bnb_compute_dtype == torch.float16)
        with torch.cuda.amp.autocast(enabled=use_autocast):
            outputs = self.model.generate(input_ids, **gen_kwargs)

        # decode
        answer = self.processing_class.decode(outputs[0], skip_special_tokens=True)
        
        # Submit answer to judge service
        self.judge_client.submit_answer(
            session_id=result["session_id"],
            round_number=state.round,
            user_answer=answer
        )

    @torch.no_grad()
    def play_prg_game_logits(
        self, prg_history_dict: dict
    ) -> dict:
        if not self.judge_client:
            return {'status': PRGGameStatus.ERROR}

        # Get current clue from judge service
        game_clue_dict = self.judge_client.get_current_clue()
        
        if not isinstance(game_clue_dict, dict):
            return {'status': PRGGameStatus.ERROR}
        
        # If no clue or game_id or clue_id is -1, take no action
        game_id = game_clue_dict.get("game_id", -1)
        clue_id = game_clue_dict.get("clue_id", -1)
        rounds_remaining = game_clue_dict.get("rounds_remaining", -1)
        clue = game_clue_dict.get("clue") or ""
        choices = game_clue_dict.get("choices") or []
        
        # No active game
        if any(val < 0 for val in (game_id, clue_id, rounds_remaining)):
            return {'status': PRGGameStatus.NO_ACTIVE_GAME}
        # We have already answered this clue
        if game_id in prg_history_dict and clue_id <= prg_history_dict[game_id]:
            return {'status': PRGGameStatus.ALREADY_ANSWERED}
        # malformed input
        if not clue or not isinstance(choices, list) or not choices:
            return {'status': PRGGameStatus.ERROR}
        
        get_logger().info(f"New clue received for PRG: {game_clue_dict}")

        try:
            choices_str = ", ".join(choices)
            custom_prompt = f"{clue}\nPossible Answers: {choices_str}\nAnswer:"
            
            # Generate answer using the model with custom prompt
            prompt = [
                {"role": "system", "content": PRG_SYSTEM_PROMPT_NO_THINKING},
                {"role": "user", "content": custom_prompt},
            ]
            input_ids = self.processing_class.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            # ensure device
            device = self.model.device if hasattr(self.model, "device") else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            input_ids = input_ids.to(device)
            
            # Get logits for each choice
            choice_logits = self._get_choice_logits(input_ids, choices)
            
            # Select the choice with highest probability
            choice_idx = torch.argmax(choice_logits).item()
            return {
                "game_idx": game_id,
                "clue_idx": clue_id,
                "choice_idx": choice_idx,
                "choice": choices[choice_idx],
                "rounds_remaining": rounds_remaining,
                "status": PRGGameStatus.SUCCESS
            }

        except Exception as e:
            get_logger().info(f"Error while computing logits for choices: {e}")
            return {'status': PRGGameStatus.ERROR}

    def _get_choice_logits(self, input_ids: torch.Tensor, choices: List[str]) -> torch.Tensor:
        """
        Returns a tensor of shape (len(choices),) giving, for each choice,
        the sum of log-probabilities that the model assigns to generating
        "<answer>{choice}</answer>" after the given input_ids.

        Works with quantized models loaded via BitsAndBytes.
        """
        device = input_ids.device
        batch_size, prompt_len = input_ids.shape
        logits_list = []

        # Use autocast if compute dtype suggests float16
        use_autocast = (torch.cuda.is_available() and self.bnb_compute_dtype == torch.float16)

        for choice in choices:
            # 1) build the full token sequence: prompt + "<answer>…</answer>"
            answer_str = f"<answer>{choice}</answer>"
            # If we have tokenizer from loading stage, use it; otherwise use processing_class
            if getattr(self, "tokenizer", None) is not None:
                choice_ids = self.tokenizer(answer_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            else:
                choice_ids = self.processing_class(
                    answer_str,
                    return_tensors="pt",
                    add_special_tokens=False
                ).input_ids.to(device)    # shape (1, L)

            seq = torch.cat([input_ids, choice_ids], dim=1)  # (1, prompt_len + L)

            # build labels that only include the answer positions
            labels = seq.clone()
            labels[:, :prompt_len] = -100  # ignore prompt positions in loss

            # forward pass — for quantized models, still works with labels
            with torch.cuda.amp.autocast(enabled=use_autocast):
                outputs = self.model(input_ids=seq, labels=labels)

            # outputs.loss is average negative log-likelihood over the L answer tokens
            # convert to total log-prob (sum of log-probs across tokens)
            total_log_prob = -outputs.loss * choice_ids.size(1)
            logits_list.append(total_log_prob.detach().to("cpu"))

        # stack into a single tensor of shape (num_choices,)
        return torch.stack(logits_list).squeeze()
