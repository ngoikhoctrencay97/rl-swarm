from typing import Any, List
import torch
import traceback
import sys

from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer.grpo_trainer import GRPOLanguageTrainerModule

from code_gen_exp.src.utils.judge_client import JudgeClient
from code_gen_exp.src.solver_data import SYSTEM_PROMPTS

# Optional
try:
    from transformers import AutoTokenizer
except:
    AutoTokenizer = None


class GRPOTrainerModule(GRPOLanguageTrainerModule, LoggerMixin):
    """
    FINAL FULL VERSION:
    Stable for Qwen2.5 Coder 4-bit on HF+BitsAndBytes.
    Avoids dtype cast, avoids HF multinomial sampling, fixes pad tokens,
    overrides _model_generate(), safe for batch size > 1.
    """

    def __init__(self, models: List[Any], **kwargs):
        self._local_tokenizer = None

        # super() calls our overridden _initialize_model()
        super().__init__(models, **kwargs)

        # Judge client
        judge_base_url = kwargs.get("judge_base_url", None)
        self.judge_client = JudgeClient(judge_base_url) if judge_base_url else None

        self.system_prompt = SYSTEM_PROMPTS.get("solver", SYSTEM_PROMPTS["default"])

        # Load tokenizer
        self._init_tokenizer(kwargs)

        # Ensure pad token exists
        try:
            self._ensure_pad_token()
        except Exception:
            print("[GRPOTrainerModule] Warning: pad token ensure failed:\n",
                  traceback.format_exc(),
                  file=sys.stderr)

    # ===============================================================
    #  OVERRIDE: BLOCK genrl from dtype-casting quantized models
    # ===============================================================
    def _initialize_model(self, enable_gradient_checkpointing: bool):
        model_type = str(type(self.model)).lower()
        quantized = (
            hasattr(self.model, "is_quantized")
            or "bitsandbytes" in model_type
            or "bnb" in model_type
        )

        if quantized:
            print("[GRPOTrainerModule] Detected BitsAndBytes 4-bit model → skipping dtype cast")

            try:
                self.model = self.model.to(self.device)
            except:
                pass

            try:
                self.model.config.use_cache = False
            except:
                pass

            if enable_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
                try:
                    self.model.gradient_checkpointing_enable()
                except:
                    pass

            return

        # Regular, non-quantized
        super()._initialize_model(enable_gradient_checkpointing)

    # ===============================================================
    #  Load tokenizer
    # ===============================================================
    def _init_tokenizer(self, kwargs):
        tok = getattr(self.processing_class, "tokenizer", None)
        if tok is not None:
            self._local_tokenizer = tok
            return

        tok = kwargs.get("tokenizer", None)
        if tok is not None:
            self._local_tokenizer = tok
            return

        # Auto-load tokenizer
        if AutoTokenizer and hasattr(self.model, "name_or_path"):
            try:
                t = AutoTokenizer.from_pretrained(
                    self.model.name_or_path,
                    use_fast=True
                )
                self._local_tokenizer = t
                print(f"[GRPOTrainerModule] Auto-loaded tokenizer: {self.model.name_or_path}")
            except:
                pass

    # ===============================================================
    #  Ensure pad token exists
    # ===============================================================
    def _ensure_pad_token(self):
        tok = self._local_tokenizer
        if tok is None:
            return

        if tok.pad_token is None:
            if tok.eos_token:
                tok.add_special_tokens({"pad_token": tok.eos_token})
                print("[GRPOTrainerModule] pad_token added = eos_token")

                if hasattr(self.model, "resize_token_embeddings"):
                    try:
                        self.model.resize_token_embeddings(len(tok))
                        print("[GRPOTrainerModule] Resized embeddings:", len(tok))
                    except:
                        print("[GRPOTrainerModule] resize_token_embeddings failed", file=sys.stderr)

        self._local_tokenizer = tok

    # ===============================================================
    #  SAFE GENERATE (NO multinomial, NO sampling)
    # ===============================================================
    def safe_generate(self, input_ids: torch.Tensor, **gkw):
        device = self.model.device
        input_ids = input_ids.to(device)

        # Hard-disable sampling
        gkw["do_sample"] = False
        gkw["top_k"] = 1
        gkw["top_p"] = 1.0
        gkw["temperature"] = 1.0
        gkw["num_beams"] = 1
        gkw["renormalize_logits"] = False
        gkw["use_cache"] = False

        if self._local_tokenizer:
            gkw.setdefault("pad_token_id", self._local_tokenizer.pad_token_id)
            gkw.setdefault("eos_token_id", self._local_tokenizer.eos_token_id)

        # Batch 1
        if input_ids.shape[0] == 1:
            return self.model.generate(input_ids, **gkw)

        # Batch >1 → generate one by one (safe for 4bit)
        outs = []
        for i in range(input_ids.shape[0]):
            one = input_ids[i:i+1]
            out = self.model.generate(one, **gkw)
            outs.append(out.cpu())

        # Pad to same length
        max_len = max(o.shape[1] for o in outs)
        pad_id = gkw.get("pad_token_id", 0)
        padded = []

        for o in outs:
            if o.shape[1] < max_len:
                pad = torch.full((1, max_len - o.shape[1]), pad_id, dtype=o.dtype)
                padded.append(torch.cat([o, pad], dim=1))
            else:
                padded.append(o)

        return torch.cat(padded, dim=0)

    # ===============================================================
    #  OVERRIDE genrl internal model generate
    # ===============================================================
    def _model_generate(self, input_ids, **gen_kwargs):
        """
        genrl always calls this to generate tokens.
        We override it to enforce safe greedy generation.
        """

        gen_kwargs["do_sample"] = False
        gen_kwargs["top_k"] = 1
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["num_beams"] = 1
        gen_kwargs["use_cache"] = False
        gen_kwargs["renormalize_logits"] = False

        if self._local_tokenizer:
            gen_kwargs.setdefault("pad_token_id", self._local_tokenizer.pad_token_id)
            gen_kwargs.setdefault("eos_token_id", self._local_tokenizer.eos_token_id)

        return self.safe_generate(input_ids, **gen_kwargs)

    # ===============================================================
    #  EVALUATE
    # ===============================================================
    @torch.no_grad()
    def evaluate(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager):
        if not self.judge_client:
            return

        model_name = getattr(self.model, "name_or_path", "unknown")

        result = self.judge_client.request_question(
            user_id=state.peer_id,
            round_number=state.round,
            model_name=model_name
        )
        if not result:
            return

        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": result["question"]},
        ]

        input_ids = self.processing_class.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        try:
            outputs = self.safe_generate(
                input_ids,
                max_new_tokens=self.args.max_new_tokens
            )
        except Exception as e:
            print("[GRPOTrainerModule] Generation failed:", e, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return

        try:
            if hasattr(self.processing_class, "decode"):
                answer = self.processing_class.decode(outputs[0], skip_special_tokens=True)
            elif self._local_tokenizer:
                answer = self._local_tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
            else:
                answer = outputs[0].tolist()
        except:
            answer = "[decode_error]"

        try:
            self.judge_client.submit_answer(
                session_id=result["session_id"],
                round_number=state.round,
                user_answer=answer
            )
        except Exception:
            print("[GRPOTrainerModule] submit_answer failed", file=sys.stderr)
