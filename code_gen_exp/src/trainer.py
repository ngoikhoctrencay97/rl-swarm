from typing import Any, List
import torch
import traceback
import sys
import re

from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer.grpo_trainer import GRPOLanguageTrainerModule

from code_gen_exp.src.utils.judge_client import JudgeClient
from code_gen_exp.src.solver_data import SYSTEM_PROMPTS

try:
    from transformers import AutoTokenizer
except:
    AutoTokenizer = None


# =============================================================
#               FINAL GRPO Trainer (FULL FIXED)
# =============================================================

class GRPOTrainerModule(GRPOLanguageTrainerModule, LoggerMixin):

    def __init__(self, models: List[Any], **kwargs):
        self._local_tokenizer = None

        # super() will call our overridden _initialize_model()
        super().__init__(models, **kwargs)

        # Judge service
        judge_base_url = kwargs.get("judge_base_url", None)
        self.judge_client = JudgeClient(judge_base_url) if judge_base_url else None

        self.system_prompt = SYSTEM_PROMPTS.get("solver", SYSTEM_PROMPTS["default"])

        # Load tokenizer
        self._init_tokenizer(kwargs)
        self._ensure_pad_token()

        print("[trainer] Init complete. FIXED for 4-bit + genrl.")


    # =========================================================
    #                MODEL INIT (block dtype cast)
    # =========================================================
    def _initialize_model(self, enable_gradient_checkpointing: bool):
        model_type = str(type(self.model)).lower()
        quantized = (
            hasattr(self.model, "is_quantized")
            or hasattr(self.model, "is_loaded_in_4bit")
            or "bnb" in model_type
            or "bitsandbytes" in model_type
        )

        if quantized:
            print("[trainer] 4-bit model detected → skipping dtype cast")

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

        # Normal model
        super()._initialize_model(enable_gradient_checkpointing)


    # =========================================================
    #                TOKENIZER LOAD + PAD FIX
    # =========================================================
    def _init_tokenizer(self, kwargs):
        tok = getattr(self.processing_class, "tokenizer", None)
        if tok:
            self._local_tokenizer = tok
            return

        tok = kwargs.get("tokenizer")
        if tok:
            self._local_tokenizer = tok
            return

        if AutoTokenizer and hasattr(self.model, "name_or_path"):
            try:
                tok = AutoTokenizer.from_pretrained(
                    self.model.name_or_path,
                    use_fast=True
                )
                self._local_tokenizer = tok
                print("[trainer] Auto-loaded tokenizer:", self.model.name_or_path)
            except:
                pass

    def _ensure_pad_token(self):
        tok = self._local_tokenizer
        if tok is None:
            return

        if tok.pad_token is None:
            if tok.eos_token:
                tok.add_special_tokens({"pad_token": tok.eos_token})
                print("[trainer] pad_token set = eos_token")

                if hasattr(self.model, "resize_token_embeddings"):
                    try:
                        self.model.resize_token_embeddings(len(tok))
                    except:
                        print("[trainer] resize embeddings failed")


    # =========================================================
    #       *** CRITICAL FIX: OVERRIDE _model_generate ***
    # =========================================================
    @torch.no_grad()
    def _model_generate(self, input_ids, attention_mask=None, **kwargs):
        """
        THIS is the method called by genrl.
        Safe GREEDY decoding to avoid multinomial / CUDA errors.
        """

        device = self.model.device
        input_ids = input_ids.to(device)

        if attention_mask is None:
            if self._local_tokenizer:
                pad_id = self._local_tokenizer.pad_token_id
                attention_mask = (input_ids != pad_id).long()
            else:
                attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(device)

        # Force safe generation
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.args.max_new_tokens),
            "do_sample": False,
            "num_beams": 1,
            "use_cache": False,
            "pad_token_id": self._local_tokenizer.pad_token_id if self._local_tokenizer else 0,
            "eos_token_id": self._local_tokenizer.eos_token_id if self._local_tokenizer else 1,
        }

        # Remove unsafe params
        unsafe = ["temperature", "top_k", "top_p", "typical_p",
                  "repetition_penalty", "penalty_alpha"]
        for k in unsafe:
            gen_kwargs.pop(k, None)

        batch = input_ids.shape[0]
        outputs = []

        # Generate one sample at a time → stable for 4-bit
        for i in range(batch):
            ids = input_ids[i:i+1]
            mask = attention_mask[i:i+1]

            try:
                out = self.model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    **gen_kwargs
                )
                outputs.append(out.cpu())
            except Exception as e:
                print(f"[trainer] _model_generate failed: {e}")
                outputs.append(ids.cpu())

        # pad all outputs to same length
        max_len = max(o.shape[1] for o in outputs)
        pad_id = gen_kwargs["pad_token_id"]

        padded = []
        for o in outputs:
            if o.shape[1] < max_len:
                pad = torch.full((1, max_len - o.shape[1]), pad_id, dtype=o.dtype)
                padded.append(torch.cat([o, pad], dim=1))
            else:
                padded.append(o)

        return torch.cat(padded, dim=0).to(device)


    # =========================================================
    #                   EVALUATE (SAFE)
    # =========================================================
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

        question = result.get("question", "")

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPTS["default"]},
            {"role": "user", "content": question}
        ]

        # tokenize input
        input_ids = self.processing_class.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = input_ids.to(self.model.device)

        # generate using fixed method
        try:
            gen = self._model_generate(
                input_ids,
                max_new_tokens=min(128, self.args.max_new_tokens)
            )
        except Exception as e:
            print("[trainer] evaluate generation failed:", e)
            return

        # decode output
        try:
            text = self.processing_class.decode(gen[0], skip_special_tokens=True)
        except:
            if self._local_tokenizer:
                text = self._local_tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)
            else:
                text = ""

        text = text.strip()
        print("[trainer] Raw model output:", text)

        # Extract clean answer
        ans_match = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
        if ans_match:
            final = ans_match[-1].strip()
        else:
            # Fallback: first non-empty line
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            final = lines[0] if lines else ""

        final = final.strip()
        print("[trainer] Final extracted answer:", final)

        # submit
        try:
            self.judge_client.submit_answer(
                session_id=result.get("session_id"),
                round_number=state.round,
                user_answer=final
            )
        except Exception as e:
            print("[trainer] submit_answer failed:", e)
