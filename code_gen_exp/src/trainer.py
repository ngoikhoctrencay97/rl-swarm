from typing import Any, List, Optional
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

# Optional: for tokenizers when needed
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


class GRPOTrainerModule(GRPOLanguageTrainerModule, LoggerMixin):
    """
    Custom GRPO Trainer with robust support for BitsAndBytes quantized models.
    - Override _initialize_model to skip dtype casting for bnb models.
    - Ensure pad_token exists (map to eos_token if missing).
    - Provide safe_generate to avoid batch-related CUDA asserts on quantized models.
    """

    def __init__(self, models: List[Any], **kwargs):
        """
        Note: we rely on overriding _initialize_model to skip dtype casting that genrl
        would otherwise perform during super().__init__.
        """
        # Keep a place to store tokenizer if available
        self._local_tokenizer = None

        # Call parent init (our _initialize_model will be used inside parent's init)
        super().__init__(models, **kwargs)

        # judge client
        judge_base_url = kwargs.get("judge_base_url", None)
        self.judge_client = JudgeClient(judge_base_url) if judge_base_url else None
        self.system_prompt = SYSTEM_PROMPTS.get("solver", SYSTEM_PROMPTS["default"])

        # Try to obtain tokenizer:
        # 1) from processing_class (common in your pipeline)
        # 2) from kwargs (maybe passed by Hydra)
        # 3) attempt to auto-load from model.name_or_path (best-effort)
        try:
            if getattr(self, "processing_class", None) is not None:
                tok = getattr(self.processing_class, "tokenizer", None)
                if tok is not None:
                    self._local_tokenizer = tok
            if self._local_tokenizer is None:
                # check kwargs
                tok_kw = kwargs.get("tokenizer", None)
                if tok_kw is not None:
                    self._local_tokenizer = tok_kw
            if self._local_tokenizer is None and getattr(self.model, "name_or_path", None) and AutoTokenizer is not None:
                try:
                    self._local_tokenizer = AutoTokenizer.from_pretrained(self.model.name_or_path, use_fast=True)
                except Exception:
                    # ignore failures to auto-load tokenizer
                    self._local_tokenizer = None
        except Exception:
            # be resilient
            self._local_tokenizer = None

        # Ensure pad token exists and embeddings adjusted if needed
        try:
            self._ensure_pad_token()
        except Exception:
            # don't crash trainer init due to tokenizer adjustments; just warn
            tb = traceback.format_exc()
            print("[GRPOTrainerModule] Warning while ensuring pad token:", tb, file=sys.stderr)

    # ---------------------------------------------------------------------
    # Override default genrl initialization to avoid dtype cast on bnb models
    # ---------------------------------------------------------------------
    def _initialize_model(self, enable_gradient_checkpointing: bool):
        """
        Override to bypass dtype casting for bitsandbytes quantized models.

        If the model is detected as quantized (bitsandbytes), we only move it to device
        without casting dtype; otherwise, call the original implementation.
        """
        try:
            model_type_str = str(type(self.model)).lower()
        except Exception:
            model_type_str = ""

        model_is_quantized = False
        try:
            # Common indicators of a bnb model
            model_is_quantized = (
                hasattr(self.model, "is_quantized")
                or "bnb" in model_type_str
                or "bitsandbytes" in model_type_str
            )
        except Exception:
            model_is_quantized = False

        if model_is_quantized:
            # Informative log
            print("[GRPOTrainerModule] Detected BitsAndBytes quantized model -> skipping dtype cast.")

            # Move model to device only (no dtype)
            try:
                # if model has device_map handling, .to(self.device) will be safe
                self.model = self.model.to(self.device)
            except Exception as e:
                # fallback: attempt lighter device set
                print("[GRPOTrainerModule] Warning: model.to(device) failed, exception:", e, file=sys.stderr)

            # disable use_cache / generation cache if present (helps with some quantized implementations)
            try:
                if hasattr(self.model.config, "use_cache"):
                    self.model.config.use_cache = False
            except Exception:
                pass

            # Enable gradient checkpointing if requested and supported
            if enable_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
                try:
                    self.model.gradient_checkpointing_enable()
                except Exception:
                    pass

            return

        # Non-quantized → fallback to original behavior
        super()._initialize_model(enable_gradient_checkpointing)

    # ---------------------------------------------------------------------
    # Tokenizer helpers
    # ---------------------------------------------------------------------
    def _ensure_pad_token(self):
        """
        Ensure tokenizer has a pad_token. If missing, set pad_token = eos_token.
        Resize model embeddings if tokenizer expanded.
        """
        tokenizer = self._local_tokenizer
        if tokenizer is None:
            # try processing_class.tokenizer again (maybe assigned later)
            tokenizer = getattr(self.processing_class, "tokenizer", None)

        if tokenizer is None:
            # nothing to do
            return

        # If tokenizer has no pad token, map pad->eos
        if getattr(tokenizer, "pad_token", None) is None:
            eos = getattr(tokenizer, "eos_token", None)
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos is not None:
                try:
                    tokenizer.add_special_tokens({"pad_token": eos})
                    print("[GRPOTrainerModule] Added pad_token mapped to eos_token for tokenizer.")
                    # if model supports resize embeddings, do it
                    if hasattr(self.model, "resize_token_embeddings"):
                        try:
                            old_embed_size = None
                            if hasattr(self.model, "get_input_embeddings") and self.model.get_input_embeddings() is not None:
                                old_embed_size = self.model.get_input_embeddings().weight.shape[0]
                            tokenizer_len = len(tokenizer)
                            # Only resize if increased
                            if old_embed_size is None or tokenizer_len != old_embed_size:
                                self.model.resize_token_embeddings(tokenizer_len)
                                print("[GRPOTrainerModule] Resized token embeddings to", tokenizer_len)
                        except Exception as e:
                            print("[GRPOTrainerModule] Warning: resize_token_embeddings failed:", e, file=sys.stderr)
                except Exception as e:
                    print("[GRPOTrainerModule] Warning: failed to add pad token to tokenizer:", e, file=sys.stderr)

        # store back
        self._local_tokenizer = tokenizer

    # ---------------------------------------------------------------------
    # Safe generation wrapper
    # ---------------------------------------------------------------------
    def safe_generate(self, input_ids: torch.Tensor, **generate_kwargs) -> torch.Tensor:
        """
        Safe wrapper around model.generate:
        - ensures pad_token_id/eos_token_id provided
        - disables cache/use_cache for quantized models
        - if batch_size>1, generate per-example to avoid CUDA device-side asserts
        """
        # Ensure device
        device = getattr(self.model, "device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_ids = input_ids.to(device)

        # Ensure tokenizer ids
        pad_id = None
        eos_id = None
        if self._local_tokenizer is None:
            self._ensure_pad_token()
        if self._local_tokenizer is not None:
            try:
                pad_id = getattr(self._local_tokenizer, "pad_token_id", None)
                eos_id = getattr(self._local_tokenizer, "eos_token_id", None)
            except Exception:
                pad_id = None
                eos_id = None

        # Defaults for generate kwargs
        if "pad_token_id" not in generate_kwargs and pad_id is not None:
            generate_kwargs["pad_token_id"] = pad_id
        if "eos_token_id" not in generate_kwargs and eos_id is not None:
            generate_kwargs["eos_token_id"] = eos_id

        # enforce no sampling by default (deterministic generation)
        generate_kwargs.setdefault("do_sample", False)

        # disable use_cache by default for quantized models (can cause issues)
        try:
            if hasattr(self.model.config, "use_cache") and self.model.config.use_cache:
                self.model.config.use_cache = False
            generate_kwargs.setdefault("use_cache", False)
        except Exception:
            generate_kwargs.setdefault("use_cache", False)

        # If batch > 1, generate per example to avoid problems with quantized models on some GPUs
        batch_size = input_ids.shape[0]
        if batch_size <= 1:
            # single-batch path
            return self.model.generate(input_ids, **generate_kwargs)

        # multi-batch path: iterate per sample (keeps memory low and avoids kernel asserts)
        outputs = []
        for i in range(batch_size):
            single_input = input_ids[i : i + 1, :].to(device)
            try:
                out = self.model.generate(single_input, **generate_kwargs)
                outputs.append(out.detach().cpu())
            except Exception as e:
                # if a single sample fails, log and re-raise with context
                tb = traceback.format_exc()
                print(f"[GRPOTrainerModule] Error while generating sample {i}: {e}\n{tb}", file=sys.stderr)
                raise
        # concatenate along batch dim (assumes returns tensor of token ids)
        # outputs is list of tensors of shape (1, seq_len_out) -> pad to same length
        max_len = max(o.shape[1] for o in outputs)
        padded = []
        for o in outputs:
            if o.shape[1] < max_len:
                pad_tensor = torch.full((1, max_len - o.shape[1]), generate_kwargs.get("pad_token_id", 0), dtype=o.dtype)
                padded.append(torch.cat([o, pad_tensor], dim=1))
            else:
                padded.append(o)
        return torch.cat(padded, dim=0)

    # ---------------------------------------------------------------------
    # Evaluation flow
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager):
        if not self.judge_client:
            return

        try:
            model_name = getattr(self.model, "name_or_path", "none")
        except Exception:
            model_name = "none"

        # Request question from judge
        result = self.judge_client.request_question(
            user_id=state.peer_id,
            round_number=state.round,
            model_name=model_name,
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
            return_tensors="pt",
        )

        # safe generate
        try:
            outputs = self.safe_generate(
                input_ids,
                max_new_tokens=self.args.max_new_tokens,
            )
        except Exception as e:
            # log and return gracefully
            tb = traceback.format_exc()
            print("[GRPOTrainerModule] Generation failed:", e, file=sys.stderr)
            print(tb, file=sys.stderr)
            return

        # decode first returned example (or handle full batch)
        try:
            # if processing_class has decode, use it; otherwise fallback to tokenizer
            if hasattr(self.processing_class, "decode"):
                answer = self.processing_class.decode(outputs[0], skip_special_tokens=True)
            elif self._local_tokenizer is not None:
                answer = self._local_tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
            else:
                answer = outputs[0].tolist()
        except Exception as e:
            answer = str(e)

        # Submit answer
        try:
            self.judge_client.submit_answer(
                session_id=result["session_id"],
                round_number=state.round,
                user_answer=answer,
            )
        except Exception:
            # don't crash evaluation loop on submission issues
            print("[GRPOTrainerModule] Warning: submit_answer failed", file=sys.stderr)
