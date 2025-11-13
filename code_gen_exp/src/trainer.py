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
    CRITICAL FIX: Override generate() method to prevent multinomial sampling errors
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
    #  CRITICAL: Override generate() - this is what genrl calls!
    # ===============================================================
    @torch.no_grad()
    def generate(self, prompts, **kwargs):
        """
        CRITICAL OVERRIDE: This is the method genrl.trainer.grpo_trainer calls!
        We must intercept it here to prevent sampling errors.
        
        Args:
            prompts: Can be a Dataset, list of dicts, or tensor
        """
        print(f"[DEBUG] prompts type: {type(prompts)}")
        print(f"[DEBUG] prompts dir: {dir(prompts)[:10]}")  # First 10 attributes
        if hasattr(prompts, '__len__'):
            print(f"[DEBUG] prompts length: {len(prompts)}")
        if hasattr(prompts, '__getitem__'):
            try:
                print(f"[DEBUG] prompts[0] type: {type(prompts[0])}")
                print(f"[DEBUG] prompts[0]: {prompts[0]}")
            except:
                pass
        print(f"[GRPOTrainerModule] generate() called, type: {type(prompts)}")
        
        # Handle different input types
        if not isinstance(prompts, (list, tuple, torch.Tensor)):
            # It's a Dataset or similar iterable
            try:
                # Try to get the actual prompts data
                if hasattr(prompts, 'data'):
                    prompts = prompts.data
                elif hasattr(prompts, 'dataset'):
                    prompts = prompts.dataset
                elif hasattr(prompts, '__iter__'):
                    # Convert to list
                    prompts = list(prompts)
                else:
                    raise TypeError(f"Cannot process prompts of type {type(prompts)}")
            except Exception as e:
                print(f"[GRPOTrainerModule] Error extracting prompts: {e}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                raise
        
        print(f"[GRPOTrainerModule] Processing {len(prompts) if hasattr(prompts, '__len__') else '?'} prompts")
        
        # Get input_ids from prompts
        if isinstance(prompts, (list, tuple)):
            # Prompts are chat messages or already processed
            input_ids_list = []
            for prompt in prompts:
                # Check if already tokenized
                if isinstance(prompt, torch.Tensor):
                    input_ids_list.append(prompt.unsqueeze(0) if prompt.dim() == 1 else prompt)
                elif isinstance(prompt, dict):
                    # Chat format
                    ids = self.processing_class.apply_chat_template(
                        prompt if isinstance(prompt, list) else [prompt],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )
                    input_ids_list.append(ids)
                else:
                    # Assume it's a message list
                    ids = self.processing_class.apply_chat_template(
                        prompt,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )
                    input_ids_list.append(ids)
            
            # Pad to same length
            max_len = max(ids.shape[1] for ids in input_ids_list)
            pad_id = self._local_tokenizer.pad_token_id if self._local_tokenizer else 0
            
            padded = []
            for ids in input_ids_list:
                if ids.shape[1] < max_len:
                    pad = torch.full((1, max_len - ids.shape[1]), pad_id, dtype=ids.dtype)
                    padded.append(torch.cat([ids, pad], dim=1))
                else:
                    padded.append(ids)
            
            input_ids = torch.cat(padded, dim=0)
        elif isinstance(prompts, torch.Tensor):
            input_ids = prompts
        else:
            raise TypeError(f"Unsupported prompts type: {type(prompts)}")

        # Move to device
        device = self.model.device
        input_ids = input_ids.to(device)
        
        # Create attention mask
        if self._local_tokenizer and self._local_tokenizer.pad_token_id is not None:
            attention_mask = (input_ids != self._local_tokenizer.pad_token_id).long()
        else:
            attention_mask = torch.ones_like(input_ids)

        # FORCE greedy decoding - remove ALL sampling parameters
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": kwargs.get("max_new_tokens", self.args.max_new_tokens),
            "do_sample": False,
            "num_beams": 1,
            "use_cache": False,
            "pad_token_id": self._local_tokenizer.pad_token_id if self._local_tokenizer else 0,
            "eos_token_id": self._local_tokenizer.eos_token_id if self._local_tokenizer else 1,
        }

        # Remove any dangerous parameters
        dangerous_keys = ['temperature', 'top_k', 'top_p', 'renormalize_logits', 
                         'typical_p', 'penalty_alpha', 'repetition_penalty']
        for key in dangerous_keys:
            gen_kwargs.pop(key, None)

        print(f"[GRPOTrainerModule] Generating with greedy decoding (batch_size={input_ids.shape[0]})")

        # Generate one by one for safety with 4-bit models
        if input_ids.shape[0] == 1:
            try:
                outputs = self.model.generate(**gen_kwargs)
                return outputs
            except Exception as e:
                print(f"[GRPOTrainerModule] Generation failed: {e}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                # Fallback: return input
                return input_ids

        # Batch > 1: generate one by one
        all_outputs = []
        for i in range(input_ids.shape[0]):
            single_kwargs = gen_kwargs.copy()
            single_kwargs["input_ids"] = input_ids[i:i+1]
            single_kwargs["attention_mask"] = attention_mask[i:i+1]
            
            try:
                output = self.model.generate(**single_kwargs)
                all_outputs.append(output.cpu())
            except Exception as e:
                print(f"[GRPOTrainerModule] Generation failed for item {i}: {e}", file=sys.stderr)
                # Fallback: use input
                all_outputs.append(input_ids[i:i+1].cpu())

        # Pad outputs to same length
        max_len = max(o.shape[1] for o in all_outputs)
        pad_id = gen_kwargs["pad_token_id"]
        padded_outputs = []
        
        for o in all_outputs:
            if o.shape[1] < max_len:
                pad = torch.full((1, max_len - o.shape[1]), pad_id, dtype=o.dtype)
                padded_outputs.append(torch.cat([o, pad], dim=1))
            else:
                padded_outputs.append(o)

        return torch.cat(padded_outputs, dim=0).to(device)

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

        try:
            # Use our safe generate method
            outputs = self.generate(
                [prompt],
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
                answer = "[decode_error]"
        except Exception as e:
            print(f"[GRPOTrainerModule] Decode failed: {e}", file=sys.stderr)
            answer = "[decode_error]"

        try:
            self.judge_client.submit_answer(
                session_id=result["session_id"],
                round_number=state.round,
                user_answer=answer
            )
        except Exception as e:
            print(f"[GRPOTrainerModule] submit_answer failed: {e}", file=sys.stderr)
