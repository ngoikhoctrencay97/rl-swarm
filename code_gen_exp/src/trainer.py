from typing import Any, List
import torch
from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer.grpo_trainer import GRPOLanguageTrainerModule
from code_gen_exp.src.utils.judge_client import JudgeClient
from code_gen_exp.src.solver_data import SYSTEM_PROMPTS

# ===== Thêm import cho lượng tử hóa 4-bit =====
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    raise ImportError(
        "Bạn cần cài transformers và bitsandbytes:\n"
        "pip install transformers bitsandbytes accelerate"
    )

# ==============================================

class GRPOTrainerModule(GRPOLanguageTrainerModule, LoggerMixin):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method.
    Implements the TrainerModule interface defined in base_trainer.py.
    """

    def __init__(self, models: List[Any], **kwargs):
        """
        Initialize the GRPO trainer module.

        Args:
            models: List containing the model to be trained or model name.
            **kwargs: Additional arguments for configuration.
        """
        super().__init__(models, **kwargs)
        if hasattr(self.model, "is_quantized") or "bnb" in str(type(self.model)).lower():
            # force genrl to skip dtype casting
            self.dtype = None
            self.args.dtype = None
        judge_base_url = kwargs.get("judge_base_url", None)
        self.judge_client = JudgeClient(judge_base_url) if judge_base_url else None
        self.system_prompt = SYSTEM_PROMPTS.get("solver", SYSTEM_PROMPTS["default"])

        # ==== Cấu hình lượng tử hóa 4-bit ====
        self.use_4bit = kwargs.get("use_4bit", True)
        self.bnb_quant_type = kwargs.get("bnb_quant_type", "nf4")
        self.compute_dtype = kwargs.get("compute_dtype", torch.float16)
        self.device_map = kwargs.get("device_map", "auto")

        # Nếu models[0] là tên model → tự động load model 4bit
        if isinstance(models[0], str):
            model_name = models[0]
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.use_4bit,
                bnb_4bit_quant_type=self.bnb_quant_type,
                bnb_4bit_compute_dtype=self.compute_dtype,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=self.device_map,
                trust_remote_code=True,
            )
            self.model.name_or_path = model_name  # để giữ tương thích với phần evaluate
        # =====================================

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
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": result["question"]},
        ]
        input_ids = self.processing_class.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Đảm bảo input_ids ở cùng device
        device = self.model.device if hasattr(self.model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = input_ids.to(device)

        # Dùng autocast để tương thích float16
        with torch.cuda.amp.autocast(enabled=(torch.cuda.is_available() and self.compute_dtype == torch.float16)):
            outputs = self.model.generate(input_ids, max_new_tokens=self.args.max_new_tokens)

        answer = self.processing_class.decode(outputs[0], skip_special_tokens=True)
        
        # Submit answer to judge service
        self.judge_client.submit_answer(
            session_id=result["session_id"],
            round_number=state.round,
            user_answer=answer
        )
