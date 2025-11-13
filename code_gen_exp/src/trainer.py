from typing import Any, List
import torch

from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer.grpo_trainer import GRPOLanguageTrainerModule

from code_gen_exp.src.utils.judge_client import JudgeClient
from code_gen_exp.src.solver_data import SYSTEM_PROMPTS


class GRPOTrainerModule(GRPOLanguageTrainerModule, LoggerMixin):
    """
    Custom GRPO Trainer for code-generation tasks.
    This version fully supports BitsAndBytes quantized models (4-bit / 8-bit)
    without modifying the genrl library.
    """

    def __init__(self, models: List[Any], **kwargs):
        """
        Initialize trainer.
        We do NOT patch dtype here — dtype casting happens too early in genrl,
        so we override _initialize_model instead.
        """
        self.judge_client = None
        super().__init__(models, **kwargs)

        judge_base_url = kwargs.get("judge_base_url", None)
        self.judge_client = JudgeClient(judge_base_url) if judge_base_url else None
        self.system_prompt = SYSTEM_PROMPTS.get("solver", SYSTEM_PROMPTS["default"])

    # ===============================================================
    # 🚀 KEY FIX: OVERRIDE _initialize_model TO DISABLE DTYPE CAST
    # ===============================================================
    def _initialize_model(self, enable_gradient_checkpointing: bool):
        """
        Override genrl's model init to bypass dtype casting for quantized models.
        """
        model_is_quantized = (
            hasattr(self.model, "is_quantized")
            or "bnb" in str(type(self.model)).lower()
            or "bitsandbytes" in str(type(self.model)).lower()
        )

        if model_is_quantized:
            print("[GRPOTrainerModule] Detected quantized BitsAndBytes model → skipping dtype cast.")

            # Move to device only (BitsAndBytes does not allow .to(dtype=...))
            self.model = self.model.to(self.device)

            # Enable gradient checkpointing if requested
            if enable_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()

            return  # skip parent's dtype cast logic

        # otherwise use original behavior
        super()._initialize_model(enable_gradient_checkpointing)

    # ===============================================================
    #  Evaluation / Query logic
    # ===============================================================
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

        # Request question
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
            return_tensors="pt",
        )

        device = self.model.device
        input_ids = input_ids.to(device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.args.max_new_tokens
        )

        answer = self.processing_class.decode(
            outputs[0], skip_special_tokens=True
        )

        # Submit result
        self.judge_client.submit_answer(
            session_id=result["session_id"],
            round_number=state.round,
            user_answer=answer
        )
