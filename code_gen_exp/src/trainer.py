from typing import Any, List, Optional
import re
import sys
import traceback
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
    Loại C: giữ nguyên generate() của GenRL,
    chỉ override evaluate() để sanitize output an toàn hơn.
    """

    def __init__(self, models: List[Any], **kwargs):
        super().__init__(models, **kwargs)

        judge_base_url = kwargs.get("judge_base_url", None)
        self.judge_client = JudgeClient(judge_base_url) if judge_base_url else None
        self.system_prompt = SYSTEM_PROMPTS.get("solver", SYSTEM_PROMPTS["default"])


    # --------------------------
    # SANITIZER
    # --------------------------
    def _sanitize_answer(self, text: str, choices: Optional[List[str]] = None) -> str:
        """
        Loại bỏ lời giải thích, chỉ lấy nội dung câu trả lời.
        Ưu tiên <answer>...</answer>.
        """

        if not text:
            return ""

        # 1. Nếu có <answer>…</answer>, lấy cái cuối
        matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
        if matches:
            ans = matches[-1].strip()
            ans = re.sub(r"\s+", " ", ans)
            return ans

        # 2. Nếu có choices: cố match đúng lựa chọn
        if choices:
            low = text.lower()
            for c in choices:
                if c.lower() in low:
                    return c

        # 3. Lấy dòng đầu tiên không rỗng
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            first = lines[0]

            # Cắt prefix kiểu "Answer: ..."
            if ":" in first:
                first = first.split(":", 1)[1].strip()

            first = re.sub(r"\s+", " ", first).strip()
            return first

        return ""


    # --------------------------
    # OVERRIDE evaluate()
    # --------------------------
    @torch.no_grad()
    def evaluate(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager):
        if not self.judge_client:
            return

        try:
            model_name = getattr(self.model, "name_or_path", "none")
        except:
            model_name = "none"

        # Lấy câu hỏi từ Judge
        payload = self.judge_client.request_question(
            user_id=state.peer_id,
            round_number=state.round,
            model_name=model_name
        )

        if not payload:
            return

        question = payload["question"]
        choices = payload.get("choices", None)

        # Chuẩn bị prompt
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        # Tokenize bằng GenRL processing_class
        try:
            input_ids = self.processing_class.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
        except Exception:
            print("[GRPOTrainer] Failed apply_chat_template", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return

        # Gọi generate của GenRL (đã fix ở bạn)
        try:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.args.max_new_tokens
            )
        except Exception as e:
            print("[GRPOTrainer] generate failed:", e, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return

        # Decode
        try:
            raw = self.processing_class.decode(outputs[0], skip_special_tokens=True)
        except:
            try:
                raw = self.processing_class.tokenizer.decode(outputs[0])
            except:
                raw = ""

        raw = raw.strip()
        print("[GRPOTrainer] Raw:", raw)

        # Sanitize answer
        final_answer = self._sanitize_answer(raw, choices)
        print("[GRPOTrainer] Final:", final_answer)

        # Submit
        try:
            self.judge_client.submit_answer(
                session_id=payload["session_id"],
                round_number=state.round,
                user_answer=final_answer
            )
        except Exception as e:
            print("[GRPOTrainer] submit failed:", e, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
