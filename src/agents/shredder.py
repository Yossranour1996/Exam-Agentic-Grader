# src/agents/shredder.py
from typing import Dict, Any
from pathlib import Path

from src.agents.agent_base import AgentBase
from src.schemas.extractor_schemas import Result
from src.prompts.extractor_prompts import shredder_prompt_template
from src.utils.logging import Logger
from src.utils import io


class ShredderAgent(AgentBase):
    """Group raw page text under the canonical exam question IDs."""

    def __init__(self, model: str, temperature: float, max_tokens: int | None):
        super().__init__("Shredder Agent", model, temperature, max_tokens, None, None, Result, shredder_prompt_template)

    def skip(self, extract_dir: Path, logger: Logger | None) -> Dict[str, Any]:
        """Reuse the latest structured answer mapping."""
        if logger:
            logger.log("Shredder step skipped as per configuration.", level="warning", step_label="Extractor-Shredder")

        paths = sorted(extract_dir.glob('structured_answers*.json'))
        if not paths:
            raise FileNotFoundError(f"No prior structured answers in {extract_dir}")

        data = io.read_json(filepath=paths[-1], logger=logger)
        result = self.response_format(**data)

        return {
            "result_json": data,
            "result_str": self.parse_result(result)
        }

    @staticmethod
    def parse_result(result: Result) -> str:
        """Render mapped answers for saved extraction reports."""
        text = (
f"""STUDENT ANSWERS:

{"\n\n\n".join([(
    f'Question: {question.question}\n'
    f'{"\n\n".join([(
        f'Sub-Question: {sub_question.sub_question}\n'
        f'Answer: {sub_question.answer}'
    )for sub_question in question.sub_questions])}'
)for question in result.questions])}"""
        )

        return text
