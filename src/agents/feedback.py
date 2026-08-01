# src/agents/feedback.py
from typing import Dict, Any
from pathlib import Path

from src.agents.agent_base import AgentBase
from src.schemas.feedback_schema import Result
from src.prompts.feedback_prompt import prompt_template
from src.utils.logging import Logger


class FeedbackAgent(AgentBase):
    """Translate grading evidence into concise constructive feedback."""

    def __init__(self, model: str, temperature: float, max_tokens: int | None):
        super().__init__("Feedback Agent", model, temperature, max_tokens, [], (), Result, prompt_template)

    def skip(self, logger: Logger) -> Dict[str, Any]:
        """Reuse the most recent feedback when generation is disabled."""
        if logger:
            logger.log("Feedback generation step skipped as per configuration.", level="warning", step_label="Feedback")

        data = {
            "positive_highlights": ["<NO_FEEDBACK>"],
            "areas_for_improvement": ["<NO_FEEDBACK>"],
            "encouraging_closing": "<NO_FEEDBACK>"
        }
        result = self.response_format(**data)

        return {
            "result_json": data,
            "result_str": self.parse_result(result)
        }

    @staticmethod
    def parse_result(result: Result) -> str:
        """Render the three feedback components as readable text."""
        text = (
f"""Positive Highlights:
- {"\n- ".join(result.positive_highlights)}

Areas for Improvement:
- {"\n- ".join(result.areas_for_improvement)}

{result.encouraging_closing}"""
        )

        return text
