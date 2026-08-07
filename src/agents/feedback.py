# src/agents/feedback.py
"""Feedback agent that converts grading evidence into student-facing guidance."""

from typing import Dict, Any

from src.agents.agent_base import AgentBase
from src.schemas.feedback_schema import Result
from src.prompts.feedback_prompt import prompt_template
from src.utils.logging import Logger


# Feedback agent that turns grading outcomes into clear, explainable guidance
# for the student or reviewer.
class FeedbackAgent(AgentBase):
    """Translate grading evidence into concise, constructive feedback."""

    def __init__(self, model: str, temperature: float, max_tokens: int | None):
        super().__init__("Feedback Agent", model, temperature, max_tokens, [], (), Result, prompt_template)

    def skip(self, logger: Logger) -> Dict[str, Any]:
        """Return a placeholder feedback report when feedback generation is disabled."""
        if logger:
            logger.log("Feedback generation step skipped as per configuration.", level="warning", step_label="Feedback")

        return {
            "result_json": {"positive_highlights": ["<NO_FEEDBACK>"], "areas_for_improvement": ["<NO_FEEDBACK>"], "encouraging_closing": "<NO_FEEDBACK>"},
            "result_str": "Positive Highlights:\n- <NO_FEEDBACK>\n\nAreas for Improvement:\n- <NO_FEEDBACK>\n\n<NO_FEEDBACK>",
        }

    @staticmethod
    def parse_result(result: Result) -> str:
        """Render the feedback components as readable text for reports."""
        text = (
f"""Positive Highlights:
- {"\n- ".join(result.positive_highlights)}

Areas for Improvement:
- {"\n- ".join(result.areas_for_improvement)}

{result.encouraging_closing}"""
        )

        return text
