# src/agents/qa.py
"""QA agent that audits grading decisions and flags questions that need a regrade."""

from typing import Dict, Any

from src.agents.agent_base import AgentBase
from src.tools.qa_tools import (
	calculate_partial_credit, validate_score_constraints, force_tools
)
from src.schemas.qa_schema import Result
from src.prompts.qa_prompt import prompt_template
from src.utils.logging import Logger


# QA-oriented agent that interprets the extracted content and organizes it into
# question/answer structures that later stages can grade.
class QAAgent(AgentBase):
    """Audit rubric adherence and identify question IDs that need correction."""

    def __init__(self, model: str, temperature: float, max_tokens: int | None):
        super().__init__("QA Agent", model, temperature, max_tokens, [validate_score_constraints, calculate_partial_credit], [force_tools], Result, prompt_template)

    def skip(self, logger: Logger) -> Dict[str, Any]:
        """Return a no-regrade result when QA is disabled for the run."""
        if logger:
            logger.log("Quality assurance step skipped as per configuration.", level="warning", step_label="Quality-Assurance")

        return {
            "result_json": {"decision": "<NO_REGRADE>", "audits": []},
            "result_str": "Regrade Decision: <NO_REGRADE>\nAudits:\n",
        }

    @staticmethod
    def parse_result(result: Result) -> str:
        """Render QA findings in a form that can guide a targeted regrade."""
        text = (
f"""Regrade Decision: {result.decision}
Audits:
{"\n\n".join([(
    f'Question: {audit.question}\n'
    f'{"\n\n".join([(
        f'Sub-Question: {error.sub_question}\n'
        f'Description: {error.description}'
    )for error in audit.identified_errors])}'
)for audit in result.audits])}"""
        )

        return text
