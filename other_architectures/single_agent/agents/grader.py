# other_architectures/single_agent/agents/grader.py
"""Grading agent used to score exam answers question by question."""

from single_agent.agents.agent_base import AgentBase
from single_agent.tools.grader_tools import (
    calculate_total_score, fuzzy_keyword_match, check_java_syntax, force_tools
)
from single_agent.schemas.grader_schema import Result
from single_agent.prompts.grader_prompt import prompt_template


# Grading agent that evaluates extracted answers against the expected rubric or
# reference output and produces a score or feedback signal.
class GraderAgent(AgentBase):
    """Grade one question and its sub-questions in isolation with section-specific tools."""

    def __init__(self, model: str, temperature: float, max_tokens: int | None):
        super().__init__(f"Grader Agen", model, temperature, max_tokens, [calculate_total_score, check_java_syntax], None, Result, prompt_template)

    @staticmethod
    def parse_result(result: Result) -> str:
        """Render one question and its sub-question evidence as readable text."""
        text = (
f""""Total Mark: {result.total_mark}

Evaluations:
{"\n\n\n".join([(
     f'Question: {eval.question}\n'
     f'Total Points: {eval.total_points}\n\n'
     f'{"\n\n".join([(
          f'Sub-Question: {sub_evaluation.sub_question}\n'
          f'Awarded Points: {sub_evaluation.awarded_points}\n'
          f'Justification: {sub_evaluation.justification}'
    )for sub_evaluation in eval.sub_evaluations])}'
)for eval in result.evaluations])}"""
        )

        return text