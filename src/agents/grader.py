# src/agents/grader.py
"""Grading agent used to score exam answers question by question."""

from src.agents.agent_base import AgentBase
from src.tools.grader_tools import (
    calculate_total_score, fuzzy_keyword_match, check_java_syntax, force_tools
)
from src.schemas.grader_schema import Result
from src.prompts.grader_prompt import prompt_template


# Grading agent that evaluates extracted answers against the expected rubric or
# reference output and produces a score or feedback signal.
class GraderAgent(AgentBase):
    """Grade one question and its sub-questions in isolation with section-specific tools."""

    def __init__(self, model: str, temperature: float, max_tokens: int | None, section: str):
        tools = {
            "Q1": [calculate_total_score],
            "Q2": [calculate_total_score],
            "Q3": [calculate_total_score, fuzzy_keyword_match],
            "Q4": [calculate_total_score, check_java_syntax]
        }[section]
        middleware = {
            "Q1": [force_tools],
            "Q2": [force_tools],
            "Q3": [force_tools],
            "Q4": [force_tools]
        }[section]
        super().__init__(f"Grader Agent {section}", model, temperature, max_tokens, tools, middleware, Result, prompt_template)

    @staticmethod
    def parse_result(result: Result) -> str:
        """Render one question and its sub-question evidence as readable text."""
        text = (
f""""Question: {result.question}
Total Points: {result.total_points}

{"\n\n".join([(
    f'Sub-Question: {sub_evaluation.sub_question}\n'
    f'Awarded Points: {sub_evaluation.awarded_points}\n'
    f'Justification: {sub_evaluation.justification}'
)for sub_evaluation in result.sub_evaluations])}"""
        )

        return text