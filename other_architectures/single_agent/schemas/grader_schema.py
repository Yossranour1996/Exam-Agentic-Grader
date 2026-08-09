# other_architectures/single_agent/schemas/grader_schema.py
"""Structured schema for the detailed grading output produced per question."""

from typing import List
from pydantic import BaseModel, Field


# Schema for structured output
class SubQuestionEvaluation(BaseModel):
    """Grading for a single sub-question."""
    sub_question: str = Field(description="The string identifier of the sub-question being scored.")
    awarded_points: int = Field(description="The integer value representing the assigned score.", ge=0)
    justification: str = Field(description="The textual rationale for the assigned score.")

class QuestionEvaluation(BaseModel):
    """Detailed breakdown for a single question."""
    question: str = Field(description="The string identifier of the parent question.")
    total_points: int = Field(description="The arithmetic sum of all points awarded in the sub-evaluations.", ge=0)
    sub_evaluations: List[SubQuestionEvaluation] = Field(description="The list of score breakdowns for each associated sub-question.")

class Result(BaseModel):
    """Comprehensive grading result for a single question."""
    total_mark: int = Field(description="The total student mark for the exam, which is the sum of all question points.", ge=0)
    evaluations: List[QuestionEvaluation] = Field(description="The list of detailed evaluations for each question and its sub-questions.")