# src/schemas/grader_schema.py
from typing import List
from pydantic import BaseModel, Field


# Schema for structured output
class SubQuestionEvaluation(BaseModel):
    """Grading for a single sub-question."""
    sub_question: str = Field(description="The string identifier of the sub-question being scored.")
    awarded_points: int = Field(description="The integer value representing the assigned score.", ge=0)
    justification: str = Field(description="The textual rationale for the assigned score.")

class Result(BaseModel):
    """Detailed breakdown for a single question."""
    question: str = Field(description="The string identifier of the parent question.")
    total_points: int = Field(description="The arithmetic sum of all points awarded in the sub-evaluations.", ge=0)
    sub_evaluations: List[SubQuestionEvaluation] = Field(description="The list of score breakdowns for each associated sub-question.")
