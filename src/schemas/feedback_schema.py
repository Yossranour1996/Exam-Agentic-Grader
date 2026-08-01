# src/schemas/feedback_schema.py
from typing import List
from pydantic import BaseModel, Field


# Schema for structured output to use in feedback
class Result(BaseModel):
    """Structured output for the Pedagogical Feedback Agent."""
    positive_highlights: List[str] = Field(description="A list of statements detailing correct concepts from the grading data.")
    areas_for_improvement: List[str] = Field(description="A list of statements detailing the reasons for point deductions.")
    encouraging_closing: str = Field(description="A concluding textual statement for the report.")
