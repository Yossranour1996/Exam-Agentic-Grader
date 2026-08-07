# src/schemas/qa_schema.py
"""Structured schema for the QA agent's audit findings and regrade decision."""

from typing import List, Literal
from pydantic import BaseModel, Field


# Schema for structured output to use in QA
class Error(BaseModel):
    """Detailed breakdown of a single error identified in the Grader's evaluation."""
    sub_question: str = Field(description="The string identifier of the sub-question containing the evaluation error.")
    description: str = Field(description="The textual explanation of the identified error.")

class QuestionAudit(BaseModel):
    question: str = Field(description="The string identifier of the parent question containing evaluation errors.")
    identified_errors: List[Error] = Field(description="The list of error records associated with this parent question, if any.")
    summary: str = Field(description="The concluding textual remarks regarding the audit of this question. Should also log any arithmetic errors in score calculation.")

class Result(BaseModel):
    """Structured output for the QA Auditor Agent."""
    decision: Literal["<REGRADE_REQUIRED>", "<NO_REGRADE>"] = Field(description="The final outcome state string of the audit process.")
    audits: List[QuestionAudit] = Field(description="The collection of question-level audit records of only parent questions that contain errors.")
