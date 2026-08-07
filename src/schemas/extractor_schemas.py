# src/schemas/extractor_schemas.py
"""Structured schemas for OCR extraction and answer mapping results."""

from typing import List
from pydantic import BaseModel, Field


# Schemas for structured output
# OCR structured result
class PageExtraction(BaseModel):
    """Structured output for page text extraction."""
    page: str = Field(description="The original filename string of the input image.")
    text: str = Field(description="The transcribed text string from the page image.")

# Shredder structured result
class SubQuestionBundle(BaseModel):
    """Q/A mapping for a sub-question."""
    sub_question: str = Field(description="The string identifier of the sub-question.")
    answer: str = Field(description="The text block containing the student's answer.")

class QuestionBundle(BaseModel):
    """Q/A mapping for a single question."""
    question: str = Field(description="The string identifier of the parent question.")
    sub_questions: List[SubQuestionBundle] = Field(description="A list mapping sub-questions to their corresponding answers.")

class Result(BaseModel):
    """Structured output for exam Q/A structuring."""
    questions: List[QuestionBundle] = Field(description="The complete list of mapped parent questions.")
