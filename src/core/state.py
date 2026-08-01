# src/core/state.py
from __future__ import annotations

from typing import TypedDict, Required, Annotated, List, Dict, Any
from dataclasses import dataclass
from operator import add
from pathlib import Path

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.utils.logging import Logger


class InputState(TypedDict, total=False):
    """User-controlled inputs and optional stage switches."""

    sheet_id: Required[str]
    exam_file: str
    rubric_file: str
    review_criteria_file: str
    dpi: int
    max_regrade: int
    do_extract: bool
    do_pdf_to_imgs: bool
    do_ocr: bool
    do_shredder: bool
    do_grade: bool
    do_qa: bool
    do_feedback: bool
    do_export: bool
    do_save_results: bool
    do_export_to_sheet: bool


class OutputState(TypedDict):
    """Small public result returned to callers."""

    extract_dir: Path
    reports_dir: Path
    results_dir: Path
    log_path: Path
    student_answers_str: str
    grading_result_str: str
    qa_result_str: str
    feedback_result_str: str


class State(InputState, OutputState):
    """Only data exchanged between workflow stages or parallel branches."""

    # Run resources.
    run_id: str
    sheet_path: Path
    pages_dir: Path
    logger: Logger

    # Structured inputs.
    exam: Dict[str, Any]
    rubric: Dict[str, Any]
    review_criteria: Dict[str, Any]

    # extraction data.
    pages: List[Path]
    page_index: int
    ocr_pages: Annotated[List[Dict[str, str]], add]
    student_answers: Dict[str, Any]

    # Grading and selective-regrade state.
    grading_result: Dict[str, Any]
    qa_result: Dict[str, Any]
    regrade: bool
    regrade_counter: int
    regrade_question_ids: List[str]

    # Final feedback and trace data.
    feedback_result: Dict[str, Any]
    messages: Annotated[List[AnyMessage], add_messages]


class QuestionTask(TypedDict):
    """Minimal isolated payload sent to one parallel grader."""

    id: str
    do_grade: bool
    question: str
    rubric: str
    student_answers: str
    review: str
    evaluation: Dict[str, Any]
    logger: Logger


@dataclass(frozen=True)
class Context:
    """Stable directory configuration kept outside mutable graph state."""

    sheets_dir: Path
    exams_dir: Path
    criteria_dir: Path
    output_base: Path
    export_dir: Path
