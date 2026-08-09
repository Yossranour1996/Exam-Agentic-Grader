# other_architectures/zero_shot/core/state.py
"""Typed state definitions for the grading workflow and its runtime context."""

from __future__ import annotations

from typing import TypedDict, Required, Annotated, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from zero_shot.utils.logging import Logger


class InputState(TypedDict, total=False):
    """User-controlled inputs and optional stage switches for a single run."""

    sheet_id: Required[str]
    exam_file: str
    rubric_file: str

    do_save_results: bool
    do_export_to_sheet: bool


class OutputState(TypedDict):
    """Summary fields returned to callers after the workflow completes."""

    extract_dir: Path
    results_dir: Path
    log_path: Path
    grading_result: Dict[str, Any]
    grading_result_str : str

# Shared workflow state object that carries information between stages of the
# exam-grading pipeline, such as extracted content, grading results, and export data.
class State(InputState, OutputState):
    """Mutable state passed between workflow nodes and parallel branches."""

    # Run resources.
    run_id: str
    sheet_path: Path
    logger: Logger

    # Structured inputs.
    exam: Dict[str, Any]
    rubric: Dict[str, Any]

    student_answers: Dict[str, Any]

    messages: Annotated[List[AnyMessage], add_messages]


@dataclass(frozen=True)
class Context:
    """Stable directory configuration that stays outside the mutable graph state."""

    sheets_dir: Path
    exams_dir: Path
    criteria_dir: Path
    output_base: Path
    export_dir: Path
