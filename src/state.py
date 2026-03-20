# src/state.py
from __future__ import annotations

from typing import TypedDict, Required, Annotated, List, Dict, Any
from dataclasses import dataclass, field

from langchain.messages import AIMessage
from langgraph.graph.message import add_messages

from src.utils.config import Paths


# Graph state schema
class InputState(TypedDict, total=False):
    student_id: Required[str]

    dpi: int
    max_pages: int

    rubric_file: str

class OutputState(TypedDict):
    extract_dir: str
    export_dir: str
    log_path: str

    grader_result: Dict[str, Any]

    messages: Annotated[List[AIMessage], add_messages]

class State(InputState, OutputState):
    pages: List[str]
    ocr_pages: List[Dict[str, Any]]

# Context schema
@dataclass
class Context(Paths):
	model: str = field(default="gpt-4o")
	temperature: float = field(default=0.0)
	max_tokens: int | None = field(default=None)
