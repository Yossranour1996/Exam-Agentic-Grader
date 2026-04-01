# src/state.py
from __future__ import annotations

from typing import TypedDict, Required, Annotated, List, Dict, Any
from dataclasses import dataclass

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.utils.config import Paths


# Graph state schema
class InputState(TypedDict, total=False):
	student_id: Required[str]
	
	dpi: int
	max_pages: int
	
	exam_file: str
	rubric_file: str

class OutputState(TypedDict):
	extract_dir: str
	export_dir: str
	log_path: str

	grader_result: str

	messages: Annotated[List[AnyMessage], add_messages]

class State(InputState, OutputState):
	pages: List[str]
	ocr_pages: List[Dict[str, Any]]

# Context schema
@dataclass
class Context(Paths):
	pass
