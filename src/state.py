# src/state.py
from __future__ import annotations

from typing import TypedDict, Required, Annotated, Literal, List, Dict, Any
from operator import add
from dataclasses import dataclass

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.utils.config import Paths


# Graph state schema
class InputState(TypedDict, total=False):
	student_id: Required[str]

	dpi: int
	max_pages: int
	max_regrade: int
	
	exam_file: str
	rubric_file: str
	review_criteria_file: str

class OutputState(TypedDict):
	extract_dir: str
	export_dir: str
	log_path: str

	grader_results: Annotated[List[str], add]
	qa_results: Annotated[List[str], add]
	feedback_result: str

	messages: Annotated[List[AnyMessage], add_messages]

class State(InputState, OutputState):
	pages: List[str]
	ocr_pages: List[Dict[str, Any]]

	regrade: Literal["<REGRADE_REQUIRED>", "<NO_REGRADE>"]
    
	grader_messages: Annotated[List[AnyMessage], add_messages]
	qa_messages: Annotated[List[AnyMessage], add_messages]

# Context schema
@dataclass
class Context(Paths):
	pass
