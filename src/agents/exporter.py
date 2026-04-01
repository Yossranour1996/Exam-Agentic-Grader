# src/agents/exporter.py
from __future__ import annotations

from pathlib import Path
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from src.utils import io 


# Exporter graph state schema
class ExporterState(TypedDict):
	grader_results: List[str]
	qa_results: List[str]
	feedback_result: str
	export_dir: str | Path


def export(state: ExporterState) -> ExporterState:
	"""Export grading results to disk for later analysis.

	This exporter writes per-attempt logs, then final overall grading and feedback results.
	"""

	grader_results = state["grader_results"]
	qa_results = state["qa_results"]
	feedback_result = state["feedback_result"]
	export_dir = Path(state["export_dir"])
	
	export_dir.mkdir(parents=True, exist_ok=True)
	export_path = export_dir / "grading_attempts.txt"
	io.delete_file(export_path)  # Clear previous attempts log
	
	for i in range(len(grader_results)):
		io.write_text(
			export_path,
			(
f"""Attempt {i+1}:
 
Grading:
{grader_results[i]}

Review: 
{qa_results[i]}
{'-' * 80}

"""
		)
	)
	
	export_path = export_dir / "final_result.txt"
	io.delete_file(export_path)  # Clear previous result
	io.write_text(
		export_path,
		(
f"""Grading:
{grader_results[-1]}
{'-' * 80}

Feedback: 
{feedback_result}
"""
		),
	)

	return state


def build_agent():
	graph = StateGraph(ExporterState)
	graph.add_node("export", export)
	graph.set_entry_point("export")
	graph.add_edge("export", END)

	return graph.compile()
