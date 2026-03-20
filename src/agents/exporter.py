# src/agents/exporter.py
from __future__ import annotations

from pathlib import Path
from typing import TypedDict, Dict, Any

from langgraph.graph import StateGraph, END

from src.utils import io 


# Exporter graph state schema
class ExporterState(TypedDict):
	grader_result: Dict[str, Any]
	export_dir: Path


def export(state: ExporterState) -> ExporterState:
	"""Export grading results to disk for later analysis.

	This exporter writes per-attempt logs, then final overall grading and feedback results."""

	grader_result = state["grader_result"]
	export_dir = Path(state["export_dir"])
	
	export_dir.mkdir(parents=True, exist_ok=True)
	export_path = export_dir / "grading_result.txt"
	export_path.unlink(missing_ok=True)
	io.write_text(
		export_path,
		"Grading Result:\n" + 
		"".join([f"\t{k}: {v}\n" for k,v in grader_result.items()])
		)
	
	return state


def build_agent():
	graph = StateGraph(ExporterState)
	graph.add_node("export", export)
	graph.set_entry_point("export")
	graph.add_edge("export", END)

	return graph.compile()
