# src/graph/workflow.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, Optional, List, Dict, Any
from pathlib import Path

from langgraph.graph import StateGraph, END

from src.tools.pdf_to_images import pdf_to_images
from src.agents.extractor import extract_exam_pages

class ExtractState(TypedDict, total=False):
    student_id: str
    student_pdf: str
    dpi: int
    max_pages: int

    pages_dir: str
    extracted_dir: str

    ocr_pages: List[Dict[str, Any]]

def node_pdf_to_images(state: ExtractState) -> ExtractState:
    student_id = state["student_id"]
    out_base = Path("data/output") / student_id
    pages_dir = out_base / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    pdf_to_images(state["student_pdf"], str(pages_dir), dpi=state.get("dpi", 300))

    state["pages_dir"] = str(pages_dir)
    state["extracted_dir"] = str(out_base / "extracted_text")
    return state

def node_gemini_ocr(state: ExtractState) -> ExtractState:
    pages_dir = state["pages_dir"]
    extracted_dir = state["extracted_dir"]
    max_pages = state.get("max_pages", 3)  # جرّبي أولاً 3 صفحات

    ocr_pages = extract_exam_pages(pages_dir=pages_dir, out_dir=extracted_dir, max_pages=max_pages)
    state["ocr_pages"] = ocr_pages
    return state

def build_extraction_graph():
    g = StateGraph(ExtractState)
    g.add_node("pdf_to_images", node_pdf_to_images)
    g.add_node("gemini_ocr", node_gemini_ocr)

    g.set_entry_point("pdf_to_images")
    g.add_edge("pdf_to_images", "gemini_ocr")
    g.add_edge("gemini_ocr", END)

    return g.compile()
