# src/graph/workflow.py
from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime

from src.state import InputState, OutputState, State, Context
from src.agents import extractor, grader, exporter
from src.tools.pdf_to_images import pdf_to_images
from src.utils import io, logging


# Global variable to hold the agents instances
grader_agent = None
exporter_agent = None


def node_pdf_to_images(state: InputState, runtime: Runtime[Context]) -> State:
    """Convert student's answer sheet PDF to images and update state."""

    student_id = state["student_id"]
    dpi = state.get("dpi", 300)
    answer_sheets_dir = runtime.context.answer_sheets_dir
    output_base = runtime.context.output_base

    student_pdf_path = answer_sheets_dir / f"{student_id}.pdf"
    pages_dir = output_base / student_id / "pages"

    # pages = pdf_to_images(
    #     pdf_path=student_pdf_path,
    #     out_dir=pages_dir,
    #     dpi=dpi
    #     )
    pages = [f"page_{i:02d}.png" for i in range(1, 11)]

    print("Converted", len(pages), "pages.")
    return {
        "pages": pages
        }


def node_gemini_ocr(state: State, runtime: Runtime[Context]) -> State:
    """Extract text from answer sheet pages using Gemini OCR and update state."""

    student_id = state["student_id"]
    max_pages = state.get("max_pages", 3)  # جرّبي أولاً 3 صفحات
    output_base = runtime.context.output_base

    extract_dir = output_base / student_id / "extracted_text"
    log_path = output_base / student_id / "messages_log.log"
    io.delete_file(log_path)  # Clear previous logs.

    # ocr_pages = extractor.extract_exam_pages(
    #     images=state.get("pages", []),
    #     out_dir=extract_dir,
    #     max_pages=max_pages
    #     )
    ocr_pages = io.read_json(extract_dir / "raw_extraction.json")

    print("Extracted text from", len(ocr_pages), "pages.")
    return {
        "extract_dir": str(extract_dir),
        "log_path": str(log_path),
        "ocr_pages": ocr_pages,
        }


def node_grader(state: State, runtime: Runtime[Context]) -> State:
    """Grade the extracted answers based on the rubric and update state with response."""

    exam_file = state.get("exam_file")
    rubric_file = state.get("rubric_file")
    exam_dir = runtime.context.exams_dir
    criteria_dir = runtime.context.criteria_dir

    exam = io.read_text(exam_dir / exam_file) if exam_file else None
    rubric = io.read_text(criteria_dir / rubric_file) if rubric_file else None
    
    print("\nGrading exam...")
    grader_output = grader_agent.invoke(
        input=grader.InputState({
            "student_answers": state.get("ocr_pages", "Problem getting answers"),
            "exam": exam or "Exam content unavailable",
            "rubric": rubric or "Standard evaluation criteria",
            }),
        )
    
    # Keep logs of LLM messages for traceability and debugging.
    logging.log_messages(
        grader_output["messages"],
        state.get("log_path", "grading_log.txt"),
        step_label="Grader"
        )
    
    print("Grading completed.")
    return {
        "grader_result": grader_output["grading_result"],
        "messages": grader_output["messages"]
        }


def node_exporter(state: State, runtime: Runtime[Context]) -> OutputState:
    """Export grading results to specified directory."""
    
    student_id = state["student_id"]
    out_dir = runtime.context.output_base

    export_dir = out_dir / student_id / "grading_results"

    print("\nExporting results...")
    exporter_agent.invoke(
        input=exporter.ExporterState({
            "grader_result": state["grader_result"],
            "export_dir": export_dir
            })
        )

    logging.logging.shutdown()  # Ensure all logs are flushed before program exit.
    print("Results exported.")
    return {"export_dir": str(export_dir)}


def build_graph(
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int | None = None,
):
    
    global grader_agent, exporter_agent
    grader_agent = grader.build_agent(model, temperature, max_tokens)
    exporter_agent = exporter.build_agent()

    g = StateGraph(
        State,
        input_schema=InputState,
        output_schema=OutputState,
        context_schema=Context
        )
    g.add_node("pdf_to_images", node_pdf_to_images)
    g.add_node("gemini_ocr", node_gemini_ocr)
    g.add_node("grader", node_grader)
    g.add_node("exporter", node_exporter)

    g.set_entry_point("pdf_to_images")
    g.add_edge("pdf_to_images", "gemini_ocr")
    g.add_edge("gemini_ocr", "grader")
    g.add_edge("grader", "exporter")
    g.add_edge("exporter", END)

    return g.compile()
