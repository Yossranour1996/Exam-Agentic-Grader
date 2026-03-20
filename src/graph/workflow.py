# src/graph/workflow.py
from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver

from src.state import InputState, OutputState, State, Context
from src.agents import extractor, grader, exporter
from src.tools.pdf_to_images import pdf_to_images
from src.utils import io, logging


def node_pdf_to_images(state: InputState, runtime: Runtime[Context]) -> State:
    """Convert student's answer sheet PDF to images and update state."""

    student_id = state["student_id"]
    dpi = state.get("dpi", 300)
    answer_sheets_dir = runtime.context.answer_sheets_dir
    output_base = runtime.context.output_base

    student_pdf_path = answer_sheets_dir / f"{student_id}.pdf"
    pages_dir = output_base / student_id / "pages"
    log_path = output_base / student_id / "logging.txt"
    log_path.unlink(missing_ok=True)

    pages = pdf_to_images(
        pdf_path=student_pdf_path,
        out_dir=pages_dir,
        dpi=dpi
        )
    # pages = ["page_01.png","page_02.png","page_03.png","page_04.png","page_05.png","page_06.png","page_07.png","page_08.png","page_09.png","page_10.png"] # For testing

    print("Converted", len(pages), "pages.")
    return state | {
        "log_path": str(log_path),
        "pages": pages
        }

def node_gemini_ocr(state: State, runtime: Runtime[Context]) -> State:
    """Extract text from answer sheet pages using Gemini OCR and update state."""

    student_id = state["student_id"]
    max_pages = state.get("max_pages", 3)  # جرّبي أولاً 3 صفحات
    output_base = runtime.context.output_base

    extract_dir = output_base / student_id / "extracted_text"

    ocr_pages = extractor.extract_exam_pages(
        images=state.get("pages", []),
        out_dir=extract_dir,
        max_pages=max_pages
        )
    # ocr_pages = [{"page": page, "text": "Test: Extractor"} for page in state["pages"]] # For testing
    
    print("Extracted text from", len(ocr_pages or []), "pages.")
    return state | {
        "extract_dir": str(extract_dir),
        "ocr_pages": ocr_pages,
        }

def node_grader(state: State, runtime: Runtime[Context]) -> State:
    """Grade the extracted answers based on the rubric and update state with response."""

    rubric_file = state.get("rubric_file", None)
    criteria_dir = runtime.context.rubric_dir
    
    rubric = io.read_text(criteria_dir / rubric_file) if rubric_file else None
    # rubric = "Test: Rubric" # For testing
    
    print("\nGrading exam...")
    grader_agent = grader.build_agent()
    grader_output = grader_agent.invoke(
        input=grader.InputState({
            "student_answers": state.get("ocr_pages", "Problem getting answers"),
            "rubric": rubric or "Standard evaluation criteria",
            }),
        context=grader.Context(
            model=runtime.context.model,
            temperature=runtime.context.temperature,
            max_tokens=runtime.context.max_tokens
            )
        )
    
    # Keep logs of LLM messages for traceability and debugging.
    logging.log(
        grader_output["messages"],
        state["log_path"],
        step_label="Grader"
        )
    
    print("Grading completed.")
    return state | {
        "grader_result": grader_output["grading_result"],
        "messages": [grader_output["messages"][-1]]
        }

def node_exporter(state: State, runtime: Runtime[Context]) -> OutputState:
    """Export grading results to specified directory."""
    
    student_id = state["student_id"]
    out_dir = runtime.context.output_base

    export_dir = out_dir / student_id / "grading_results"

    print("\nExporting results...")
    exporter_agent = exporter.build_agent()
    exporter_agent.invoke(
        input=exporter.ExporterState({
            "grader_result": state["grader_result"],
            "export_dir": export_dir
            })
        )

    print("Results exported.")
    return state | {"export_dir": str(export_dir)}


def build_graph():
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

    # memory = InMemorySaver()
    memory = None
    return g.compile(checkpointer=memory)
