# src/subgraphs/extraction_graph.py
"""Builds the extraction subgraph that turns scanned pages into structured student answers."""

from __future__ import annotations

import json
from pathlib import Path

from langgraph.graph import END, StateGraph

from src.agents.ocr import OCRAgent
from src.agents.shredder import ShredderAgent
from src.core.state import State
from src.tools.pdf_to_images import pdf_to_images
from src.utils.logging import Logger
from src.utils import io


class ExtractionGraph:
    """Convert pages, OCR them, and map the text to exam question IDs."""

    def __init__(self, ocr_model="gpt-4o", model="gpt-4o", temperature=0.0, max_tokens=None):
        self.ocr = OCRAgent(ocr_model, temperature, max_tokens)
        self.shredder = ShredderAgent(model, temperature, max_tokens)
        self.graph = self.compile()


    def node_pdf_to_images(self, state: State) -> State:
        """Create page images or reuse any images already present for the sheet."""
        logger = state['logger']
        if not state.get('do_pdf_to_imgs', True):
            print("Image extraction step skipped as per configuration.")
            logger.log("Image extraction step skipped as per configuration.", level="warning", step_label="Extractor-PDF-to-Images")
            pages = sorted(state['pages_dir'].glob('*.png'))
            return {
                "pages": pages,
                "page_index": 0
            }

        logger.log(f"Converting PDF {state['sheet_path']} to images: dpi={state.get('dpi', 300)}", level="info", step_label="Extractor-PDF-to-Images")

        pages = pdf_to_images(state['sheet_path'], state['pages_dir'], state.get('dpi', 300), logger)

        return {
            "pages": pages,
            "page_index": 0
        }


    def node_ocr(self, state: State) -> State:
        """Transcribe one page and advance the page cursor until all pages are processed."""
        logger = state['logger']
        if not state.get('do_ocr', True):
            print("OCR step skipped as per configuration.")
            output = self.ocr.skip(state['extract_dir'], logger)
            return {
                "ocr_pages": output['result_json'],
                "page_index": len(output['result_json']),
                "messages": []
            }

        index, page = state['page_index'], state['pages'][state['page_index']]

        logger.log(f"Extracting text from page {index + 1}/{len(state['pages'])}: {page.name}", level="info", step_label="Extractor-OCR")

        output = self.ocr.invoke({"page_b64": io.read_base64(page, logger), "logger": logger})

        io.write_text(filepath=state['extract_dir'] / f"{output['result_json']['page']}.txt", content=output['result_json']['text'], logger=logger)
        io.write_json(filepath=state['extract_dir'] / f"raw_extraction_{state['run_id']}.json", data=[output['result_json']], logger=logger)

        logger.log_messages(output['messages'], step_label="Extractor-OCR")
        return {
            "ocr_pages": [output['result_json']],
            "page_index": index + 1,
            "messages": output['messages']
        }


    def node_shredder(self, state: State) -> State:
        """Map the accumulated OCR text into the canonical structured answer schema."""
        logger = state['logger']
        if not state.get('do_shredder', True):
            print("Shredder step skipped as per configuration.")
            output = self.shredder.skip(state['extract_dir'], logger)
            return {
                "student_answers": output['result_json'],
                "student_answers_str": output['result_str'],
                "messages": []
            }

        logger.log("Mapping student answers to exam questions...", level="info", step_label="Extractor")

        output = self.shredder.invoke({
            "exam": json.dumps(state['exam'], ensure_ascii=False),
            "ocr_pages": json.dumps(state['ocr_pages'], ensure_ascii=False),
            "logger": logger
        })

        io.write_text(filepath=state['extract_dir'] / f"structured_answers_{state['run_id']}.txt", content=output['result_str'], logger=logger)
        io.write_json(filepath=state['extract_dir'] / f"structured_answers_{state['run_id']}.json", data=output['result_json'], logger=logger)

        logger.log_messages(output['messages'], step_label="Extractor-Shredder")
        return {
            "student_answers": output['result_json'],
            "student_answers_str": output['result_str'],
            "messages": output['messages']
        }


    def compile(self):
        """Build the page loop followed by one structuring pass for the extracted answers."""
        graph = StateGraph(State)

        graph.add_node("pdf_to_images", self.node_pdf_to_images)
        graph.add_node("ocr", self.node_ocr)
        graph.add_node("shredder", self.node_shredder)

        graph.set_entry_point("pdf_to_images")
        graph.add_edge("pdf_to_images", "ocr")
        graph.add_conditional_edges("ocr", lambda state: "ocr" if state['page_index'] < len(state['pages']) else "shredder")
        graph.add_edge("shredder", END)

        return graph.compile()

    def invoke(self, state: State) -> State:
        """Run extraction using the same wrapper interface as other subgraphs."""
        return self.graph.invoke(state)

    def skip(self, extract_dir: Path, logger: Logger | None) -> State:
        """Reuse the most recent structured extraction for this sheet."""
        if logger:
            logger.log("Extraction step skipped as per configuration.", level="warning", step_label="Extractor")

        paths = sorted(extract_dir.glob('structured_answers*.json'))
        if not paths:
            raise FileNotFoundError(f"No prior structured answers in {extract_dir}")
        data = io.read_json(filepath=paths[-1], logger=logger)
        result = self.shredder.response_format(**data)

        return {
            "student_answers": data,
            "student_answers_str": self.shredder.parse_result(result),
            "messages": []
        }
