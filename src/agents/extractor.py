# src/agents/extractor.py
from __future__ import annotations

from pathlib import Path
from typing import TypedDict, Annotated, List, Dict
from operator import add
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from src.utils import io
from src.tools.pdf_to_images import pdf_to_images
from src.tools.extractor_tools import (
    enhance_image_contrast,
    regex_anchor_locator,
)


# Extractor graph state schema
class InputState(TypedDict):
    answer_sheet_path: str | Path
    pages_dir: str | Path
    extract_dir: str | Path
    
    dpi: int
    max_pages: int | None
	
    exam: str

class OutputState(TypedDict):
    extraction_result: str

    messages: List[AnyMessage]

class ExtractorState(InputState, OutputState):
    page_counter: int
    pages: List[str]
    ocr_pages: Annotated[List[Dict[str, str]], add]

# Schemas for structured output
# OCR structured result
class PageExtraction(BaseModel):
    """Structured output for page text extraction."""
    page_id: str = Field(description="The original filename of the page image (e.g., 'page_01.png').")
    extracted_text: str = Field(description="The OCR-extracted text from the page.")

# Shredder structured result
class QuestionBundle(BaseModel):
    """Q/A mapping for a single question."""
    question_id: str = Field(description="The unique ID from the exam (e.g., '1.1')")
    student_answer: str = Field(description="The extracted text of the student's response")

class Result(BaseModel):
    """Structured output for exam Q/A structuring."""
    exam_structure: List[QuestionBundle] = Field(description="Step-by-step Question-Answer mapping per question.")

# Prompt templates
# OCR template
ocr_template = ChatPromptTemplate(
	[
		("system",
			[
				{
					"type": "text",
					"text":
"""You are an expert OCR (Optical Character Recognition) processor for handwritten exam sheets.

CORE DIRECTIVES:
1. HANDWRITING HANDLING: Transcribe handwriting to the best of your ability. If a word is completely unclear, write [UNK].
2. VERBATIM TRANSCRIPTION: Transcribe all text exactly as written, including typos, grammatical errors, and incomplete sentences. Do NOT "fix" the student's work and Do NOT invent missing words.
3. LAYOUT PRESERVATION: Maintain the spatial structure of the document:
    - Use whitespace and newlines to reflect the original layout.
	- For code blocks, preserve indentation and brackets (e.g., { }) exactly.
4. NON-TEXTUAL ARTIFACTS: Ignore stamps, coffee stains, or page borders, and drawn diagrams/symbols.
5. TOOL USAGE: If the image is too dark, washed out, or has heavy shadows that make text unreadable, use the available tools to preprocess the image before extracting text. Do not attempt to guess or hallucinate text that is obscured.

Extract ONLY the readable text and return plain text only (no markdown).""",
				},
        	]
    	),
		
		("human",
            [
				{
					"type": "image",
					"base64": "{page_b64}",
					"mime_type": "image/jpeg",
                },
				{
					"type": "text",
					"text": "Extract the text from the provided page image."
                }
            ]
		)
	]
)

# Shredder template
shredder_template = ChatPromptTemplate(
	[
		("system",
			[
				{
					"type": "text",
					"text":
"""You are an Exam Document Structurer and Question-Answer Mapper.
You receive an extraction object with a raw text stream (OCR output) for each page in the answer sheet, and your job is to segment the student's text into discrete "Question Bundles."

CORE DIRECTIVES:
1. ID-BASED ANCHORING: Use the Question IDs from the Exam (e.g., "1.1", "2.4", "4.2") as delimiters to slice the raw OCR text.
2. MAPPING LOGIC: For each ID in the Exam:
    - Identify the start of the student's answer (usually following the Question text or ID in the OCRed text).
	- Identify the end of the answer (before the next Question ID or the end of the document).
	- Extract the text in between as the student's response to that question.
3. EMPTY STATE HANDLING: If a Question ID exists in the Exam but no corresponding text is found in the student's OCR stream, mark the `student_answer` as "[NO_RESPONSE_PROVIDED]". 
4. FRAGMENT ASSEMBLY: If a student's answer spans across multiple pages, stitch the fragments into a single, continuous string.
5. CLEANING: Remove redundant headers, page numbers and repeating exam instructions from the middle of answers and keep only relevant content.
6. TOOL USAGE: Use the available tools when needed (eg, to locate question boundaries), but do not rely on them too much. Use your best judgment to produce the most accurate mapping possible.

Perform step-by-step mapping starting with the first question.""",
				},
				{
					"type": "text",
                    "text":
"""EXAM:

{exam}"""
                }
        	]
    	),
		
		("human",
"""OCRed pages from the student's answer sheet:

{ocr_pages}

Map student answers to exam questions."""
		)
	]
)

# Global variable to hold the agent instance
ocr_agent = None
shredder_agent = None


def get_images(state: InputState) -> ExtractorState:
    """Convert the student's answer sheet PDF into individual page images."""
    
    pdf_path = state["answer_sheet_path"]
    dpi = state["dpi"]
    out_dir = state["pages_dir"]
    max_pages = state["max_pages"]

    pages = pdf_to_images(
		pdf_path=pdf_path,
		out_dir=out_dir,
		dpi=dpi
		)
	
    return {
        "pages": pages[:max_pages],
		"page_counter": 0,
    }


def ocr(state: ExtractorState) -> ExtractorState:
    """Extract text from exam pages."""

    pages = state["pages"]
    out_dir = state["extract_dir"]
    index = state["page_counter"]

    page = pages[index]
    page_b64 = io.read_base64(page)

    # Create a chain that combines the prompt template with the agent.
    chain = ocr_template | ocr_agent

    output = chain.invoke({
		"page_b64": page_b64
	})

    # Extract the structured extraction result from the agent's output
    result = output.get("structured_response")
    if isinstance(result, PageExtraction):
        ocr_result = parse_ocr(result)
        io.write_text(
            filepath=Path(out_dir) / f"{ocr_result["page"]}.txt",
            content=ocr_result["text"]
            )
        io.write_json(
            filepath=Path(out_dir) / "raw_extraction.json",
            data=[ocr_result]
            )

    else:
        ocr_result = "Problem getting ocr result."

    return {
		"ocr_pages": [ocr_result],
		"page_counter": index + 1,
		"messages": [message for message in output["messages"] if not isinstance(message, SystemMessage)]
	}


def shredder(state: ExtractorState) -> OutputState:
    """Map student answers from the ocr text to exam question."""

    # Create a chain that combines the prompt template with the agent.
    chain = shredder_template | shredder_agent

    output = chain.invoke({
		"exam": state["exam"],
        "ocr_pages": state["ocr_pages"]
	})

    # Extract the structured extraction result from the agent's output
    result = output.get("structured_response")
    if isinstance(result, Result):
        extraction_result = parse_result(result)
    else:
        extraction_result = "Problem getting extraction result."

    print("Extracted answers from", len(state["ocr_pages"]), "pages.")
    return {
		"extraction_result": extraction_result,
		"messages": [message for message in output["messages"] if not isinstance(message, SystemMessage)]
	}
    

def build_agent(
		model: str,
		temperature: float,
		max_tokens: int | None
	):

    global ocr_agent, shredder_agent
    ocr_agent = create_agent(
        model=init_chat_model(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        tools=[enhance_image_contrast],
        response_format=Result,
    )
	
    shredder_agent = create_agent(
        model=init_chat_model(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        tools=[regex_anchor_locator],
        response_format=Result,
    )
	
    graph = StateGraph(
        ExtractorState, 
        input_schema=InputState,
        output_schema=OutputState,
        )
    graph.add_node("get_images", get_images)
    graph.add_node("ocr", ocr)
    graph.add_node("shredder", shredder)

    graph.set_entry_point("get_images")
    graph.add_edge("get_images", "ocr")
    graph.add_conditional_edges(
        "ocr",
        lambda state: "ocr" if state["page_counter"] < len(state["pages"]) else "shredder",
    )
    graph.add_edge("shredder", END)

    return graph.compile()


def parse_ocr(
		result: PageExtraction
	) -> Dict[str, str]:
    """Parse the ocr result into a human-readable format."""

    ocr_result = (
{
	"page": result.page_id,
	"text": result.extracted_text,
}
	    )

    return ocr_result

def parse_result(
		result: Result
	) -> str:
    """Parse the extraction result into a human-readable format."""

    map_result = (
f"""Student Answers:

{"\n\n".join(
        [
		    (
			f"Question: {bundle.question_id}\n"
			f"Answer: {bundle.student_answer}"
            )
		for bundle in result.exam_structure
        ]
    )	
}
"""
        )

    return map_result
