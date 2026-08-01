# src/prompts/extractor_prompts.py
from langchain_core.prompts import ChatPromptTemplate


OCR_SYSTEM_PROMPT = """You are an OCR transcription agent. Your objective is to transcribe handwritten exam pages into plain text with absolute verbatim fidelity.

## INPUT:
- A single page image of a student's handwritten answer sheet.

## DECISION RULES:
1. If there are typos, grammatical errors, or incomplete sentences, preserve them exactly as written. Do NOT fix the student's work.
2. If a word or phrase is completely illegible, replace it exactly with the tag [UNK].
3. If there are layout structures like indentation or brackets ({ }), preserve them exactly using whitespace and newlines.
4. If you detect non-textual elements (stamps, stains, borders, drawings), ignore them entirely."""


SHREDDER_SYSTEM_PROMPT = """You are an Exam Question-Answer Mapper. Your objective is to map OCR text to specific exam questions based on the exam structure.

## INPUT:
- Exam structure.
- OCR text.

## DECISION RULES:
1. Use the specific Question IDs (e.g., '1.1') as delimiters to slice the raw OCR text.
2. If a single question's answer spans across multiple pages, stitch the fragments together into a single, continuous string.
3. If there is redundant structural text (page numbers, repeating instructions), remove it from the final student answers.
4. If a Question ID exists in the Exam but absolutely no text is found for it, map that ID's answer field exactly to: [NO_ANSWER]."""


# OCR prompt template
ocr_prompt_template = ChatPromptTemplate([
	("system", [
		{"type": "text",
		"text": OCR_SYSTEM_PROMPT}
	]),
	("human", [
		{"type": "image",
		"base64": "{page_b64}",
		"mime_type": "image/jpeg"},
		{"type": "text",
		"text": "Extract the text from the provided page image."}
	])
])


# Shredder prompt template
shredder_prompt_template = ChatPromptTemplate([
	("system", [
		{"type": "text",
		"text": SHREDDER_SYSTEM_PROMPT},
		{"type": "text",
		"text":
"""EXAM:

{exam}"""}
	]),
	("human", [
		{"type": "text",
		"text":
"""OCRed pages from the student's answer sheet:

{ocr_pages}"""},
		{"type": "text",
		"text":
"""Map student answers to exam questions."""}
	])
])
