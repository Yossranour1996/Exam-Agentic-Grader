# src/tools/extractor_tools.py
"""Optional image enhancement and text-boundary extraction helpers."""
import os
import re
import json
import tempfile
import cv2

from src.utils import io
from langchain_core.tools import tool


# Define any custom tools here that can be used by the grader agent.


@tool
def enhance_image_contrast(image_path: str) -> str:
    """
    Enhances image contrast using CLAHE to make faint or shadowed handwriting legible for OCR.
    Call this tool if an image page is washed out, dark, or has heavy shadows.

    Args:
        image_path: Absolute file system path to the image.

    Returns:
        JSON string containing status and base64 string of the processed image.
    """
    if not os.path.exists(image_path):
        return f"TOOL ERROR: Image path does not exist: {image_path}"

    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "TOOL ERROR: Unable to decode image file."

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(img)

        fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="ocr_enhanced_")
        os.close(fd)
        cv2.imwrite(temp_path, enhanced_img)

        b64_data = io.read_base64(temp_path)
        os.remove(temp_path)

        return json.dumps({
            "status": "SUCCESS",
            "message": "Image contrast enhanced successfully.",
            "image_base64": b64_data
        })
    except Exception as e:
        return f"TOOL ERROR during image contrast enhancement: {str(e)}"


@tool
def regex_anchor_locator(text: str, regex_pattern: str) -> str:
    r"""
    Locates exact character index boundaries and surrounding text snippets for question delimiters in raw OCR output.
    REQUIRED TOOL for 'The Shredder' agent when segmenting exam questions.

    Args:
        text: The raw OCR text payload.
        regex_pattern: Standard Python regex pattern (e.g., r'(?i)question\s*\d+\.\d+' or r'^\d+\.\d+').

    Returns:
        JSON string listing match count, matched string, character indices, and context previews.
    """
    try:
        compiled_pattern = re.compile(regex_pattern, re.MULTILINE)
        matches = []

        for match in compiled_pattern.finditer(text):
            start, end = match.start(), match.end()
            preview_start = max(0, start - 30)
            preview_end = min(len(text), end + 50)

            matches.append({
                "matched_delimiter": match.group(0),
                "start_index": start,
                "end_index": end,
                "context_preview": text[preview_start:preview_end].replace("\n", " ")
            })

        if not matches:
            return f"NO MATCHES FOUND for pattern: '{regex_pattern}'"

        return json.dumps({
            "total_matches": len(matches),
            "matches": matches
        }, indent=2)

    except re.error as e:
        return f"TOOL ERROR: Invalid regex pattern syntax: {str(e)}"
    except Exception as e:
        return f"TOOL ERROR during regex boundary location: {str(e)}"

