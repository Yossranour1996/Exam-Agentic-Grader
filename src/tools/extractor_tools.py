# src/tools/extractor_tools.py
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
	Enhances the contrast of an image to make text more readable for OCR.
	Use this if the image is too dark, washed out, or has heavy shadows.

	:param image_path: The absolute path to the image file.
	:return: Base64 string of the newly enhanced image.
	"""
	if not os.path.exists(image_path):
		return f"TOOL ERROR: File not found at {image_path}"

	try:
		# Load the image in grayscale, which is optimal for OCR text extraction
		img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		if img is None:
			return "TOOL ERROR: OpenCV could not read the image."

		# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		enhanced_img = clahe.apply(img)

		# Save to a temporary file
		fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="enhanced_")
		os.close(fd) # Close the file descriptor, cv2 will open it to write
		cv2.imwrite(temp_path, enhanced_img)

		enhanced_b64 = io.read_base64(temp_path)

		return f"SUCCESS: Contrast enhanced. Base64 string of the new image: {enhanced_b64}"

	except Exception as e:
		return f"TOOL ERROR during contrast enhancement: {str(e)}"
    

@tool
def regex_anchor_locator(text: str, regex_pattern: str) -> str:
	r"""
	Finds exact character indices for specific patterns in a raw text string.
	Use this to safely locate where questions begin and end in the OCR text without hallucinating.

	:param text: The raw OCR text string to search through.
	:param regex_pattern: A Python standard regex pattern (e.g., r'Question \d+' or r'\d+\.\d+').
	:returns: A JSON string containing a list of matches with their exact start and end character indices.
    """
	try:
		# We compile the pattern with MULTILINE so it handles exam structures better
		compiled_pattern = re.compile(regex_pattern, re.MULTILINE)

		matches = []
		for match in compiled_pattern.finditer(text):
			matches.append({
				"matched_string": match.group(0),
				"start_index": match.start(),
				"end_index": match.end()
			})

		if not matches:
			return f"NO MATCHES FOUND for pattern: {regex_pattern}"
            
		# Return as formatted JSON so the LangGraph agent can easily parse it
		return json.dumps({
			"total_matches": len(matches),
			"match_data": matches
		}, indent=2)
        
	except re.error as e:
		return f"TOOL ERROR: Invalid regex pattern provided. Details: {str(e)}"
	except Exception as e:
		return f"TOOL ERROR during regex search: {str(e)}"
