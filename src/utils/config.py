# src/utils/config.py
"""Centralized configuration for the exam grader application.

Contains the paths for all the base directories used in the worklfow"""

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path("data")
INPUT_BASE = BASE_DIR / "input"
OUTPUT_BASE = BASE_DIR / "output"
EXAMS_DIR = INPUT_BASE / "exams"
CRITERIA_DIR = INPUT_BASE / "criteria"
ANSWER_SHEETS_DIR = INPUT_BASE / "answer_sheets"


@dataclass
class Paths:
	"""Data directory paths class."""
	base: Path = BASE_DIR
	input_base: Path = INPUT_BASE
	output_base: Path = OUTPUT_BASE
	exams_dir: Path = EXAMS_DIR
	criteria_dir: Path = CRITERIA_DIR
	answer_sheets_dir: Path = ANSWER_SHEETS_DIR
