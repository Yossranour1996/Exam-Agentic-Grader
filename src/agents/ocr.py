# src/agents/orc.py
from typing import Dict, Any
from pathlib import Path

from src.agents.agent_base import AgentBase
from src.schemas.extractor_schemas import PageExtraction
from src.prompts.extractor_prompts import ocr_prompt_template
from src.utils.logging import Logger
from src.utils import io


class OCRAgent(AgentBase):
    """Transcribe one page into a filename/text pair."""

    def __init__(self, model: str, temperature: float, max_tokens: int | None):
        super().__init__("OCR Agent", model, temperature, max_tokens, None, None, PageExtraction, ocr_prompt_template)

    def skip(self, extract_dir: Path, logger: Logger | None) -> Dict[str, Any]:
        """Reuse all page transcriptions from the latest OCR run."""
        if logger:
            logger.log("OCR step skipped as per configuration.", level="warning", step_label="Extractor-OCR")

        paths = sorted(extract_dir.glob('raw_extraction*.json'))
        if not paths:
            raise FileNotFoundError(f"No prior OCR extraction in {extract_dir}")

        return {
            "result_json": io.read_json(filepath=paths[-1], logger=logger)
        }

    @staticmethod
    def parse_result(result: PageExtraction) -> str:
        """OCR text is already the desired human-readable representation."""
        return result.text
