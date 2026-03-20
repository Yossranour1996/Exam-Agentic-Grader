# src/agents/extractor.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from src.tools.gemini_ocr import extract_text_from_image_gemini

def extract_exam_pages(images: List[str], out_dir: str | Path, max_pages: int | None = None) -> List[Dict[str, Any]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = images[:max_pages]

    results = []
    for img_path in images:
        print(f"🧠 OCR (Gemini) -> {img_path.name}")
        text = extract_text_from_image_gemini(str(img_path))

        results.append({"page": img_path.name, "text": text})

        (out_dir / f"{img_path.stem}.txt").write_text(text, encoding="utf-8")

    (out_dir / "raw_extraction.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return results
