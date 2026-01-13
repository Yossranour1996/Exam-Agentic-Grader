# src/tools/gemini_ocr.py
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

DEFAULT_PROMPT = """You are an OCR engine for handwritten exam sheets.
Extract ONLY the readable text.
- Preserve line breaks.
- Do NOT invent missing words.
- If a word is unclear, write [UNK].
- Ignore drawings/figures.
Return plain text only (no markdown)."""

def extract_text_from_image_gemini(image_path: str, prompt: str | None = None) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY env var. Put it in .env or set it in your shell.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    prompt = prompt or DEFAULT_PROMPT

    img_path = Path(image_path)
    data = img_path.read_bytes()

    # basic mime detect
    suffix = img_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"

    client = genai.Client(api_key=api_key)

    # IMPORTANT: pass Parts (text + bytes) not dicts
    resp = client.models.generate_content(
        model=model_name,
        contents=[
            prompt,
            types.Part.from_bytes(data=data, mime_type=mime),
        ],
    )
    return (resp.text or "").strip()
