# src/tools/hunyuan_ocr.py
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration


MODEL_NAME = "tencent/HunyuanOCR"


@lru_cache(maxsize=1)
def _load_hunyuan() -> Tuple[HunYuanVLForConditionalGeneration, AutoProcessor]:
    """
    Lazy singleton loader:
    - Loads once then re-used for all pages.
    - Uses device_map="auto" when possible.
    """
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=False)

    # If you have CUDA, this will place on GPU automatically.
    # If not, it will run on CPU (slow).
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = HunYuanVLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="eager",  # safest on Windows
    )
    model.eval()
    return model, processor


def extract_text_from_image(
    image_path: str,
    prompt: str | None = None,
    max_new_tokens: int = 4096,
) -> str:
    """
    Runs HunyuanOCR on a single image and returns extracted text.

    Tip: start with simple "extract text" prompt (no coordinates).
    """
    model, processor = _load_hunyuan()

    if prompt is None:
        # Simple prompt for handwriting / noisy scans
        prompt = (
            "Extract all readable text from the image exactly as written. "
            "Do not add explanations. Keep line breaks. Ignore drawings."
        )

    img = Image.open(image_path).convert("RGB")

    # Chat template used by their recommended usage
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ]},
    ]

    chat_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[chat_prompt],
        images=[img],
        padding=True,
        return_tensors="pt",
    )

    # Move tensors to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Trim prompt tokens
    input_ids = inputs.get("input_ids", None)
    if input_ids is not None:
        generated = generated[:, input_ids.shape[1]:]

    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return text.strip()
