# src/app.py
from __future__ import annotations

from src.graph.workflow import build_extraction_graph

def main():
    graph = build_extraction_graph()

    final_state = graph.invoke({
        "student_id": "student_03",
        "student_pdf": "data/input/answers_sheets/student_03.pdf",
        "dpi": 300,
        "max_pages": 10
    })

    print("\n✅ DONE")
    print("Saved under:", f"data/output/{final_state['student_id']}/extracted_text")
    print("Pages OCR'd:", len(final_state.get("ocr_pages", [])))

if __name__ == "__main__":
    main()
