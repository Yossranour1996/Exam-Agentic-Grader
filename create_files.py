# scripts/init_repo.py
from pathlib import Path

STRUCTURE = [
    "data/input/exams",
    "data/input/rubrics",
    "data/output/exports",
    "src/agents",
    "src/core",
    "src/graph",
    "src/prompts",
    "src/schemas",
    "src/subgraphs",
    "src/tools",
    "src/utils",
]

for sheet_num in range(1, 11):
    sheet_id = f"sheet_{sheet_num:03d}"
    STRUCTURE.extend(
        [
            f"data/output/{sheet_id}/extracted_answers",
            f"data/output/{sheet_id}/grading_results",
            f"data/output/{sheet_id}/logs",
            f"data/output/{sheet_id}/reports",
        ]
    )

FILES = {
    "README.md": "# Agentic Grader\n",
    "requirements.txt": "",
    "src/.env.example": "OPENAI_API_KEY=\n",
    "src/app.py": "",
    "src/test.ipynb": "",
    "src/agents/agent_base.py": "",
    "src/agents/feedback.py": "",
    "src/agents/grader.py": "",
    "src/agents/ocr.py": "",
    "src/agents/qa.py": "",
    "src/agents/shredder.py": "",
    "src/core/config.yaml": "",
    "src/core/paths.yaml": "",
    "src/core/state.py": "",
    "src/graph/workflow.py": "",
    "src/prompts/extractor_prompts.py": "",
    "src/prompts/feedback_prompt.py": "",
    "src/prompts/grader_prompt.py": "",
    "src/prompts/qa_prompt.py": "",
    "src/schemas/extractor_schemas.py": "",
    "src/schemas/feedback_schema.py": "",
    "src/schemas/grader_schema.py": "",
    "src/schemas/qa_schema.py": "",
    "src/subgraphs/export_graph.py": "",
    "src/subgraphs/extraction_graph.py": "",
    "src/subgraphs/grading_graph.py": "",
    "src/tools/extractor_tools.py": "",
    "src/tools/feedback_tools.py": "",
    "src/tools/gemini_ocr.py": "",
    "src/tools/grader_tools.py": "",
    "src/tools/hunyuan_ocr.py": "",
    "src/tools/pdf_to_images.py": "",
    "src/tools/qa_tools.py": "",
    "src/tools/retriever.py": "",
    "src/utils/io.py": "",
    "src/utils/logging.py": "",
    "data/input/human_grading.csv": "student_id,score,feedback\n",
    "data/input/exams/exam_java.yaml": "# Java exam configuration\n",
    "data/input/rubrics/java_criteria.yaml": "# Java grading rubric\n",
    "data/input/rubrics/review_criteria.yaml": "# Review criteria\n",
}


def main() -> None:
    root = Path(".")
    for directory in STRUCTURE:
        (root / directory).mkdir(parents=True, exist_ok=True)

    for file_path, content in FILES.items():
        path = root / file_path
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    print("Repository structure created.")


if __name__ == "__main__":
    main()
