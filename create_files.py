# scripts/init_repo.py
from pathlib import Path

STRUCTURE = [
    "data/input/exams",
    "data/input/answer_keys",
    "data/input/attendance",
    "data/output/extracted_json",
    "data/output/grading_results",
    "data/output/exports",
    "data/output/logs",
    "src/graph",
    "src/agents",
    "src/tools",
    "src/utils",
    "scripts",
]

FILES = {
    "README.md": "# Agentic Grader\n",
    ".env.example": "OPENAI_API_KEY=\n",
    "src/app.py": "",
    "src/state.py": "",
    "src/graph/workflow.py": "",
    "src/agents/coordinator.py": "",
    "src/agents/extractor.py": "",
    "src/agents/grader_java.py": "",
    "src/agents/qa.py": "",
    "src/agents/attendance.py": "",
    "src/agents/exporter.py": "",
    "src/tools/pdf_to_images.py": "",
    "src/tools/ocr.py": "",
    "src/tools/java_runner.py": "",
    "src/tools/tests_runner.py": "",
    "src/tools/fuzzy_match.py": "",
    "src/tools/excel_export.py": "",
    "src/utils/config.py": "",
    "src/utils/io.py": "",
    "src/utils/logging.py": "",
}

def main():
    root = Path(".")
    for d in STRUCTURE:
        (root / d).mkdir(parents=True, exist_ok=True)

    for fpath, content in FILES.items():
        p = root / fpath
        if not p.exists():
            p.write_text(content, encoding="utf-8")

    print("✅ Repo structure created.")

if __name__ == "__main__":
    main()
