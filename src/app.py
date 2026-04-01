# src/app.py
from __future__ import annotations
from dotenv import load_dotenv

from src.state import InputState, Context
from src.graph.workflow import build_graph

load_dotenv()

# Entry point for research test runs.
# This script builds the workflow graph, invokes it with example input, and prints output paths.
def main():
    print("Building workflow graph...\n")
    graph = build_graph(
        model="gpt-5",
        temperature=0.0,
    )

    print("Invoking graph with test data...\n")
    final_state = graph.invoke(
        input=InputState({
            "student_id": "student_01",
            "dpi": 300,
            "max_pages": 10,
            "exam_file": "Exam Java alternative 2025.txt",
            "rubric_file": "java_criteria.txt",
            }),
        context=Context(),
        )

    print("\n✅ DONE")
    print("Pages saved under:", f"{final_state['extract_dir']}")
    print("Grading results saved under:", f"{final_state['export_dir']}")
    print("Message log saved in:", f"{final_state['log_path']}")

if __name__ == "__main__":
    main()
