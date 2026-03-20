# src/app.py
from __future__ import annotations
from dotenv import load_dotenv

from src.state import InputState, Context
from src.graph.workflow import build_graph

load_dotenv()

# Entry point for research test runs.
# This script builds the workflow graph, invokes it with example input, and prints output paths.
def main():
    print("Building workflow graph...")
    graph = build_graph()

    print("\nInvoking graph with test data...")
    final_state = graph.invoke(
        input=InputState({
        "student_id": "student_01",
        "dpi": 300,
        "max_pages": 10,
        "rubric_file": "java_criteria.txt",
        }),
        context=Context(
            model="gpt-4o",
            temperature=0.0
            ),
        # config={"configurable": {"thread_id": "1"}}
        )

    print("\n✅ DONE")
    print("Pages saved under:", f"{final_state['extract_dir']}")
    print("Grading results saved under:", f"{final_state['export_dir']}")
    print("Message log saved in:", f"{final_state['log_path']}")

if __name__ == "__main__":
    main()
