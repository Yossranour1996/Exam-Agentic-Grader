# src/app.py
from __future__ import annotations

import yaml
from pathlib import Path
from dotenv import load_dotenv

from src.core.state import InputState, Context
from src.graph.workflow import build_graph

load_dotenv()

def get_config(config_dir: str = "src/core") -> dict:
    """Load model and path settings into one application mapping."""
    config_dir_path = Path(config_dir)

    with open(config_dir_path / "config.yaml", "r") as file:
        config_data = yaml.safe_load(file)
    with open(config_dir_path / "paths.yaml", "r") as file:
        paths_data = yaml.safe_load(file)

    return {**config_data, **paths_data}


def main():
    print("Loading application configuration...\n")
    app_config = get_config()

    global_settings = app_config["global"]
    extractor_settings = app_config["agents"]["extractor"]
    base_dirs = app_config["base_directories"]
    input_dirs = app_config["inputs"]
    output_dirs = app_config["outputs"]

    print("Building workflow graph...\n")
    graph = build_graph(
        model=global_settings["default_model"],
        ocr_model=extractor_settings["ocr_model"],
        temperature=global_settings["default_temperature"],
        max_tokens=global_settings["max_tokens"]
    )

    print("Invoking graph with test data...\n")
    final_state = graph.invoke(
        input=InputState({
            "sheet_id": "sheet_003",
            "dpi": 300,
            "max_regrade": 2,
            "do_pdf_to_imgs": False,
            "do_ocr": False,
            "exam_file": "exam_java.yaml",
            "rubric_file": "java_criteria.yaml",
            "review_criteria_file": "review_criteria.yaml"
        }),
        context=Context(
            sheets_dir=Path(input_dirs["sheets_dir"]),
            exams_dir=Path(input_dirs["exams_dir"]),
            criteria_dir=Path(input_dirs["criteria_dir"]),
            output_base=Path(base_dirs["output_base"]),
            export_dir=Path(output_dirs["exports_dir"])
        )
    )

    print("DONE")
    print("Extraction results saved under:", f"{final_state['extract_dir']}")
    print("Grading and QA reports saved under:", f"{final_state['reports_dir']}")
    print("Final grading, QA and feedback results saved under:", f"{final_state['results_dir']}")
    print("Grading data exported to:", f"{output_dirs["exports_dir"]}")
    print("Message log saved in:", f"{final_state['log_path']}")

if __name__ == "__main__":
    main()
