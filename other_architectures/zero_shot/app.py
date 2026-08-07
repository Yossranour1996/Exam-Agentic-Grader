# other_architectures/zero_shot/app.py
"""Application entry point that loads configuration and runs the grading workflow."""

from __future__ import annotations

import yaml
from pathlib import Path
from dotenv import load_dotenv

from zero_shot.core.state import InputState, Context
from zero_shot.graph.workflow import build_graph

# Load configuration from the local environment file so the app can access
# API keys and runtime settings without hard-coding them into the source.
load_dotenv()

def get_config(config_dir: str = "zero_shot/core") -> dict:
    """Load model, path, and directory settings into a single config mapping.

    The application keeps runtime settings in separate YAML files, so this helper
    merges them into one dictionary that the workflow can consume consistently.
    """
    config_dir_path = Path(config_dir)

    with open(config_dir_path / "config.yaml", "r") as file:
        config_data = yaml.safe_load(file)
    with open(config_dir_path / "paths.yaml", "r") as file:
        paths_data = yaml.safe_load(file)

    return {**config_data, **paths_data}


def main():
    """Build and run the workflow against a sample exam input for local testing."""
    print("Loading application configuration...\n")
    app_config = get_config()

    global_settings = app_config["global"]
    base_dirs = app_config["base_directories"]
    input_dirs = app_config["inputs"]
    output_dirs = app_config["outputs"]

    print("Building workflow graph...\n")
    graph = build_graph(
        model=global_settings["default_model"],
        temperature=global_settings["default_temperature"],
        max_tokens=global_settings["max_tokens"]
    )

    print("Invoking graph with test data...\n")
    final_state = graph.invoke(
        input=InputState({
            "sheet_id": "sheet_001",
            "exam_file": "exam_java.yaml",
            "rubric_file": "java_criteria.yaml",
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
    print("Final grading results saved under:", f"{final_state['results_dir']}")
    print("Grading data exported to:", f"{output_dirs["exports_dir"]}")
    print("Message log saved in:", f"{final_state['log_path']}")

if __name__ == "__main__":
    main()
