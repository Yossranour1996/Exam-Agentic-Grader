# other_architectures/zero_shot/graph/workflow.py
"""Builds the top-level LangGraph workflow for extraction, grading, QA, feedback, and export."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langchain.chat_models import init_chat_model

from zero_shot.subgraphs.export_graph import ExportGraph
from zero_shot.core.state import Context, InputState, OutputState, State
from zero_shot.prompts.prompt import prompt_template
from zero_shot.utils import io
from zero_shot.utils.logging import Logger


class GradingWorkflow:
    """Coordinate the full grading pipeline from raw input to exported results."""

    def __init__(self, model="gpt-4o", temperature=0.0, max_tokens=None):
        """Instantiate the extractor, grader, QA, feedback, and exporter components."""
        self.llm = init_chat_model(model=model, temperature=temperature, max_tokens=max_tokens)
        self.exporter = ExportGraph()

    @staticmethod
    def _load_yaml(path: Path, logger: Logger) -> dict[str, Any]:
        """Load one required mapping and fail early with a useful log entry."""
        if not path.is_file():
            logger.log(f"Required input not found: {path}", level="error", step_label="Prepare")
            raise FileNotFoundError(path)

        value = io.read_yaml(path, logger)

        if not isinstance(value, dict):
            logger.log(f"Invalid YAML mapping: {path}", level="error", step_label="Prepare")
            raise ValueError(f"Expected a YAML mapping in {path}")

        return value

    def node_prepare(self, state: InputState, runtime: Runtime[Context]) -> State:
        """Resolve runtime paths, create a per-run logger, and load immutable inputs."""
        required = ("exam_file", "rubric_file")
        missing = [name for name in required if not state.get(name)]
        if missing:
            raise ValueError(f"Missing input configuration: {', '.join(missing)}")

        context, sheet_id = runtime.context, state['sheet_id']
        output_dir = context.output_base / sheet_id
        run_id = datetime.now().strftime("%d_%H.%M")
        directories = {
            "extract_dir": output_dir / "extracted_answers",
            "results_dir": output_dir / "grading_results",
        }
        log_path = output_dir / "logs" / f"log_{run_id}.log"
        logger = Logger(log_path)
        files = {
            "exam": self._load_yaml(context.exams_dir / state['exam_file'], logger),
            "rubric": self._load_yaml(context.criteria_dir / state['rubric_file'], logger),
        }

        logger.log(f"Prepared state for {sheet_id}.\nPaths ready.", level="debug", step_label="Prepare")

        return {
            "run_id": run_id,
            "sheet_path": context.sheets_dir / f"{sheet_id}.pdf",
            **directories,
            "log_path": log_path,
            "logger": logger,
            **files,
        }


    def node_extraction(self, state: State, runtime: Runtime[Context]) -> State:
        """Run the extraction subgraph or reuse the latest extraction when disabled."""
        logger = state['logger']
        logger.log(f"Extracting answers for {state['sheet_id']} from {state['sheet_path']}", level="debug", step_label="Extraction")
        
        paths = sorted(state['extract_dir'].glob('structured_answers*.json'))
        if not paths:
            raise FileNotFoundError(f"No prior structured answers in {state['extract_dir']}")
        data = io.read_json(filepath=paths[-1], logger=logger)

        return {
            "student_answers": data,
            "messages": []
        }

    def node_grading(self, state: State, runtime: Runtime[Context]) -> State:
        """Run the grading subgraph and persist a grading report for the run."""
        logger = state['logger']
        
        print("Grading exam...")
        logger.log("Grading exam based on extracted answers and rubric.", level="debug", step_label="Grading")

        prompt = prompt_template.invoke({
            "exam": json.dumps(state['exam'], ensure_ascii=False),
            "rubric": json.dumps(state['rubric'], ensure_ascii=False),
            "student_answers": json.dumps(state['student_answers'], ensure_ascii=False)
        }).to_messages()
        output = self.llm.invoke(prompt)
        messages = prompt + [output]
        logger.log_messages(messages, step_label="Grading")

        print("Grading completed.\n")
        logger.log(f"Grading completed.", level="debug", step_label="Grading")

        grading_result, grading_result_str = self.parse_result(output.content, logger)
        return {
            "grading_result": grading_result,
            "grading_result_str": grading_result_str,
            "messages": prompt + [output]
        }


    def node_export(self, state: State, runtime: Runtime[Context]) -> OutputState:
        """Persist the final artifacts and close the per-run logger."""
        logger = state['logger']

        print("Exporting results...")
        logger.log("Exporting grading results to specified directory.", level="debug", step_label="Exporter")

        self.exporter.invoke(state, runtime.context)

        print("Results exported.\n")
        logger.log(f"Exported final grading, qa and feedback results to {state['results_dir']}. Exported data to {runtime.context.export_dir}\n", level="debug", step_label="Exporter")
        logger.shutdown()  # Flush the run-specific handler before returning.
        return state


    def parse_result(self, result: str, logger: Logger) -> tuple[dict[str, Any], str]:
        """Parse the grading result string into a structured dictionary."""
        if isinstance(result, str):
            result = re.sub(r"^```json\s*|\s*```$", "", result.strip(), flags=re.MULTILINE)
            try:
                result_json = json.loads(result)
                result_str = (
f"""Total Mark: {result_json.get('total_mark', '')}

Evaluations:
{"\n\n\n".join([(
    f'Question: {eval.get('question_id', '')}\n'
    f'Total Score: {eval.get('total_score', '')}\n\n'
    f'{"\n\n".join([(
        f'Sub-Question: {sub_evaluation.get('sub_question_id', '')}\n'
        f'Score: {sub_evaluation.get('score', '')}\n'
        f'Justification: {sub_evaluation.get('justification', '')}'
    )for sub_evaluation in eval.get('sub_evaluations', '')])}'
)for eval in result_json.get('evaluations', [])])}"""
                )
                return result_json, result_str

            except json.JSONDecodeError as e:
                logger.log(f"Failed to parse grading result: {e}", level="error", step_label="Grading")
                raise ValueError("Grading result is not valid JSON.") from e

        return result


    def compile(self):
        """Build the workflow graph with the main pipeline and the QA regrade loop."""
        graph = StateGraph(State, input_schema=InputState, output_schema=OutputState, context_schema=Context)

        for name, node in (("prepare", self.node_prepare), ("extraction", self.node_extraction), ("grading", self.node_grading), ("export", self.node_export)):
            graph.add_node(name, node)

        graph.set_entry_point("prepare")
        graph.add_edge("prepare", "extraction")
        graph.add_edge("extraction", "grading")
        graph.add_edge("grading", "export")
        graph.add_edge("export", END)

        return graph.compile()


# Compose the full grading pipeline as a stateful graph so each stage can
# read and update the same shared context as the process advances.
def build_graph(model="gpt-4o", temperature=0.0, max_tokens=None):
    """Public factory used by the application and tests."""
    return GradingWorkflow(model, temperature, max_tokens).compile()
