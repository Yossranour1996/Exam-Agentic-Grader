# src/graph/workflow.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime

from src.agents.feedback import FeedbackAgent
from src.agents.qa import QAAgent
from src.core.state import Context, InputState, OutputState, State
from src.subgraphs.export_graph import ExportGraph
from src.subgraphs.extraction_graph import ExtractionGraph
from src.subgraphs.grading_graph import GradingGraph
from src.utils import io
from src.utils.logging import Logger


class GradingWorkflow:
    """Coordinate extraction, parallel grading, QA, feedback, and export."""

    def __init__(self, model="gpt-4o", ocr_model="gpt-4o", temperature=0.0, max_tokens=None):
        self.extractor = ExtractionGraph(ocr_model, model, temperature, max_tokens)
        self.grader = GradingGraph(model, temperature, max_tokens)
        self.qa = QAAgent(model, temperature, max_tokens)
        self.feedback = FeedbackAgent(model, temperature, max_tokens)
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
        """Resolve paths, initialize logging, and load immutable grading inputs."""
        required = ("exam_file", "rubric_file", "review_criteria_file")
        missing = [name for name in required if not state.get(name)]
        if missing:
            raise ValueError(f"Missing input configuration: {', '.join(missing)}")

        context, sheet_id = runtime.context, state['sheet_id']
        output_dir = context.output_base / sheet_id
        run_id = datetime.now().strftime("%d_%H.%M")
        directories = {
            "pages_dir": output_dir / "pages",
            "extract_dir": output_dir / "extracted_answers",
            "reports_dir": output_dir / "reports",
            "results_dir": output_dir / "grading_results",
        }
        log_path = output_dir / "logs" / f"log_{run_id}.log"
        logger = Logger(log_path)
        files = {
            "exam": self._load_yaml(context.exams_dir / state['exam_file'], logger),
            "rubric": self._load_yaml(context.criteria_dir / state['rubric_file'], logger),
            "review_criteria": self._load_yaml(context.criteria_dir / state['review_criteria_file'], logger)
        }

        logger.log(f"Prepared state for {sheet_id}.\nPaths ready.", level="debug", step_label="Prepare")

        return {
            "run_id": run_id,
            "sheet_path": context.sheets_dir / f"{sheet_id}.pdf",
            **directories,
            "log_path": log_path,
            "logger": logger,
            **files,
            "regrade_counter": state.get('max_regrade', 1)
        }


    def node_extractor(self, state: State, runtime: Runtime[Context]) -> State:
        """Extract and structure answers, or reuse the latest extraction."""
        logger = state['logger']
        if not state.get('do_extract', True):
            print("Extraction step skipped as per configuration.\n")
            return self.extractor.skip(state['extract_dir'], logger)

        print("Extracting answers...")
        logger.log(f"Extracting answers for {state['sheet_id']} from {state['sheet_path']}", level="debug", step_label="Extractor")

        state = self.extractor.invoke(state)

        print("Extracted answers.\n")
        logger.log(f"Extracted answers, saved at {state['extract_dir']}.", level="info", step_label="Extractor")

        return state


    def node_grader(self, state: State, runtime: Runtime[Context]) -> State:
        """Run the requested question workers and save the aggregated attempt."""
        logger = state['logger']
        if not state.get('do_grade', True):
            print("Grading step skipped as per configuration.\n")
            return self.grader.skip(state['results_dir'], logger)

        print("Grading exam...")
        logger.log("Grading exam based on extracted answers and rubric.", level="debug", step_label="Grader")

        state = self.grader.invoke(state)

        io.write_json(filepath=state['reports_dir'] / f"grading_reports_{state['run_id']}.json", data=[state['grading_result']], logger=logger)

        print("Grading completed.\n")
        logger.log(f"Grading completed. Reports saved at {state['reports_dir']}", level="debug", step_label="Grader")

        return state


    def node_qa(self, state: State, runtime: Runtime[Context]) -> State:
        """Audit the combined grade and identify only questions needing another pass."""
        logger = state['logger']
        if not state.get('do_qa', True):
            print("Quality assurance step skipped as per configuration.\n")
            output = self.qa.skip(logger)
            data, text = output['result_json'], output['result_str']
            return {
                "regrade": "<NO_REGRADE>",
                "regrade_counter": 0,
                "qa_result": data,
                "qa_result_str": text,
                "messages": output.get('messages', [])
            }

        if state.get('regrade'):
            state['regrade_counter'] -= 1

        print("Performing quality assurance on grading results...")
        logger.log("Performing quality assurance on grading results based on review criteria.", level="debug", step_label="Quality-Assurance")

        output = self.qa.invoke({
            "rubric": json.dumps(state['rubric'], ensure_ascii=False),
            "review_criteria": json.dumps(state['review_criteria'], ensure_ascii=False),
            "remaining_regrade_attempts": state['regrade_counter'],
            "student_answers": json.dumps(state['student_answers'], ensure_ascii=False),
            "grading": json.dumps(state['grading_result'], ensure_ascii=False)
        })

        io.write_json(filepath=state['reports_dir'] / f"qa_reports_{state['run_id']}.json", data=[output['result_json']], logger=logger)

        logger.log_messages(output['messages'], step_label="Quality-Assurance")
        print("Quality assurance completed.\n")
        logger.log(f"Quality assurance completed. Reports saved at {state['reports_dir']}", level="debug", step_label="Quality-Assurance")

        data, text = output['result_json'], output['result_str']
        question_ids = sorted([audit.get('question', '') for audit in data.get('audits', [])])
        return {
            "regrade": data.get('decision', '') == "<REGRADE_REQUIRED>",
            "regrade_question_ids": question_ids,
            "regrade_counter": state['regrade_counter'],
            "qa_result": data,
            "qa_result_str": text,
            "messages": output.get('messages', [])
        }


    def node_feedback(self, state: State, runtime: Runtime[Context]) -> State:
        """Convert final grading evidence into student-facing guidance."""
        logger = state['logger']
        if not state.get('do_feedback', True):
            print("Feedback generation step skipped as per configuration.\n")
            result = self.feedback.skip(logger)
            return {
                "feedback_result_str": result['result_str'],
                "feedback_result_json": result['result_json'],
                "messages": []
            }

        print("Generating feedback for the student...")
        logger.log("Generating feedback for the student based on grading and QA results.", level="debug", step_label="Feedback")

        result = self.feedback.invoke({
            "exam": json.dumps(state['exam'], ensure_ascii=False),
            "rubric": json.dumps(state['rubric'], ensure_ascii=False),
            "student_answers": json.dumps(state['student_answers'], ensure_ascii=False),
            "grading": json.dumps(state['grading_result'], ensure_ascii=False)
        })

        logger.log_messages(result['messages'], step_label="Feedback")
        print("Feedback generation completed.\n")
        logger.log("Feedback generation completed.", level="debug", step_label="Feedback")
        return {
            "feedback_result": result['result_json'],
            "feedback_result_str": result['result_str'],
            "messages": result['messages']
        }


    def node_export(self, state: State, runtime: Runtime[Context]) -> OutputState:
        """Persist final artifacts and close the per-run logger."""
        logger = state['logger']
        if not state.get('do_export', True):
            print("Exporting step skipped as per configuration.\n")
            logger.log("Exporting step skipped as per configuration.", level="warning", step_label="Exporter")
            return state

        print("Exporting results...")
        logger.log("Exporting grading results to specified directory.", level="debug", step_label="Exporter")

        self.exporter.invoke(state, runtime.context)

        print("Results exported.\n")
        logger.log(f"Exported final grading, qa and feedback results to {state['results_dir']}. Exported data to {runtime.context.export_dir}\n", level="debug", step_label="Exporter")
        logger.shutdown()  # Flush the run-specific handler before returning.
        return state

    def regrade_decision(self, state: State) -> str:
        """Route QA failures back only while the configured budget remains."""
        logger = state['logger']
        if state['regrade'] and state['regrade_counter'] > 0:
            print("QA requested regrade. Routing back to grader.\n")
            logger.log(f"QA requested regrade. Remaining regrade attempts: {state['regrade_counter']}. Routing back to grader.", level="warning", step_label="Quality-Assurance (Regrade Decision)")
            return "regrade"

        if state['regrade'] and state['regrade_counter'] <= 0:
            print("QA requested regrade. Reached maximum regrades allowed. Proceeding to feedback.\n")
            logger.log("QA requested regrade. Reached maximum regrades allowed. Proceeding to feedback.", level="debug", step_label="Quality-Assurance (Regrade Decision)")

        else:
            print("QA cleared. Proceeding to feedback.\n")
            logger.log("QA cleared. Proceeding to feedback.", level="debug", step_label="Quality-Assurance (Regrade Decision)")

        return "continue"

    def compile(self):
        """Build the linear workflow with one conditional QA loop."""
        graph = StateGraph(State, input_schema=InputState, output_schema=OutputState, context_schema=Context)

        for name, node in (("prepare", self.node_prepare), ("extractor", self.node_extractor), ("grader", self.node_grader), ("quality_assurance", self.node_qa), ("feedback", self.node_feedback), ("export", self.node_export)):
            graph.add_node(name, node)

        graph.set_entry_point("prepare")
        graph.add_edge("prepare", "extractor")
        graph.add_edge("extractor", "grader")
        graph.add_edge("grader", "quality_assurance")
        graph.add_conditional_edges("quality_assurance", self.regrade_decision, {"regrade": "grader", "continue": "feedback"})
        graph.add_edge("feedback", "export")
        graph.add_edge("export", END)

        return graph.compile()


def build_graph(model="gpt-4o", ocr_model="gpt-4o", temperature=0.0, max_tokens=None):
    """Public factory used by the application and tests."""
    return GradingWorkflow(model, ocr_model, temperature, max_tokens).compile()
