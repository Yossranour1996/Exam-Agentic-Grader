# other_architectures/zero_shot/subgraphs/export_graph.py
"""Exports the final grading artifacts as text, JSON, and spreadsheet data."""

from __future__ import annotations

from typing import Dict, Any

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime

from zero_shot.core.state import State, Context
from zero_shot.utils import io


class ExportGraph:
    """Persist readable artifacts and append analysis-friendly workbook rows."""

    def __init__(self):
        self.graph = self.compile()


    def node_save_results(self, state: State, runtime: Runtime[Context]) -> State:
        """Persist the final grading, QA, and feedback results as text and JSON."""
        logger = state['logger']
        if not state.get('do_save_results', True):
            print("Skipping save_results step as per configuration.")
            logger.log("Skipping save_results step as per configuration.", level="info", step_label="Exporter")
            return state

        grading = state['grading_result']
        grading_str = state['grading_result_str']
        run_id, results_dir = state['run_id'], state['results_dir']

        io.write_text(filepath=results_dir / f"final_result_{run_id}.txt", content=grading_str, logger=logger)
        io.write_json(filepath=results_dir / f"final_result_{run_id}.json", data={"grading": grading}, logger=logger)

        logger.log(f"Saved final grading results to {results_dir}.", level="info", step_label="Exporter")
        return state


    def node_data_to_sheet(self, state: State, runtime: Runtime[Context]) -> State:
        """Append a compact results row and a tool-usage row to Excel workbooks."""
        logger = state['logger']
        if not state.get('do_export_to_sheet', True):
            print("Skipping data_to_sheet step as per configuration.")
            logger.log("Skipping data_to_sheet step as per configuration.", level="info", step_label="Exporter")
            return state

        export_dir = runtime.context.export_dir
        io.write_excel(filepath=export_dir / "results.xlsx", data=self.get_results_data(state), logger=logger)

        logger.log(f"Exported final grading, qa and feedback results to {state['results_dir']}. Exported data to {export_dir}\n", level="debug", step_label="Exporter")
        return state


    @staticmethod
    def get_results_data(state: State) -> Dict[str, Any]:
        """Flatten nested evaluation data into one row per answer sheet."""
        row = {"sheet_id": state['sheet_id']}
        grading = state['grading_result']
        row["total_mark"] = grading.get('total_mark', "")
        for evaluation in grading.get('evaluations', []):
            row[evaluation.get('question_id', "")] = evaluation.get('total_score', "")
            for item in evaluation.get('sub_evaluations', []):
                row[item.get('sub_question_id', "")] = item.get('score', "")
        return {"Results": [row]}


    def compile(self):
        """Build the two-step export subgraph used to save artifacts and workbook rows."""
        graph = StateGraph(State, context_schema=Context)
        graph.add_node("save_results", self.node_save_results)
        graph.add_node("data_to_sheet", self.node_data_to_sheet)

        graph.set_entry_point("save_results")
        graph.add_edge("save_results", "data_to_sheet")
        graph.add_edge("data_to_sheet", END)

        return graph.compile()

    def invoke(self, state: State, context: Context) -> State:
        """Execute both export stages using the provided runtime context."""
        return self.graph.invoke(state, context=context)
