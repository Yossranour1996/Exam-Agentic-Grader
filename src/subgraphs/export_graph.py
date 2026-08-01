# src/subgraphs/export_graph.py
from __future__ import annotations

from typing import Dict, Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime

from src.core.state import State, Context
from src.utils import io


class ExportGraph:
    """Save readable artifacts first, then append analysis-friendly workbooks."""

    def __init__(self):
        self.graph = self.compile()


    def node_save_results(self, state: State, runtime: Runtime[Context]) -> State:
        """Persist final grading, QA, and feedback as text and JSON."""
        logger = state['logger']
        if not state.get('do_save_results', True):
            print("Skipping save_results step as per configuration.")
            logger.log("Skipping save_results step as per configuration.", level="info", step_label="Exporter")
            return state

        grading, qa, feedback = state['grading_result'], state['qa_result'], state['feedback_result']
        text = (f"""Final Grading and Feedback Results
Grading:
\n{state['grading_result_str']}
{'-' * 80}\n
Feedback:
\n{state['feedback_result_str']}"""
        )
        run_id, results_dir = state['run_id'], state['results_dir']

        io.write_text(filepath=results_dir / f"final_result_{run_id}.txt", content=text, logger=logger)
        io.write_json(filepath=results_dir / f"final_result_{run_id}.json", data={"grading": grading, "feedback": feedback}, logger=logger)
        io.write_json(filepath=results_dir / f"grading_final_{run_id}.json", data=grading, logger=logger)
        io.write_json(filepath=results_dir / f"qa_final_{run_id}.json", data=qa, logger=logger)
        io.write_json(filepath=results_dir / f"feedback_{run_id}.json", data=feedback, logger=logger)

        logger.log(f"Saved final grading, qa and feedback results to {results_dir}.", level="info", step_label="Exporter")
        return state


    def node_data_to_sheet(self, state: State, runtime: Runtime[Context]) -> State:
        """Append one compact results row and one tool-usage row to Excel."""
        logger = state['logger']
        if not state.get('do_export_to_sheet', True):
            print("Skipping data_to_sheet step as per configuration.")
            logger.log("Skipping data_to_sheet step as per configuration.", level="info", step_label="Exporter")
            return state

        export_dir = runtime.context.export_dir
        io.write_excel(filepath=export_dir / "results.xlsx", data=self.get_results_data(state), logger=logger)
        io.write_excel(filepath=export_dir / "tools.xlsx", data=self.get_tools_data(state), logger=logger)

        logger.log(f"Exported final grading, qa and feedback results to {state['results_dir']}. Exported data to {export_dir}\n", level="debug", step_label="Exporter")
        return state


    @staticmethod
    def get_results_data(state: State) -> Dict[str, Any]:
        """Flatten nested item scores into one row per answer sheet."""
        row = {"sheet_id": state['sheet_id'], "regrade_attempts": state.get('max_regrade', 1) - state['regrade_counter']}
        grading = state['grading_result']
        row["total_mark"] = grading.get('total_mark', "")
        for evaluation in grading.get('evaluations', []):
            row[evaluation.get('question', "")] = evaluation.get('total_points', "")
            for item in evaluation.get('sub_evaluations', []):
                row[item.get('sub_question', "")] = item.get('awarded_points', "")
        return {"Results": [row]}

    @staticmethod
    def get_tools_data(state: State) -> Dict[str, Any]:
        """Count tool calls retained in the shared trace channel."""
        row, total = {"sheet_id": state['sheet_id']}, 0
        for message in state.get('messages', []):
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    key = f"tool_{call.get('name', '')}"
                    row[key], total = row.get(key, 0) + 1, total + 1
        row["tools_total"] = total
        return {"Tools": [row]}


    def compile(self):
        """Build the same wrapper-style two-node subgraph used elsewhere."""
        graph = StateGraph(State, context_schema=Context)
        graph.add_node("save_results", self.node_save_results)
        graph.add_node("data_to_sheet", self.node_data_to_sheet)

        graph.set_entry_point("save_results")
        graph.add_edge("save_results", "data_to_sheet")
        graph.add_edge("data_to_sheet", END)

        return graph.compile()

    def invoke(self, state: State, context: Context) -> State:
        """Execute both export stages."""
        return self.graph.invoke(state, context=context)
