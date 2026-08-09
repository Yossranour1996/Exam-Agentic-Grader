# src/subgraphs/grading_graph.py
"""Builds the grading subgraph that fans out question-level grading work and merges the results."""

from __future__ import annotations

import json
from operator import add
from pathlib import Path
from typing import Annotated, List, Dict, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.grader import GraderAgent
from src.core.state import State, QuestionTask
from src.utils.logging import Logger
from src.utils import io


class GradingState(State):
    grading_workers_results: Annotated[List[Dict[str, Any]], add]


class GradingGraph:
    """Run question-scoped grading in parallel and merge the results deterministically."""

    def __init__(self, model="gpt-4o", temperature=0.0, max_tokens=None):
        self.graders = {
            section: GraderAgent(model, temperature, max_tokens, section)
            for section in ("Q1", "Q2", "Q3", "Q4")
        }
        self.graph = self.compile()


    def map_questions(self, state: State) -> list[Send]:
        """Fan out all questions initially, or only the QA-selected questions on regrade."""
        logger = state["logger"]
        logger.log("Fanning out grading tasks per question.", level="debug", step_label="Grader-Map/Reduce")

        evaluations = {evaluation["question"]: evaluation
            for evaluation in state.get('grading_result', {}).get("evaluations", [])}
        audits = {audit["question"]: audit
            for audit in state.get("qa_result", {}).get("audits", [])}
        rubrics = {str(section["id"]): section
            for section in state["rubric"]["sections"]}
        answers = {group["question"]: group.get("sub_questions", [])
            for group in state["student_answers"].get("questions", [])}

        tasks: list[Send] = []
        requested = state.get('regrade_question_ids')
        for question in state["exam"]["questions"]:
            id = question['id']
            if requested and id not in requested:
                task: QuestionTask = {
                    "id": id,
                    "do_grade": False,
                    "evaluation": evaluations[id],
                    "logger": logger
                }
                tasks.append(Send("grader", task))
            else:
                task: QuestionTask = {
                    "id": id,
                    "question": json.dumps(question, ensure_ascii=False),
                    "rubric": json.dumps(rubrics.get(id, []), ensure_ascii=False),
                    "student_answers": json.dumps(answers.get(id, []), ensure_ascii=False),
                    "review": json.dumps(audits.get(id, []), ensure_ascii=False),
                    "logger": logger
                }
                tasks.append(Send("grader", task))
        return tasks


    def node_grader(self, state: QuestionTask) -> GradingState:
        """Run one isolated grader and tag its output so it can be merged cleanly."""
        logger = state["logger"]
        if not state.get('do_grade', True):
            data = state['evaluation']
            result = self.graders[state["id"]].response_format(**data)
            text = self.graders[state["id"]].parse_result(result)
            return {
                "grading_workers_results": [{"id": state["id"], "result_json": data, "result_str": text}],
                "messages": []
            }

        output = self.graders[state["id"]].invoke(state)

        logger.log_messages(output["messages"], step_label=f"Grader-{state['id']}")
        return {
            "grading_workers_results": [{"id": state["id"], "result_json": output["result_json"], "result_str": output["result_str"]}],
            "messages": output["messages"]
        }


    def node_synthesizer(self, state: GradingState) -> State:
        """Keep the latest result per question, validate it, and calculate the exam total."""
        logger = state["logger"]
        logger.log("Synthesizing parallel grading results.", level="debug", step_label="Synthesizer")

        latest = {item["id"]: item for item in state.get("grading_workers_results", [])}
        evaluations = [latest[key]["result_json"] for key in sorted(list(latest.keys()))]
        total = sum(item["total_points"] for item in evaluations)
        text = self.parse_result(total, [latest[key]["result_str"] for key in sorted(list(latest.keys()))])

        return {
            "grading_result": {"total_mark": total, "evaluations": evaluations}, "grading_result_str": text,
            "messages": state['messages']
        }


    @staticmethod
    def parse_result(total: int, results: list[str]) -> str:
        """Join each question report beneath the aggregate total for a readable summary."""
        text = (
f"""Total Mark: {total}

Evaluations:
{'\n\n\n'.join(results)}
"""
        )

        return text


    def compile(self):
        """Build the map/reduce graph used to grade questions in parallel."""
        graph = StateGraph(State)

        graph.add_node("grader", self.node_grader)
        graph.add_node("synthesizer", self.node_synthesizer)

        graph.add_conditional_edges(START, self.map_questions, ["grader"])
        graph.add_edge("grader", "synthesizer")
        graph.add_edge("synthesizer", END)

        return graph.compile()

    def invoke(self, state: State) -> State:
        """Execute the grading fan-out and aggregation."""
        return self.graph.invoke(state)

    def skip(self, results_dir: Path, read_results: bool, logger: Logger | None) -> State:
        """Reuse the most recent final grading result."""
        if logger:
            logger.log("Grading step skipped as per configuration.", level="warning", step_label="Grader")

        if read_results:
            paths = sorted(results_dir.glob("grading_final*.json"))
            if not paths:
                raise FileNotFoundError(f"No prior grading result in {results_dir}")

            data = io.read_json(paths[-1], logger)
            results = [self.graders['Q1'].response_format(**eval) for eval in data.get("evaluations", [])]
            text = self.parse_result(data.get('total_mark', 0), [self.graders['Q1'].parse_result(result) for result in results])

            return {
                "grader_result_json": data,
                "grader_result_str": text
            }

        return {
            "grading_result": {"total_mark": 0, "evaluations": []},
            "grading_result_str": "Total Mark: 0\n\nEvaluations:\n"
        }
