# src/agents/grader.py
from __future__ import annotations

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime

from langchain_core.messages import AnyMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent


# Grader graph state schema
class InputState(TypedDict):
	student_answers: List[Dict[str, Any]] | str
	rubric: str

class OutputState(TypedDict):
	grading_result: Result

	messages: List[AnyMessage]

class GraderState(InputState, OutputState):
	pass

# Context schema
@dataclass
class Context:
	model: str
	temperature: float
	max_tokens: int | None

# Schema for structured output
class Result(TypedDict):
	"""A grading result dict with a mark and grading details."""
	mark: Annotated[int, "Total mark out of 100."]
	details: Annotated[str, "Detailed explanation of the grading, including marks for every question, which criteria were met or not met, and any key points covered/missed."]
	feedback: Annotated[str, "Detailed feedback for the student."]

# Prompt template
template = ChatPromptTemplate.from_messages(
	[
		("system",
"""You are an expert exam grader. Your task is to:
1. Assess based on the rubric if provided.
2. Identify key points covered/missed.
3. Provide a mark and detailed grading explanation.
4. Provide constructive and detailed feedback to the student.

Be fair but strict, and keep feedback concise but informative."""
		),
		("human",
"""STUDENT ANSWERS:
{student_answers}

RUBRIC:
{rubric}

Provide a mark, detailed grading explanation and a detailed feedback.

Return output as Dict with keys: mark (int), details (str) and feedback (str).
Keep response as short as possible."""
		)
	]
)


def grade(state: InputState, runtime: Runtime[Context]) -> OutputState:
	"""Grade the student's exam answers based on the provided rubric and provide feedback."""

	model = init_chat_model(
		model=runtime.context.model,
		temperature=runtime.context.temperature,
		max_tokens=runtime.context.max_tokens,
	)
	agent = create_agent(
		model=model,
		tools=[],
		response_format=Result
	)
	chain = template | agent

	if isinstance(state["student_answers"], List):
		state["student_answers"] = "".join(
					[
					"\n".join([f"{page}: {text}" for page, text in page.items()])
					for page in state["student_answers"]
					]
				)

	output = chain.invoke(state)

	# For workflow/logic testing without API cost.
	# output = {
	# 	"messages": template.invoke(state).to_messages()+[AIMessage(content=str({
	# 		"mark": 100,
	# 		"details": "Test: Grading",
	# 		"feedback": "Test: Feedback"
	# 	}))],
	# 	"structured_response": {
	# 		"mark": 100,
	# 		"details": "Test: grading",
	# 		"feedback": "Test: Feedback"
	# 	}
	# }

	return {
		"grading_result": output.get("structured_response", Result(mark=0, details="Problem getting grading result", feedback="Problem getting grading result")),
		"messages": output["messages"]
		}


def build_agent():
	graph = StateGraph(
		GraderState, 
		input_schema=InputState,
		output_schema=OutputState,
		context_schema=Context
		)
	graph.add_node("grade", grade)

	graph.set_entry_point("grade")
	graph.add_edge("grade", END)

	return graph.compile()
