# src/agents/grader.py
from __future__ import annotations

from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent


# Grader graph state schema
class InputState(TypedDict):
	student_answers: List[Dict[str, Any]] | str
	exam: str
	rubric: str

class OutputState(TypedDict):
	grading_result: str

	messages: List[AnyMessage]

class GraderState(InputState, OutputState):
	pass

# Schema for structured output
class QuestionEvaluation(BaseModel):
    """Detailed breakdown for a single question."""
    question_id: str = Field(description="The ID or number of the question being graded (e.g., '1.1', '4.2').")
    awarded_points: float = Field(description="Points awarded for this specific question.")
    justification: str = Field(description="Detailed explanation of the score. MUST quote the exact part of the student's answer if points were deducted.")

class Feedback(BaseModel):
    """Structured output for Feedback."""
    positive_highlights: List[str] = Field(description="1-2 things the student did well or concepts they understood correctly.")
    areas_for_improvement: List[str] = Field(description="Constructive explanations of where and why the student lost points, avoiding punitive language.")
    encouraging_closing: str = Field(description="A brief, encouraging final sentence to motivate the student.")

class Result(BaseModel):
	"""Structured output for the Grader Agent."""
	total_mark: float = Field(description="The final total score calculated by summing all awarded_points.")
	evaluations: List[QuestionEvaluation] = Field(description="Step-by-step breakdown of the grading per question.")
	feedback: Feedback = Field(description="Pedagogical feedback for the student.")

# Prompt template
template = ChatPromptTemplate(
	[
		("system",
			[
				{
					"type": "text",
					"text":
"""You are an expert, meticulous academic Grader. Your objective is to evaluate student exam responses against a strict rubric.

CORE DIRECTIVES:
1. STEP-BY-STEP EVALUATION: Grade one question at a time. Compare the student's answer directly to the rubric criteria.
2. EVIDENCE-BASED SCORING: For every point deducted, you must quote the exact part of the student's answer that was incorrect or missing.
3. PROVIDING CONSTRUCTIVE FEEDBACK: Provide specific, actionable feedback. Avoid punitive language. Focus on guiding the student towards improvement.

Do not be lenient. If a mandatory keyword or concept from the rubric is missing, deduct the appropriate points.""",
				},
				{
					"type": "text",
					"text":
"""EXAM:

{exam}""",
				},
				{
					"type": "text",
					"text":
"""EXAM RUBRIC:

{rubric}""",
            	}
        	]
    	),
		
		("human",
"""STUDENT ANSWERS:
{student_answers}

Analyze the student's answers question by question. Determine the final mark and provide a detailed breakdown of where points were awarded or lost."""
		)
	]
)

# Global variable to hold the agent instance
agent = None


def grade(state: InputState) -> OutputState:
	"""Grade the student's exam answers based on the provided rubric and provide feedback."""

	# Create a chain that combines the prompt template with the agent.
	chain = template | agent

	# If student_answers is a list of dictionaries,
	# convert it to a string format for the prompt.
	if isinstance(state["student_answers"], List):
		state["student_answers"] = "".join(
			[
				"\n".join([f"{k}: {v}" for k, v in page.items()])
				for page in state["student_answers"]
			]
		)

	output = chain.invoke(state)

	# Extract the structured grading result from the agent's
	# output and parse it into a human-readable format.
	result = output.get("structured_response")
	if isinstance(result, Result):
		grading_result = parse_result(result)
	else:
		grading_result = "Problem getting grading result."

	return {
		"grading_result": grading_result,
		"messages": [message for message in output["messages"] if not isinstance(message, SystemMessage)]
	}


def build_agent(
		model: str,
		temperature: float,
		max_tokens: int | None
	):

	global agent
	agent = create_agent(

		model=init_chat_model(
			model=model,
			temperature=temperature,
			max_tokens=max_tokens,
		),

		response_format=Result,
	
	)
	
	graph = StateGraph(
		GraderState, 
		input_schema=InputState,
		output_schema=OutputState,
		)
	graph.add_node("grade", grade)

	graph.set_entry_point("grade")
	graph.add_edge("grade", END)

	return graph.compile()


def parse_result(
		result: Result
	) -> str:
	"""Parse the grading result into a human-readable format."""

	grading_result = (
f"""Total Mark: {result.total_mark}

Evaluations:
	{"\n\n\t".join(
			[
				(
					f"Question {evaluation.question_id}:\n"
					f"\tAwarded Points: {evaluation.awarded_points}\n"
					f"\tJustification: {evaluation.justification}"
				)
				for evaluation in result.evaluations
			]
		)
	}
{'-' * 80}

Feedback:
Positive Highlights:
	- {"\n\t- ".join(
			result.feedback.positive_highlights
		)
	}

Areas for Improvement:
	- {"\n\t- ".join(
			result.feedback.areas_for_improvement
		)
	}

{result.feedback.encouraging_closing}"""
	)

	return grading_result
