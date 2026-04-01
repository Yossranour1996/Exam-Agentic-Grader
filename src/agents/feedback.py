# src/agents/feedback.py
from __future__ import annotations

from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent


# Feedback graph state schema
class InputState(TypedDict):
	student_answers: List[Dict[str, Any]] | str
	exam: str
	rubric: str
	grading: str

class OutputState(TypedDict):
	feedback_result: str

	messages: List[AnyMessage]

class FeedbackState(InputState, OutputState):
	pass

# Schema for structured output to use in feedback
class Result(BaseModel):
    """Structured output for the Pedagogical Feedback Agent."""
    positive_highlights: List[str] = Field(description="1-2 things the student did well or concepts they understood correctly.")
    areas_for_improvement: List[str] = Field(description="Constructive explanations of where and why the student lost points, avoiding punitive language.")
    encouraging_closing: str = Field(description="A brief, encouraging final sentence to motivate the student.")

# Prompt template
template = ChatPromptTemplate(
	[
		("system",
			[
				{
					"type": "text",
					"text":
"""You are an encouraging Pedagogical Coach. Your goal is to translate the Grader Agent's technical evaluations into constructive, student-facing feedback.

YOUR FEEDBACK STRUCTURE MUST INCLUDE:
1. What Went Well: Highlight at least one concept the student understood correctly.
2. Areas for Improvement: Clearly explain *why* they lost points on specific questions without being condescending.
3. Actionable Next Steps: Use your resource tools to provide specific topics, syllabus areas, or documentation links the student should study to improve.

TONE GUIDELINES:
- Use a growth mindset tone (e.g., "You haven't mastered this yet" instead of "You failed this").
- Speak directly to the student ("You did well on...", "Your answer to Question 2...").
- Do not mention the "Grader Agent" to the student. Present the feedback as a unified voice.""",
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
				},
        	]
    	),		

		("human",
"""STUDENT ANSWERS:
{student_answers}

GRADER'S EVALUATION:
{grading}

Synthesize this data into a structured, student-facing feedback report."""
		)
	]
)

# Global variable to hold the agent instance
agent = None


def fb(state: InputState) -> OutputState:
	"""Generate feedback for the student based on the grading response and the provided rubric."""

	# Create a chain that combines the prompt template with the agent.
	chain = template | agent

	# If student_answers is a list of dictionaries,
	# convert it to a string format for the prompt.
	if isinstance(state["student_answers"], List):
		state["student_answers"] = "\n".join(
			[
				"\n".join([f"{k}: {v}" for k, v in page.items()])
				for page in state["student_answers"]
			]
		)

	output = chain.invoke(state)

	# Extract the structured feedback result from the agent's
	# output and parse it into a human-readable format.
	result = output.get("structured_response")
	if isinstance(result, Result):
		feedback_result = parse_result(result)
	else:
		feedback_result = "Problem getting feedback result."

	return {
		"feedback_result": feedback_result,
		"messages": [message for message in output["messages"] if not isinstance(message, SystemMessage)]
		}


def build_agent(
		model: str,
		temperature: float,
		max_tokens: int | None
	):

	global agent
	agent = create_agent(
		
		model = init_chat_model(
			model=model,
			temperature=temperature,
			max_tokens=max_tokens,
		),

		response_format=Result,
	
	)	

	graph = StateGraph(
		FeedbackState,
		input_schema=InputState,
		output_schema=OutputState,
		)
	graph.add_node("feedback", fb)

	graph.set_entry_point("feedback")
	graph.add_edge("feedback", END)

	return graph.compile()


def parse_result(
		result: Result
	) -> str:
	"""Parse the grading result into a human-readable format."""
	
	feedback_result = (
f"""Positive Highlights:
	- {"\n\t- ".join(
			result.positive_highlights
		)
	}

Areas for Improvement:
	- {"\n\t- ".join(
			result.areas_for_improvement
		)
	}

{result.encouraging_closing}"""
	)

	return feedback_result
