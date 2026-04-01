# src/agents/qa.py
from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Literal
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent


# QA graph state schema
class InputState(TypedDict):
	student_answers: List[Dict[str, Any]] | str
	exam: str
	rubric: str
	criteria: str
	grading: str
	regrade_counter: int

	messages: List[AnyMessage]

class OutputState(TypedDict):
	qa_result: str
	regrade: Literal["<REGRADE_REQUIRED>", "<NO_REGRADE>"]

	messages: List[AnyMessage]

class QaState(InputState, OutputState):
	pass

# Schema for structured output to use in QA
class Result(BaseModel):
    """Structured output for the QA Auditor Agent."""
    regrade: Literal["<REGRADE_REQUIRED>", "<NO_REGRADE>"] = Field(description="Strict decision on whether the Grader must fix errors.")
    identified_errors: List[str] | None = Field(description="Specific list of errors made by the Grader (e.g., 'Math error on Q2', 'Missed missing keyword in Q3.1'). Empty if no regrade is required.")
    qa_justification: str = Field(description="Overall explanation of the audit findings and instructions for the Grader on how to correct them.")

# Prompt template
template = ChatPromptTemplate(
	[
		("system",
			[
				{
					"type": "text",
					"text":
"""You are a strict Quality Assurance Auditor for an automated grading system. Your job is to audit the Grader Agent's work for accuracy, consistency, and strict adherence to the rubric.

YOUR AUDIT CHECKLIST:
1. RUBRIC DRIFT: Did the Grader award points for an answer that sounds confident but actually misses the core rubric criteria?
2. FACT-CHECKING: Did the Grader hallucinate? Verify technical definitions if you suspect the Grader accepted a factually incorrect answer.
3. ARITHMETIC VALIDATION: Verify that the sub-scores awarded by the Grader sum up correctly to the final total score.
4. PENALTY CONSISTENCY: Ensure the Grader did not overly penalize minor typos on technical terms unless spelling was explicitly part of the rubric.

RULES OF ENGAGEMENT:
- If you find ANY discrepancy, error, or bias, you must return "<REGRADE_REQUIRED>" and explicitly state what the Grader must fix.
- If the Grader's work is flawless, return "<NO_REGRADE>".
- You have {regrade_counter} regrade attempts remaining. If the counter is 0, you must return "<NO_REGRADE>" and simply log your final objections.""",
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
				{
					"type": "text",
					"text":
"""REVIEW CRITERIA:

{criteria}""",
            	}
        	]
    	),
			
		MessagesPlaceholder(variable_name="messages", optional=True),
		
		("human",
"""STUDENT ANSWERS:
{student_answers}

GRADER'S EVALUATION TO AUDIT:
{grading}

Audit the Grader's evaluation. Provide your decision and a detailed justification for whether a regrade is necessary."""
		)
	]
)

# Global variable to hold the agent instance
agent = None


def review(state: InputState) -> OutputState:
	"""Review the grading results based on the provided criteria and determine if a regrade is necessary."""

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
		qa_result = parse_result(result)
	else:
		qa_result = "Problem getting QA result."
		

	return {
		"qa_result": qa_result,
		"regrade": output.get("structured_response", {}).regrade,
		"messages": [message for message in output["messages"] if message not in state["messages"] and not isinstance(message, SystemMessage)]
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
		QaState,
		input_schema=InputState,
		output_schema=OutputState,
		)
	graph.add_node("review", review)

	graph.set_entry_point("review")
	graph.add_edge("review", END)

	return graph.compile()


def parse_result(
		result: Result
	) -> str:
	"""Parse the QA result into a human-readable format."""

	qa_result = (
f"""Regrade Decision: {result.regrade}

Identified Errors:
	- {"\n\t- ".join(
			result.identified_errors or ["None"]
		)
	}

QA justification:  {result.qa_justification}"""
	)

	return qa_result
