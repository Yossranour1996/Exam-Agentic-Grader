# other_architectures/zero_shot/prompts/prompt.py
"""Prompt template used to evaluate answers against the rubric."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


PROMPT = """Given the following exam and rubric, grade the student's answers against the rubric to determine the total mark and provide a detailed evaluation for each question and sub-question.

## EXAM:
{exam}

## RUBRIC
{rubric}

## STUDENT ANSWERS:
{student_answers}"""


# Grader prompt template
prompt_template = ChatPromptTemplate([
	("human", [
		{"type": "text",
		"text": PROMPT},
	]),
	HumanMessage(content="""## OUTPUT FORMAT:
You MUST return your evaluation in the following JSON format:
{
  "total_mark": <integer>,
  "evaluations": [
	{
	  "question_id": "<string>",
	  "total_score": <integer>,
	  "sub_evaluations": [
		{
		  "sub_question_id": "<string>",
		  "score": <integer>,
		  "justification": "<string>"
		}
	  ]
	}
]"""),
])
