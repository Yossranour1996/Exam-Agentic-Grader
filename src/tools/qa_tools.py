# src/tools/qa_tools.py
"""Deterministic arithmetic checks available to the QA agent."""
import json
from typing import Union, Dict

from langchain_core.tools import tool
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


# Define any custom tools here that can be used by the quality assurance agent.


@tool
def validate_score_constraints(sub_scores: Union[str, Dict[str, float]], max_possible_score: float, total_awarded_score: float) -> str:
    """
    REQUIRED TOOL for verifying the Grader's arithmetic.
    Validates that sub-scores are non-negative, do not exceed max score, and sum exactly to total_awarded_score.

    Args:
        sub_scores: A dictionary or JSON string mapping sub-question IDs to awarded points (e.g., '{"Q1.1": 3.0, "Q1.2": 2.0}').
        max_possible_score: Absolute maximum points permitted for the question/exam.
        total_awarded_score: Final total points awarded by the Grader.

    Returns:
        'VALID' status or detailed 'INVALID' arithmetic error message.
    """
    try:
        if isinstance(sub_scores, str):
            scores_dict = json.loads(sub_scores)
        else:
            scores_dict = sub_scores
    except Exception as e:
        return f"INVALID: Failed to parse sub_scores payload. Error: {str(e)}"

    values = list(scores_dict.values())
    if not all(isinstance(v, (int, float)) and v >= 0 for v in values):
        return "INVALID: All sub-scores must be non-negative numeric values."

    calculated_total = round(sum(values), 2)
    total_awarded_score = round(total_awarded_score, 2)
    max_possible_score = round(max_possible_score, 2)

    if calculated_total > max_possible_score:
        return f"INVALID CRITICAL ERROR: Sum of sub-scores ({calculated_total}) exceeds maximum limit ({max_possible_score})."

    if calculated_total != total_awarded_score:
        return f"INVALID: Sum of sub-scores ({calculated_total}) does not equal total awarded score ({total_awarded_score})."

    return f"VALID: Sub-scores sum perfectly to {calculated_total}, matching total awarded score {total_awarded_score}."


@tool
def calculate_partial_credit(max_score: float, deductions: Union[str, Dict[str, float]]) -> str:
    """
    Calculates adjusted scores after applying specific rubric deductions.
    Call this tool when suggesting score modifications during a regrade request.

    Args:
        max_score: Maximum points possible for the item.
        deductions: Dictionary or JSON string mapping deduction reasons to point values.

    Returns:
        JSON string with net calculated score and itemized breakdown.
    """
    try:
        if isinstance(deductions, str):
            ded_dict = json.loads(deductions)
        else:
            ded_dict = deductions

        total_deduction = sum(ded_dict.values())
        net_score = max(0.0, max_score - total_deduction)

        return json.dumps({
            "max_score": max_score,
            "total_deductions": total_deduction,
            "net_score": net_score,
            "breakdown": ded_dict
        }, indent=2)
    except Exception as e:
        return f"TOOL ERROR: Failed to calculate partial credit: {str(e)}"


retriever = None
@tool
def retrieve_course_facts(query: str) -> str:
    """
    Searches official course textbooks, syllabi, and solution manuals for facts.

    Args:
        query: Conceptual or factual search query.

    Returns:
        Relevant source excerpts or a notice if no materials were found.
    """
    global retriever
    if retriever is None:
        return "RETRIEVAL NOTICE: Knowledge base retriever is not initialized. Proceed with default rubric criteria."

    try:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant facts found in course materials for this query."

        excerpts = "\n\n".join([f"Excerpt {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        return f"RETRIEVED FACTS:\n{excerpts}"
    except Exception as e:
        return f"TOOL ERROR: Course facts retrieval failed: {str(e)}"


@wrap_model_call
def force_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Force the use of any available tool."""
    request = request.override(tool_choice="any")
    return handler(request)
