# src/tools/qa_tools.py
import json
from src.tools.retriever import create_retriever

from langchain_core.tools import tool

# Define any custom tools here that can be used by the quality assurance agent.


@tool
def validate_score_constraints(sub_scores: str, max_possible_score: float, total_awarded_score: float) -> str:
    """
    Validates that the sum of the sub-scores exactly matches the total awarded score, 
    and that it does not exceed the maximum possible score for the question.
    
    :param sub_scores: JSON string mapping question IDs to their awarded scores.
    :param max_possible_score: Max total score for the entire exam.
    :param total_awarded_score: Total score awarded by the grader.
    :return: Validation result.
    """
    sub_scores = json.loads(sub_scores).values()

    if not all(isinstance(score, (float, int)) and score >= 0 for score in sub_scores):
        return "INVALID: All sub-scores must be non-negative numbers."
    
    calculated_total = sum(sub_scores)
    if calculated_total > max_possible_score:
        return f"INVALID CRITICAL ERROR: The sum of sub-scores ({calculated_total}) exceeds the maximum possible score ({max_possible_score})."
    
    if calculated_total != total_awarded_score:
        return f"INVALID: The sum of sub-scores ({calculated_total}) does not match the total awarded score ({total_awarded_score})."
    
    return f"VALID: Sub-scores sum to {calculated_total}, which is within the {max_possible_score} point limit and matches the total awarded score {total_awarded_score}."


# retriever = create_retriever()
retriever = None
@tool
def retrieve_course_facts(query: str) -> str:
    """
    Searches the official course textbook, syllabus, and solution manual for facts.

    :param query: Course retriever query.
    :return: Retrieved facts.
    """
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the course materials for this query."
        
        # Return the retrieved textbook context directly to the LLM
        combined_text = "\n\n".join([f"Source Excerpt:\n{doc.page_content}" for doc in docs])
        return f"RETRIEVED FACTS:\n{combined_text}"
        
    except Exception as e:
        return f"TOOL ERROR: Failed to retrieve documents. Error: {str(e)}"
