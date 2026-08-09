# other_architectures/single_agent/tools/grader_tools.py
"""Fuzzy matching and Java validation tools for question graders."""
import os
import re
import json
import difflib
import tempfile
import subprocess
from typing import Union, List, Dict

from langchain_core.tools import tool
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


# Define any custom tools here that can be used by the grader agent.


@tool
def calculate_total_score(sub_scores: Union[str, Dict[str, float]]) -> str:
    """
    Calculates the total points for a question by summing the awarded points of its sub-questions.
    Call this tool before finalizing the evaluation of a parent question to ensure accurate arithmetic.

    Args:
        sub_scores: A dictionary or JSON string mapping sub-question IDs to their awarded points (e.g., '{"Q1.1": 3, "Q1.2": 2}').

    Returns:
        A JSON string containing the calculated total score and the parsed breakdown.
    """
    try:
        # Handle stringified JSON input from the agent
        if isinstance(sub_scores, str):
            scores_dict = json.loads(sub_scores)
        else:
            scores_dict = sub_scores

        # Ensure all provided values are numeric
        if not all(isinstance(v, (int, float)) for v in scores_dict.values()):
            return "TOOL ERROR: All sub-scores must be numeric values."

        # Calculate the sum
        total_score = sum(scores_dict.values())
        
        # Determine if the result can be formatted as an integer (e.g., 5.0 -> 5)
        if total_score.is_integer():
            total_score = int(total_score)

        return json.dumps({
            "status": "SUCCESS",
            "message": "Total score calculated successfully.",
            "calculated_total": total_score,
            "breakdown": scores_dict
        }, indent=2)

    except json.JSONDecodeError:
        return "TOOL ERROR: Invalid JSON format provided for sub_scores."
    except Exception as e:
        return f"TOOL ERROR during total score calculation: {str(e)}"


@tool
def fuzzy_keyword_match(student_answer: str, accepted_terms: Union[List[str], str], threshold: float = 0.4) -> str:
    """
    Checks if a student's answer conceptually or textually matches accepted technical terms, accounting for OCR errors and minor typos.
    Call this tool whenever grading short-answer or fill-in-the-blank questions.

    Args:
        student_answer: The raw text answer submitted by the student.
        accepted_terms: A list of accepted terms or a JSON string list of accepted terms.
        threshold: The similarity threshold between 0.0 and 1.0 (default is 0.65).

    Returns:
        A string indicating 'MATCH FOUND' or 'NO MATCH', detailing similarity scores and matched terms.
    """
    if isinstance(accepted_terms, str):
        try:
            accepted_terms = json.loads(accepted_terms)
        except Exception:
            accepted_terms = [t.strip() for t in accepted_terms.split(",")]

    student_clean = re.sub(r'[^\w\s]', '', student_answer.lower().strip())
    student_tokens = set(student_clean.split())

    for term in accepted_terms:
        term_clean = re.sub(r'[^\w\s]', '', term.lower().strip())
        term_tokens = set(term_clean.split())

        # Exact or substring match
        if term_clean in student_clean or student_clean in term_clean:
            return f"MATCH FOUND: '{student_answer}' directly matches accepted concept '{term}'. Award full points."

        # Token set overlap for multi-word phrases
        if term_tokens and term_tokens.issubset(student_tokens):
            return f"MATCH FOUND: Key concept tokens for '{term}' present in '{student_answer}'. Award full points."

        # Sequence similarity ratio
        similarity = difflib.SequenceMatcher(None, student_clean, term_clean).ratio()
        if similarity >= threshold:
            return f"MATCH FOUND: '{student_answer}' matches accepted term '{term}' (Similarity: {similarity:.2f}). Award points."

    return f"NO MATCH: '{student_answer}' does not match any accepted terms {accepted_terms} above the threshold ({threshold})."


@tool
def execute_java_snippet(class_name: str, base_code: str, student_snippet: str, expected_output: str) -> str:
    """
    REQUIRED TOOL for Java programming questions. Compiles and executes code to verify standard output.

    Args:
        class_name: The exact public class name (e.g., "Exams").
        base_code: Boilerplate template containing the exact placeholder "CODE".
        student_snippet: The code submitted by the student.
        expected_output: Expected stdout string.

    Returns:
        Execution status string. If SUCCESSFUL, award full marks. If FAILED, review stderr for partial credit.
    """
    full_code = base_code.replace("CODE", student_snippet) if "CODE" in base_code else f"{base_code}\n{student_snippet}"

    with tempfile.TemporaryDirectory() as temp_dir:
        java_file_path = os.path.join(temp_dir, f"{class_name}.java")
        with open(java_file_path, "w", encoding="utf-8") as f:
            f.write(full_code)

        try:
            # Step 1: Compilation
            compile_proc = subprocess.run(
                ["javac", java_file_path],
                capture_output=True, text=True, timeout=10, shell=False
            )
            if compile_proc.returncode != 0:
                return f"COMPILATION FAILED:\n{compile_proc.stderr}\n\nInstruction: Evaluate code logic manually for partial credit."

            # Step 2: Execution
            run_proc = subprocess.run(
                ["java", "-cp", temp_dir, class_name],
                capture_output=True, text=True, timeout=10, shell=False
            )

            actual_out = run_proc.stdout.strip()
            exp_out = expected_output.strip()

            similarity = difflib.SequenceMatcher(None, actual_out.lower(), exp_out.lower()).ratio()

            if similarity >= 0.80 or exp_out in actual_out:
                return f"EXECUTION SUCCESSFUL.\nStudent Output:\n{actual_out}\nExpected Output:\n{exp_out}"
            else:
                return f"EXECUTION COMPLETED WITH OUTPUT MISMATCH.\nStudent Output:\n{actual_out}\nExpected Output:\n{exp_out}"

        except subprocess.TimeoutExpired:
            return "EXECUTION FAILED: Execution timed out (possible infinite loop)."
        except Exception as e:
            return f"SYSTEM ERROR during execution: {str(e)}"


@tool
def check_java_syntax(student_snippet: str) -> str:
    """
    Performs static checks on a Java code snippet (bracket balance, semicolon usage, control flow structure).
    Use this tool when code fails compilation to determine appropriate partial credit.

    Args:
        student_snippet: The raw code snippet submitted by the student.

    Returns:
        A summary analysis of structural correctness and syntax anomalies.
    """
    open_brackets = student_snippet.count("{") - student_snippet.count("}")
    open_parens = student_snippet.count("(") - student_snippet.count(")")
    has_semicolons = ";" in student_snippet

    issues = []
    if open_brackets != 0:
        issues.append(f"Unbalanced curly braces (Difference: {open_brackets})")
    if open_parens != 0:
        issues.append(f"Unbalanced parentheses (Difference: {open_parens})")
    if not has_semicolons and len(student_snippet.strip().splitlines()) > 1:
        issues.append("Missing termination semicolons")

    if not issues:
        return "SYNTAX ANALYSIS: Basic structural logic appears sound."
    return f"SYNTAX ANALYSIS ISSUES FOUND: {', '.join(issues)}. Consider partial credit unless the algorithmic logic is flawed."


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
