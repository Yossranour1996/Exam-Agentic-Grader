# src/tools/grader_tools.py
import os
import difflib
import tempfile
import subprocess
from src.tools.retriever import create_retriever

from langchain_core.tools import tool

# Define any custom tools here that can be used by the grader agent.


@tool
def fuzzy_keyword_match(student_answer: str, accepted_terms: list[str], threshold: float = 0.7) -> str:
    """
    Checks if the student's fill-in-the-blank answer matches the technical term, 
    allowing for minor spelling mistakes (e.g., 'polymorphism' vs 'polimorfism').

    :param student_answer: Answer submitted by the student.
    :param accepted_terms: List of accepted technical terms for this answer.
    :param threshold: Similarity threshold for accepting answers.
    :return: Match result.
    """
    student_lower = student_answer.lower().strip()
    
    for term in accepted_terms:
        term_lower = term.lower().strip()
        # difflib calculates a similarity ratio between 0.0 and 1.0
        similarity = difflib.SequenceMatcher(None, student_lower, term_lower).real_quick_ratio()
        
        if similarity >= threshold:
            return (f"MATCH FOUND: '{student_answer}' is accepted as a valid spelling of '{term}' (Similarity: {similarity:.2f}). Award points.")
            
    return f"NO MATCH: '{student_answer}' does not closely match any accepted terms {accepted_terms}."


@tool
def execute_java_snippet(class_name: str, base_code: str, student_snippet: str, expected_output: str) -> str:
    """
    Java executor that compiles and runs a Java snippet to verify its output. 
    
    :param class_name: The name of the public class (e.g., "Exams" or "Agree").
    :param base_code: The full Java code provided in the exam with the word "CODE" where the snippet goes.
    :param student_snippet: The actual Java statement provided by the student.
    :param expected_output: The exact stdout string required to get full marks.
    :return: Execution result.
    """
    # Inject the student's code into the placeholder
    full_code = base_code.replace("CODE", student_snippet)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        java_file_path = os.path.join(temp_dir, f"{class_name}.java")
        with open(java_file_path, "w") as f:
            f.write(full_code)

        try:
            # 1. Compile the Java code
            compile_process = subprocess.run(
                ["javac", java_file_path],
                shell=True, capture_output=True, text=True, timeout=10
            )
            if compile_process.returncode != 0:
                return f"COMPILATION FAILED:\n{compile_process.stderr}"
 
            # 2. Execute the Java code
            run_process = subprocess.run(
                ["java", "-cp", temp_dir, class_name],
                shell=True, capture_output=True, text=True, timeout=10
            )
            
            output = run_process.stdout.lower().strip()
            expected = expected_output.lower().strip()

            similarity = difflib.SequenceMatcher(None, output, expected).ratio()
            
            if similarity >= 0.80:  # Allow for minor formatting differences
                return f"EXECUTION SUCCESSFUL. Output similar to expected.\nStudent Output:\n{output}\n\nExpected Output:\n{expected}"
            else:
                return f"EXECUTION COMPLETED, BUT OUTPUT MISMATCH.\nStudent Output:\n{output}\n\nExpected Output:\n{expected}"
                        
        except subprocess.TimeoutExpired:
            return "EXECUTION FAILED: Time limit exceeded (possible infinite loop)."
        except Exception as e:
            return f"SYSTEM ERROR during execution: {str(e)}"


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
