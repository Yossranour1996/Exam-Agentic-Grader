# Agentic Grader

An experimental LangGraph pipeline for grading scanned exams.

The workflow converts answer-sheet PDFs to images, performs OCR, maps answers to
exam IDs, grades the four top-level questions in parallel, audits the combined
result, selectively regrades only failed question groups, generates feedback,
and exports JSON, text, and Excel reports.

## Configuration

- `data/input/exams/exam_java.yaml`: canonical exam structure.
- `data/input/rubrics/java_criteria.yaml`: canonical scoring rubric.
- `data/input/rubrics/review_criteria.yaml`: QA policy.
- `src/core/config.yaml`: model settings.
- `src/core/paths.yaml`: input and output locations.

Set `OPENAI_API_KEY` in `.env`, install `requirements.txt`, and run:

```powershell
python -m src.app
```

Python 3.11-3.13 is recommended; parts of the current LangChain stack emit
compatibility warnings on Python 3.14.
