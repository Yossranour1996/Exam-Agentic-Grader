# Exam Agentic Grader

Exam Agentic Grader is a research prototype for studying LLM-based, rubric-guided grading of scanned exams. Its central research question is whether a specialized multi-agent system can grade more consistently and accurately than:

1. human grading;
2. a direct zero-shot LLM prompt; and
3. a single grading agent.

The repository contains the three automated grading conditions, a human-score reference file, and saved artifacts for 40 anonymized answer sheets. The main implementation is the multi-agent condition: it extracts answers from scanned PDFs, assigns separate agents to grade exam sections, audits the resulting scores, selectively regrades failed sections, and produces student feedback and analysis-ready exports.

> This is experimental research software, not a production assessment system. LLM output can be incorrect or nondeterministic even at temperature 0. Grades should be reviewed by a qualified human before they affect a student.

## Research design

All automated conditions use the same structured exam, rubric, and previously extracted student answers so that their grading architectures can be compared on a common task.

| Condition | Implementation | Grading approach | Additional controls |
| --- | --- | --- | --- |
| Multi-agent | `src/` | Four section-specific graders run in parallel | OCR/structuring agents, deterministic tools, QA audit, targeted regrading, feedback |
| Single-agent | `other_architectures/single_agent/` | One tool-using agent grades the whole exam | Structured output and grading tools; reuses extracted answers |
| Zero-shot | `other_architectures/zero_shot/` | One direct model invocation grades the whole exam | Prompt-only JSON response; reuses extracted answers |
| Human | `data/input/human_grading.csv` | Human-assigned total mark | Reference scores for comparison |

The repository stores system outputs but does not currently include a statistical evaluation script. Metrics such as absolute error against human scores, question-level agreement, correlation, consistency across repeated runs, tool use, latency, and cost can be calculated downstream from the JSON/Excel exports. Human scores should be treated as a comparison reference rather than an unquestionable ground truth; inter-rater agreement and adjudication are relevant limitations for a formal study.

## Multi-agent workflow

```text
answer-sheet PDF
      |
      v
PDF-to-image conversion -> page-level OCR -> answer structuring
                                             |
                                             v
                       Q1 grader --+
                       Q2 grader ---+-> score synthesis -> QA audit
                       Q3 grader ---+                       |
                       Q4 grader --+             failed sections only
                                                    |       ^
                                                    +-------+
                                                        |
                                                        v
                                             feedback and exports
```

The stages are:

1. **Preparation** loads the exam, scoring rubric, QA policy, paths, and run settings.
2. **Extraction** converts a sheet PDF to PNG pages, transcribes each page with a vision-capable model, and maps the transcription to canonical question IDs.
3. **Parallel grading** fans out Q1-Q4 to isolated grader agents. Each receives only its question, rubric section, and corresponding student answers.
4. **Synthesis** deterministically combines section results and totals their scores.
5. **Quality assurance** checks rubric adherence, evidence, partial-credit policy, and arithmetic. It can request another pass for only the question groups that failed audit, up to the configured limit.
6. **Feedback** converts the final grading evidence into highlights, improvement areas, and an encouraging closing.
7. **Export** writes human-readable and structured results, reports, logs, and flattened Excel rows.

Agent responses are validated with Pydantic schemas. Graders also have deterministic tools for totaling scores, fuzzy matching fill-in answers, and inspecting Java snippets; the QA agent has arithmetic and partial-credit tools. The optional retriever code is scaffolded but disabled because its knowledge base is empty.

## Repository layout

```text
.
|-- src/                         # Multi-agent implementation
|   |-- agents/                  # OCR, structuring, grading, QA, feedback agents
|   |-- graph/                   # Top-level LangGraph workflow
|   |-- subgraphs/               # Extraction, parallel grading, and export graphs
|   |-- prompts/                 # Agent instructions
|   |-- schemas/                 # Structured-response models
|   |-- tools/                   # Scoring, OCR, PDF, retrieval, and Java helpers
|   |-- core/                    # Runtime state, model config, and paths
|   `-- app.py                   # Multi-agent entry point
|-- other_architectures/
|   |-- single_agent/            # Single tool-using grader baseline
|   |-- zero_shot/               # Direct-prompt baseline
|   `-- data/                    # Baseline inputs and saved baseline outputs
|-- data/
|   |-- input/                   # Exam, rubrics, and human total marks
|   `-- output/                  # Multi-agent artifacts for each sheet
|-- test.ipynb                   # Exploratory notebook
|-- create_files.py              # Historical project-scaffolding helper
`-- requirements.txt
```

The committed sample is a 100-mark Java exam with four sections: short answers, true/false, fill-in-the-blank, and code completion. `exam_java.yaml` defines its question structure, `java_criteria.yaml` defines scoring rules, and `review_criteria.yaml` defines the independent QA checks.

## Prerequisites

- Python 3.12 or 3.13. The source uses modern f-string syntax not supported by Python 3.11, while some ML dependencies may not yet support Python 3.14.
- An OpenAI API key for the configured `gpt-4o` agents.
- [Poppler](https://poppler.freedesktop.org/) available on `PATH` when converting PDFs with `pdf2image`.
- A JDK with `java` and `javac` on `PATH` if Java snippet execution is used.
- Sufficient memory and a compatible PyTorch environment only if using the optional local Hunyuan OCR helper.

The primary workflow currently uses an OpenAI vision model for OCR. `src/tools/gemini_ocr.py` and `src/tools/hunyuan_ocr.py` are alternative standalone OCR helpers rather than nodes in the default graph.

## Installation

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `.env` file (the real file is ignored by Git):

```dotenv
OPENAI_API_KEY=your-key
# Only needed by optional helpers:
GOOGLE_API_KEY=your-key
HUGGINGFACEHUB_API_TOKEN=your-token
```

Never commit credentials or identifiable student data.

## Inputs and configuration

The main workflow reads:

- `src/core/config.yaml` for model names, temperatures, token limits, OCR DPI, and maximum regrade attempts;
- `src/core/paths.yaml` for input and output directories;
- `data/input/exams/exam_java.yaml` for the canonical exam;
- `data/input/rubrics/java_criteria.yaml` for scoring criteria; and
- `data/input/rubrics/review_criteria.yaml` for QA policy.

For a new run, place a PDF at `data/input/answer_sheets/<sheet_id>.pdf` and set the matching `sheet_id` in `src/app.py`. Answer-sheet PDFs are not included in the current repository snapshot; the saved extraction and grading artifacts are included.

The `InputState` in `src/core/state.py` exposes stage switches:

| Setting | Default | Purpose |
| --- | --- | --- |
| `do_extract` | `True` | Run extraction or reuse the newest structured-answer JSON |
| `do_pdf_to_imgs` | `True` | Convert the PDF or reuse existing PNG pages |
| `do_ocr` | `True` | Run page OCR or reuse the newest raw extraction |
| `do_shredder` | `True` | Structure OCR text or reuse structured answers |
| `do_grade` | `True` | Grade or reuse prior grading where required |
| `do_qa` | `True` | Audit grades and enable selective regrading |
| `do_feedback` | `True` | Generate student feedback |
| `do_export` | `True` | Run the export subgraph |
| `do_save_results` | `True` | Save JSON and text result files |
| `do_export_to_sheet` | `True` | Append rows to Excel workbooks |

Skipped extraction stages require compatible prior artifacts in that sheet's output directory. `max_regrade` is the number of QA-directed retries available.

> Current implementation note: `src/app.py` reads `dpi_default` from the `global` configuration block, but the supplied YAML places it under `agents.extractor`. Move that key to `global` or update the lookup before running the example entry point.

## Running the experiments

### Multi-agent condition

Edit the example `InputState` in `src/app.py` for the desired sheet and stage switches, then run from the repository root:

```powershell
python -m src.app
```

The checked-in example sets `do_extract: false` and `do_feedback: false`, so it expects an existing structured extraction for `sheet_001` and writes placeholder feedback. Enable those stages for a full end-to-end run and ensure the source PDF exists.

### Single-agent and zero-shot baselines

The baseline apps use imports and paths relative to `other_architectures`, and both require an existing `structured_answers*.json` for the selected sheet:

```powershell
Set-Location other_architectures
python -m single_agent.app
python -m zero_shot.app
```

Edit each baseline's `app.py` to select a different sheet. Their model and output settings live in their respective `core/config.yaml` and `core/paths.yaml` files.

## Outputs

Multi-agent results are written below `data/output/<sheet_id>/`; baseline results are below `other_architectures/data/output/<condition>/<sheet_id>/`.

Typical artifacts include:

```text
sheet_001/
|-- pages/                       # Rendered page images
|-- extracted_answers/
|   |-- page_*.txt               # Page transcriptions
|   |-- raw_extraction_*.json    # Page-level OCR records
|   `-- structured_answers_*     # Question-aligned answers (JSON and text)
|-- reports/
|   |-- grading_reports_*.json   # Grading passes, including regrades
|   `-- qa_reports_*.json        # QA passes
|-- grading_results/
|   |-- final_result_*           # Combined grading and feedback
|   |-- grading_final_*.json
|   |-- qa_final_*.json
|   `-- feedback_*.json
`-- logs/log_*.log               # Prompts, responses, tool calls, and stage events
```

The multi-agent exporter also appends flattened scores to `data/output/exports/results.xlsx` and tool-call counts to `data/output/exports/tools.xlsx`. These workbooks are the most convenient inputs for aggregate analysis. Run IDs use day/hour/minute timestamps; reruns within one minute can therefore target the same filenames, and JSON helpers may append to existing list-valued report files.

## Reproducibility and responsible use

- Record model versions, prompts, configuration, dependency versions, and run timestamps for each experiment; hosted model aliases can change over time.
- Repeat runs when estimating stability, even with temperature set to zero.
- Keep extraction quality separate from grading quality. Reusing a common extraction across conditions isolates the grading architecture but does not evaluate end-to-end OCR error.
- Report agreement at both total-score and question level, and inspect systematic differences in partial credit or answer type.
- De-identify answer sheets and logs. Model prompts and generated traces may contain student responses.
- Use human review and an appeal mechanism for any consequential deployment.

## Known limitations

- The repository contains experimental outputs, not a complete evaluation/analysis pipeline or automated test suite.
- The entry points are configured by editing Python dictionaries rather than command-line arguments.
- The baseline directories duplicate some utilities and input data, so changes must be synchronized deliberately.
- The optional course-material retriever has no configured documents and is not active in the default agents.
- External API calls create cost, privacy, availability, and model-version dependencies.
- The generated outputs are evidence from particular runs, not proof that one architecture is generally superior.

## License and citation

No license or citation metadata is currently included. Add a license before redistribution and add the project paper or preferred citation when it becomes available.
