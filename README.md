# llm-exam-board

A production-oriented **Teacher-Student-Judge** pipeline for evaluating LLMs with other LLMs — built around one core problem: an LLM judge will confidently praise broken code if you let it. This pipeline doesn't let it.

It follows the **V14 architecture** (see [`docs/SYSTEM_PROMPT_V14.md`](docs/SYSTEM_PROMPT_V14.md) for the full specification this repository implements).

## Why this exists

Enterprise-grade eval tooling assumes you have unlimited budget and a small army of human graders. This project assumes you have neither — just a handful of standard LLM APIs — and engineers around their two biggest failure modes: **hallucinated grading** (a judge that says "great job!" to code with an obvious bug) and **brittle parsing** (a pipeline that crashes because an LLM wrapped its JSON in a sentence).

## Architecture

```
 Teacher (QuestionGenerator)
      |  generates TestCase: prompt + golden_reference + dynamic rubric
      v
 Audit Gate (EvaluationOrchestrator)
      |  rejects the task if it isn't actually solvable
      v
 Student (StudentResponder)
      |  answers under Context Isolation (zero prior memory)
      v
 Judge(s) (JudgeEngine)  x N
      |  itemized deductions (LLM) + Truth Gate (static check)
      v
 Ensemble (EnsembleManager)
      |  consensus score, outlier judges isolated
      v
 EnsembleResult
```

### Defense mechanisms

| Mechanism | Component | Purpose |
|---|---|---|
| **Truth Gate** | `JudgeEngine` | Statically scans the student's answer for critical logic (e.g., a mutex lock). If it's missing, the score is zeroed out — the LLM's own praise is overridden, not just discounted. |
| **Micro-Resilience** | `ResilienceUtils` | Recovers a JSON object from noisy LLM text (markdown fences, trailing prose, trailing commas) without ever raising, so one malformed response can't crash a batch run. |
| **Ensemble Consensus** | `EnsembleManager` | Aggregates every judge's score for one answer using a median/MAD-based (Median Absolute Deviation) modified Z-score, so a single erratic judge is isolated as an outlier instead of dragging down the majority's agreement. |
| **Audit Gate** | `EvaluationOrchestrator` | Rejects a generated task outright if it has no golden reference or no rubric, before it ever reaches the Student. |
| **Context Isolation** | `StudentResponder` | Every answer is generated from a brand-new, self-contained prompt — no conversational memory leaks between tasks. |
| **Itemized Deductions** | `interfaces.DeductionItem` | Judges must justify every lost point against a specific rubric item; free-text feedback is informational only, never authoritative. |

## Project structure

```
llm-exam-board/
├── README.md
├── pyproject.toml
├── docs/
│   └── SYSTEM_PROMPT_V14.md      # the architecture spec this repo implements
├── examples/
│   └── run_demo.py               # offline, mock-LLM walkthrough of the full pipeline
└── src/
    └── llm_exam_board/
        ├── __init__.py
        ├── interfaces.py         # TestCase, StudentAnswer, RubricItem, DeductionItem, GradedResponse, EnsembleResult
        ├── resilience.py         # ResilienceUtils.clean_json_response
        ├── generators.py         # ConfigManager, QuestionGenerator (Teacher)
        ├── responders.py         # StudentResponder (Student)
        ├── judge_engine.py       # JudgeEngine (Judge + Truth Gate)
        └── orchestrator.py       # EvaluationOrchestrator (Audit Gate) + EnsembleManager
```

## Installation

```bash
pip install -e .
```

Requires Python 3.9+. No third-party dependencies — the pipeline is pure standard library, so it plugs into whichever LLM SDK you already use.

## Quick start

Every LLM-facing component takes a plain `Callable[[str], str]` — a function that sends a prompt and returns raw text. This keeps the pipeline provider-agnostic: plug in the Anthropic SDK, OpenAI SDK, or a local model, as long as it fits that shape.

```python
from llm_exam_board import (
    ConfigManager, QuestionGenerator, StudentResponder,
    JudgeEngine, EnsembleManager, EvaluationOrchestrator,
)

def call_teacher(prompt: str) -> str:
    ...  # e.g. anthropic_client.messages.create(...).content[0].text

def call_student(prompt: str) -> str:
    ...

def call_judge(prompt: str) -> str:
    ...

config_manager = ConfigManager({"Multithreading": ["mutex locks", "deadlock avoidance"]})
orchestrator = EvaluationOrchestrator(
    question_generator=QuestionGenerator(call_teacher, config_manager),
    student_responder=StudentResponder(call_student),
    judges=[JudgeEngine(call_judge, judge_id="judge-1")],
    ensemble_manager=EnsembleManager(),
)

result = orchestrator.run_single_evaluation(topic="Multithreading", difficulty="hard")
print(result.consensus_score, result.outlier_judge_ids)
```

## Running the demo

A fully offline demo — no API keys required — exercises the Truth Gate, Micro-Resilience, and outlier detection using mock LLM clients:

```bash
python examples/run_demo.py
```

## License

MIT
