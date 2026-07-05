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
│   ├── run_demo.py               # offline, mock-LLM walkthrough of the full pipeline
│   ├── run_real_demo.py          # single live run against the Anthropic API
│   ├── run_free_demo.py          # single live run against free-tier Groq + Gemini
│   ├── run_two_teachers_demo.py  # batch live run: 2 Teacher models x N questions each
│   ├── anthropic_client.py       # Callable[[str], str] client backed by Anthropic
│   ├── groq_client.py            # Callable[[str], str] client backed by Groq (free tier)
│   └── gemini_client.py          # Callable[[str], str] client backed by Gemini (free tier)
├── runs/                         # dated, tracked records of live runs -- see "Live runs" below
│   └── NNN-<name>/
│       ├── EXPECTATIONS.md       # written before running: what we expect and why
│       ├── ISSUES.md             # any problem hit: cause, fix, how we proceeded
│       ├── output.json           # the actual captured result
│       └── ANALYSIS.md           # expected vs. actual, written after running
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

## Live runs: what actually happened when this hit real models

Every live run (real API calls, not mocks) is tracked under [`runs/`](runs/), each in its own dated folder following the same discipline: **write down what we expect *before* running** (`EXPECTATIONS.md`), **capture the real output** (`output.json`), **log any problem as it happens** with its cause, fix, and how we proceeded (`ISSUES.md`), and **compare expected vs. actual afterward** (`ANALYSIS.md`). Nothing here is cherry-picked after the fact.

| Run | What it tested | Headline result |
|---|---|---|
| [001](runs/001-truth-gate-and-ensemble-validation/) | Mock LLMs, full pipeline sanity check | 11/11 expectations met — established the median/MAD ensemble fix works before any real API cost |
| [002](runs/002-live-free-tier-run/) | First live run (Groq + Gemini), 1 question | Pipeline held up structurally, but surfaced a **false Truth Gate trigger**: a genuinely correct student answer was zeroed because the Teacher hardcoded the variable name `mutex` instead of a naming-agnostic pattern |
| [003](runs/003-two-teachers-batch/) | 2 Teacher models x 5 questions (10 total) | Looked worse on paper (4/10 completed) but found 4 real bugs: a malformed regex could crash grading entirely, the Audit Gate never actually verified a rubric was usable, `.` doesn't match newlines in Python regex by default (breaking most multi-line patterns), and Gemini's "thinking" tokens were silently eating its own answer's token budget |
| [004](runs/004-two-teachers-batch-fixed/) | Same batch, all run-003 fixes applied | Zero crashes (was 1), Gemini's Teacher role fixed, Truth Gate trigger rate dropped 100% → 75%, and the project's **first real, non-zero, non-overridden consensus score** (55.0/100) — cut short by a hard Gemini free-tier quota (20 requests/day/model), not a bug |

### A concrete example, from run 004

For a "multithreaded banking system with two threads transferring funds" task, both judges independently found the same real flaw — an inconsistent lock-acquisition order in `transfer_funds` that could deadlock — and itemized real, non-Truth-Gate deductions for it. No override was needed; the LLM judges' own grading stood on its merits:

```json
{
  "judge_id": "judge-gemini-flash",
  "raw_llm_score": 70.0,
  "final_score": 70.0,
  "truth_gate_triggered": false,
  "deductions": [
    {
      "points_deducted": 30.0,
      "justification": "The student's `transfer_funds` method... creates a nested locking scenario where a thread might acquire the lock of `from_account` and then try to acquire the lock of `to_account`... If another thread tries to transfer money in the opposite direction at the same time... this leads to a classic deadlock situation.",
      "is_truth_gate_override": false
    }
  ]
}
```

(The other judge, `judge-groq-llama`, independently scored the same answer 40.0 for the same underlying reason — the ensemble's `consensus_score` of 55.0 is their mean, since neither was flagged as an outlier.)

A different question in the same run shows the Truth Gate doing its job — a judge that had praised the code (`raw_llm_score: 80.0`) got overridden to 0 because three required thread-safety patterns were never found:

```json
{
  "judge_id": "judge-groq-llama",
  "raw_llm_score": 80.0,
  "final_score": 0.0,
  "truth_gate_triggered": true,
  "deductions": [
    {
      "points_deducted": 20.0,
      "justification": "Truth Gate: critical pattern for 'The deposit method is thread-safe' was not found in the student's answer via static scan.",
      "is_truth_gate_override": true
    }
  ]
}
```

### Key insights that came out of real runs (not visible from mocks alone)

- **An LLM-authored regex will occasionally be outright invalid.** `JudgeEngine.check_pattern` now catches `re.error` instead of letting a single bad pattern crash grading for an entire answer.
- **Python's `.` does not match `\n` by default.** Almost every multi-line `validation_pattern` the Teacher wrote (checking "is X called somewhere in this method body") silently failed to match *even correct code* until `check_pattern` was fixed to use `re.DOTALL`. This was the single highest-impact fix found across all four runs.
- **The Audit Gate's "solvable" check was too weak.** It only checked for non-empty fields — never that the Teacher's own golden reference actually satisfies its own critical patterns. It now does, and retries generation up to 3 times if a rubric fails its own self-consistency check, since an LLM Teacher won't hit that bar every time.
- **Hardcoding a specific variable, attribute, or method name in a validation pattern is a real, recurring false-positive source.** A student naming their lock `self._lock` instead of the Teacher's `self.mutex` is exactly as correct — the Teacher prompt now explicitly forbids this and requires wildcards instead, unless the exact name is spelled out in the question itself.
- **Gemini 2.5 Flash's "thinking" tokens count against `max_output_tokens`.** A moderately complex prompt spent ~980 of a 1024-token budget on invisible reasoning, truncating the actual JSON answer to a few dozen tokens — indistinguishable from a formatting bug unless you inspect `usage_metadata.thoughts_token_count` directly. Fixed by disabling thinking and raising the budget.
- **Gemini's free tier caps at 20 `generate_content` calls per day per model, per project** — not just a per-minute limit, and shared across *every* use of that model that day (testing, debugging, and actual runs all count against the same 20). Budget for this explicitly when planning a run that uses Gemini in any role.

## License

MIT
