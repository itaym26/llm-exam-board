# Run 001: Truth Gate and Ensemble Validation

**Script:** `examples/run_demo.py`
**Date:** 2026-07-05
**Purpose:** Validate, with a small and fully deterministic offline run (mock LLM clients, no real API calls), that the two signature defense mechanisms of this pipeline behave as designed before we ever point it at a real LLM.

This document is written *before* running the script. `ANALYSIS.md` in this same folder is written after, comparing the actual output against these expectations.

## What we expect to achieve

### Scenario 1 — student omits the mutex lock
Both mock judges are configured to praise the answer unconditionally ("Looks correct to me!", "Great job!") and claim zero deductions — i.e., they will hallucinate a passing grade. The student's answer genuinely omits the mutex lock.

- **Expect:** `raw_llm_score` = 15.0 for both judges (the LLM's naive, hallucinated score).
- **Expect:** `truth_gate_triggered` = `True` for both judges.
- **Expect:** `final_score` = 0.0 for both judges — the Truth Gate must override the LLM's praise entirely, not just discount it.
- **Expect:** each `GradedResponse` carries a deduction with `is_truth_gate_override=True` explaining the missing pattern.
- **Expect:** ensemble `consensus_score` = 0.0.

### Scenario 2 — student includes the mutex lock
Same two lenient judges, same rubric, but the student's answer now contains the critical pattern.

- **Expect:** `truth_gate_triggered` = `False` for both judges (nothing to override).
- **Expect:** `final_score` = 15.0 for both judges (full marks, matching their raw LLM score).
- **Expect:** ensemble `consensus_score` = 15.0.

### Scenario 3 — one judge is a statistical outlier
Three judges grade the same correct answer: two agree (`final_score` = 15.0 each), one is erratic and claims a large 14-point deduction (`final_score` = 1.0).

- **Expect:** `outlier_judge_ids` = `["judge-erratic"]` — exactly the erratic judge, not the agreeing majority.
- **Expect:** `consensus_score` = 15.0 — computed from the two agreeing judges only, unaffected by the outlier.
- **Expect:** `score_std_dev` to be reported as nonzero (for transparency), even though it is *not* the basis for the outlier decision (the implementation uses a median/MAD-based modified Z-score specifically so a lopsided std-dev calculation can't cause the majority to be flagged instead of the actual outlier — see `orchestrator.py`).

## Why this matters

If either Truth Gate scenario fails, the entire premise of the pipeline collapses: an LLM judge that can be talked into praising broken code is worse than no judge at all, because it produces false confidence. If the outlier scenario fails, a single erratic judge could silently corrupt every ensemble score the pipeline produces. This run exists to catch exactly those two failure modes before any real LLM cost is spent.
