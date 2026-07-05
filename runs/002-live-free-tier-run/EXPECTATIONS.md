# Run 002: Live Free-Tier Run

**Script:** `examples/run_free_demo.py`
**Date:** 2026-07-05
**Providers:** Groq (Teacher, Student, one Judge) + Gemini (one Judge) — real API calls, free tier, no mocks.
**Purpose:** Run the full Teacher -> Audit Gate -> Student -> Judge(s) -> Ensemble pipeline against real models for the first time, and capture what real (not scripted) LLM behavior looks like end to end.

This document is written *before* executing the script. Unlike run 001 (mocked, fully scripted, deterministic), the exact scores here cannot be predicted in advance — a real model decides them. What follows are *structural* expectations: properties the result must satisfy if every component is working correctly, not specific numbers.

## What we expect to achieve

1. **A valid TestCase is generated.** `golden_reference` is non-empty, and the rubric contains at least one item. (If not, the Audit Gate should reject it — that itself would be a valid, informative outcome, not a failure of the pipeline.)
2. **The Student produces a non-empty answer** to the generated prompt, under Context Isolation.
3. **Each judge's `final_score` is bounded**: between 0 and the sum of the rubric's `max_points`, inclusive.
4. **Every deduction is justified**: each `DeductionItem.justification` should be non-empty text tied to a specific rubric item, not generic filler.
5. **If the Truth Gate triggers**, the corresponding judge's `final_score` must be exactly 0.0, and at least one deduction must have `is_truth_gate_override=True` with a justification naming the missing pattern.
6. **If the Truth Gate does not trigger**, `final_score` should equal `raw_llm_score` minus any (non-override) LLM-proposed deductions.
7. **The ensemble's `consensus_score` is consistent with the two judges' `final_score` values** — either their mean (if neither is flagged an outlier) or one of them (if the other is excluded).

## What we are explicitly *not* expecting to control

Because a real Teacher LLM decides the question, rubric, and critical validation pattern, and a real Student LLM decides how to answer it, we cannot know in advance whether the Truth Gate will trigger on this particular run, or what the judges will actually deduct. That is the point: this run validates that the pipeline's *machinery* holds up under real, non-scripted model output — not that any specific score is produced.

## Process note

Per the working agreement for this run: any problem encountered during execution (bad API key, wrong model name, rate limit, malformed model output, etc.) is logged in `ISSUES.md` in this same folder as it happens, with the cause, the fix applied, and how we proceeded — rather than being silently worked around.
