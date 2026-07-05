# Run 004: Two-Teacher Batch, With All Run-003 Fixes Applied

**Script:** `examples/run_two_teachers_demo.py`
**Date:** 2026-07-05
**Providers:** Same as run 003 -- Groq (`llama-3.3-70b-versatile` Teacher A, `llama-3.1-8b-instant` Student, one Judge) + Gemini (`gemini-2.5-flash` Teacher B, one Judge), 10 test cases total.
**Executed against:** commit `72f7613`, which contains all four fixes motivated by run 003: `JudgeEngine.check_pattern` (crash-safe, `re.DOTALL`), the Audit Gate's new golden-reference self-check with bounded retry, the strengthened Teacher prompt, and Gemini's thinking-disabled/larger-token-budget client.

This document is written *before* executing the script. Run 003's `ANALYSIS.md` explicitly left "does the pass rate actually improve now?" as an open question for this run to answer -- it is not assumed to already be yes.

## What we expect to achieve

1. **Gemini should now succeed as Teacher.** Run 003 saw 0/5 Gemini-teacher generations succeed, root-caused entirely to thinking tokens exhausting the output budget. That specific mechanism is now fixed (thinking disabled, budget raised to 2048). We expect most or all 5 Gemini-taught test cases to at least generate and parse successfully -- though the Audit Gate's stricter self-check (new in this run) may still reject some for the same self-consistency reasons Groq's generations sometimes hit.
2. **No unhandled crashes.** Run 003 crashed one test case with a raw `re.error`. `JudgeEngine.check_pattern` now catches that. We expect zero Python tracebacks propagating out of the pipeline itself (an Audit Gate `RuntimeError` after exhausting retries is an *expected*, handled outcome recorded as an `error` entry -- not a crash).
3. **A meaningfully higher completion rate than run 003's 4/10.** We are not committing to a specific number, since some Audit Gate rejections (and retries) are expected to still occur -- that's the retry loop working as designed, not a failure. But given the DOTALL fix alone was observed (in isolated testing) to cut self-consistency failures roughly in half, and the bounded retry gives each question up to 3 attempts, we'd consider anything meaningfully above 4/10 a genuine improvement, and anything at or below it a sign the fixes didn't address the real bottleneck.
4. **Some Truth Gate triggers are still expected and fine** -- a genuinely incomplete or incorrect Student answer should still be caught. What we're specifically watching for is whether the *rate* looks more plausible than run 003's 8/8 (100%), which was itself mostly an artifact of the bugs this run's fixes address, not a reflection of real Student answer quality.
5. **The run completes within a few minutes, no hangs, no unrecovered rate-limit errors** -- same free-tier hardening as before, now with the added token cost of occasional Audit Gate retries factored in (each retry re-invokes the Teacher, so the true call count for this run may exceed the 40-call baseline from run 002/003 by some margin).

## What we are explicitly not expecting to control

Exactly which questions get generated, how many retries (if any) each one needs, and the exact final consensus scores are still left to the real models -- this run validates whether the *fixes* actually move the needle on the real, measured problems run 003 found, not any specific score.
