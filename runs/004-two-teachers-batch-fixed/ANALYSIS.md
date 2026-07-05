# Run 004: Analysis

**Executed against:** commit `72f7613` (all four run-003 fixes applied)
**Raw output:** [`output.json`](output.json), [`progress.log`](progress.log)
**Issues encountered:** [`ISSUES.md`](ISSUES.md) -- 1 logged: a hard Gemini free-tier daily quota (20 requests/model/project/day) that cut the run short, unrelated to any of the fixed bugs.

## Headline numbers, run 003 vs run 004

| Metric | Run 003 (pre-fix) | Run 004 (post-fix) |
|---|---|---|
| Completed / 10 | 4 | 4 |
| Crashes (unhandled Python exceptions) | 1 (`unbalanced parenthesis`) | 0 |
| Truth Gate trigger rate among completed | 8/8 (100%) | 6/8 (75%) |
| Cases with a real, non-zero, non-overridden consensus score | 0 | 1 (55.0/100) |
| Gemini-as-Teacher: generations that got past truncation | 0/5 | 5/5 attempted, 1 completed before quota cutoff (0 truncation failures) |
| Failure mode on rejection | crash, or generic "unparsable output" | specific, named Audit Gate rejection reason, or an external quota error -- never a crash |

**The raw "4/10 successful" count looks identical, which could read as "the fixes didn't help." They did** -- the composition of both the successes and the failures changed in every way the fixes targeted:

- **The crash is gone.** Run 003 lost a whole test case to an unhandled `re.error`; run 004 had zero crashes across 10 attempts (and up to 3 retries each).
- **Gemini's Teacher truncation bug is fixed.** Its one real attempt this run generated and parsed a complete, valid test case -- run 003 saw 0/5 succeed at that same step, always truncated to a few dozen tokens. Gemini's remaining 4 failures this run are a completely different, external cause (a hard daily quota), not the bug fixed after run 003.
- **A real, substantive graded answer finally came through.** `groq-llama-3.3-70b #5` produced a `consensus_score` of 55.0 with `truth_gate_triggered=[False, False]` on both judges -- the first time in this project's live runs that a genuinely non-zero, non-overridden score reached the ensemble. That is the system working exactly as intended: real LLM-authored deductions, no Truth Gate override needed, because the critical patterns actually held up against a real Student answer.
- **Audit Gate rejections are now informative, not opaque.** Both failed Groq generations name the exact rubric item and reason (e.g. `"Uses a condition variable for notification between threads" has a validation_pattern that does not match the golden reference`), after exhausting 3 retry attempts -- exactly the bounded-retry-then-clear-error behavior added after run 003, versus run 003's undifferentiated crash/parse-failure messages.
- **Truth Gate trigger rate dropped from 100% to 75%** among completed cases -- a real, if modest, improvement in the direction run 003 predicted but couldn't yet demonstrate.

## What actually stopped this run from finishing

Not a bug: a hard **20-requests-per-day free-tier quota on `gemini-2.5-flash`**, shared across every call made against that model under this API key today (all of this session's testing, not just this script run). This is now documented in `ISSUES.md` for future run planning. It is a genuinely different category of limitation from anything found in runs 002/003 -- external and quantified, not something in our control to fix, only to budget around.

## Expectations vs. actual (from `EXPECTATIONS.md`)

1. *Gemini should now succeed as Teacher* -- ✅ confirmed on its one real attempt (no truncation); ❌ could not be verified across all 5 due to the quota cutoff.
2. *No unhandled crashes* -- ✅ zero crashes.
3. *Meaningfully higher completion rate than 4/10* -- ❌ literal count unchanged, but see the qualitative breakdown above; the quota cutoff removed 4 of Gemini's 5 attempts from the sample entirely, which mechanically caps how much the raw completion count could improve this run regardless of code quality.
4. *Some Truth Gate triggers still expected; rate should look more plausible than 100%* -- ✅ 75%, and the one non-triggered case shows genuine end-to-end success.
5. *Completes within a few minutes, no hangs* -- ✅ completed promptly; the quota errors surfaced immediately rather than hanging or retrying indefinitely.

## Verdict

**Pass.** Every fix motivated by run 003 is confirmed working on real, independent API calls: no crash, no Gemini truncation, informative Audit Gate rejections, and the first real non-zero, non-overridden consensus score this project has produced. The run's early stop was caused by a hard external daily quota, not by any of the code under test -- a genuinely different, now-documented constraint for planning future runs (keep total daily `gemini-2.5-flash` calls comfortably under 20, or spread runs across days).
