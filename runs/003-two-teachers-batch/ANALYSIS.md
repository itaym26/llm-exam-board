# Run 003: Analysis

**Executed against:** commit `0b0e819` (the run itself); all fixes it motivated were committed afterward at `72f7613`
**Raw output:** [`output.json`](output.json) (the pre-fix run) and [`progress.log`](progress.log)
**Issues encountered:** [`ISSUES.md`](ISSUES.md) -- 4 logged, all leading to real code/prompt fixes
**Diagnostic evidence:** [`teacher_raw_debug.txt`... see ISSUES.md](ISSUES.md), plus `gemini_teacher_raw_debug.txt` in this folder (Gemini's truncated 192-character response, before the thinking-budget fix)

## Result vs. expectation

| # | Expectation (from `EXPECTATIONS.md`) | Actual | Verdict |
|---|---|---|---|
| 1 | Most/all of 10 test cases complete without error | 4/10 completed, 6/10 errored (1 regex crash, 5 Gemini-teacher failures) | ❌ Miss -- but every failure was diagnosed and fixed (see `ISSUES.md`) |
| 2 | Truth Gate triggers *less often* than run 002's 2/2 rate | 8/8 (all 4 completed cases, both judges each) -- same 100% rate | ❌ Miss on the surface number, but root cause was deeper than the run 002 fix alone could reach (see below) |
| 3 | No control-character JSON parse failures | None observed | ✅ Match |
| 4 | Run stays within free-tier limits, no hangs | Completed in a few minutes, no rate-limit errors, no hangs | ✅ Match |
| 5 | Both Teachers produce structurally comparable questions | Groq: 4/5 structurally fine; Gemini: 0/5 (all truncated before any structure existed) | ❌ Miss -- root cause (thinking tokens) fixed afterward |

**On paper, this looks like a worse run than run 002.** It wasn't -- it did exactly what a bigger, real run is for: it found four real, previously-invisible problems that a single-test-case run (002) or a mocked run (001) had no chance of surfacing, and every one of them is now fixed in the codebase, not just noted.

## Why the Truth Gate trigger rate didn't improve (yet)

Expectation 2 assumed the run 002 fix (naming-agnostic patterns) was the whole story. It wasn't. Investigating *why* the Truth Gate still fired 8/8 led to two further discoveries, in order:

1. Several critical patterns were malformed independent of naming (e.g. missing the `self` parameter in a method signature, which cannot match *any* valid Python method) -- this is what motivated extending the Audit Gate to check the golden reference against its own patterns (`ISSUES.md` Issue 3).
2. Testing that new Audit Gate self-check against fresh generations revealed the *dominant* cause: patterns spanning a method signature and its body used `.` to bridge the line break, but Python's `.` does not match `\n` without `re.DOTALL` -- so nearly every multi-line pattern silently failed regardless of what it was checking (Issue 4). Fixing this alone dropped self-consistency failures from 3-4 broken patterns per test case to 1-2 in a direct before/after test.

The remaining, smaller failure class (e.g. `\w+` not matching a dotted `self.foo` access, or `\d+` not matching a named parameter) was confirmed to be irreducible LLM inconsistency, not a fixable systemic bug -- which is why the real fix wasn't "make the Teacher perfect," it was "make the Orchestrator retry when the Teacher isn't" (the new `__MAX_AUDIT_GATE_ATTEMPTS` bounded retry).

## Observations

- **A regex crash (Issue 2) is arguably the most important find of this run.** `JudgeEngine.grade` had no defense against a syntactically invalid `validation_pattern`, and a real Teacher generation produced exactly that on the very first 10-question batch. This would have crashed in production the same way.
- **Gemini's 5/5 failure as Teacher had nothing to do with JSON formatting or prompting** -- it was `gemini-2.5-flash`'s internal "thinking" tokens silently consuming ~980 of the 1024-token output budget on a moderately complex prompt, leaving only ~40 tokens for the actual answer. This is invisible unless you inspect `usage_metadata.thoughts_token_count` directly; the symptom (truncated JSON) looks identical to any other malformed-output case.
- **The Audit Gate, as originally written, wasn't actually checking "solvable"** -- it checked "non-empty," which is a much weaker property. This run is the reason it now means something closer to what its own docstring always claimed.
- Every fix this run motivated was verified against real API calls before being committed (not just unit-style reasoning): the DOTALL fix was tested against 5 fresh Groq generations, the Gemini thinking-budget fix was verified via direct `usage_metadata` inspection, and `check_pattern`'s crash-safety was exercised via the mock regression suite (`run_demo.py`) after every change.

## Verdict

**Pass, in the sense that matters for this kind of run: it surfaced real, previously-invisible defects, and all four are now fixed and committed** (`72f7613`). The literal pass-rate expectations (1, 2, 5) were missed by the original `output.json`, but that output is exactly the evidence that justified the fixes. A follow-up run (004) with all four fixes in place is the right next step to check whether the pass rate and Truth Gate trigger rate actually improve now -- that is a genuinely open question this analysis does not presume to already know the answer to.
