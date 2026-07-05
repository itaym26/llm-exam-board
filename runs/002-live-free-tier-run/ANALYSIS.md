# Run 002: Analysis

**Executed against:** commit `0867d50` (the `resilience.py` and Gemini-model fixes from `ISSUES.md` were committed as part of getting this run to succeed)
**Providers:** Groq (`llama-3.3-70b-versatile` Teacher, `llama-1.1-8b-instant`* Student, one Judge) + Gemini (`gemini-2.5-flash` Judge) — real API calls, free tier
**Script exit code:** 0
**Raw output:** [`output.json`](output.json)
**Issues encountered en route:** [`ISSUES.md`](ISSUES.md) (5 logged; 2 required code fixes, 2 are documented findings/limitations, 1 was a stale-bytecode false alarm)

<sub>*as configured in `run_free_demo.py`: `STUDENT_MODEL = "llama-3.1-8b-instant"`</sub>

## Result vs. expectation

| # | Expectation (from `EXPECTATIONS.md`) | Actual | Verdict |
|---|---|---|---|
| 1 | Valid TestCase: non-empty golden reference, >=1 rubric item | 5 rubric items, non-empty golden reference, `is_audited=true` | ✅ Match |
| 2 | Non-empty Student answer under Context Isolation | Full, well-formed `Counter` class implementation; `is_context_isolated=true` | ✅ Match |
| 3 | `final_score` bounded in `[0, sum(max_points)]` (here `[0, 100]`) | Both judges: `final_score=0.0` | ✅ Match (within bounds) |
| 4 | Every deduction justified with specific, non-generic text | All 8 deductions (4 per judge) name the specific rubric item and the missing pattern | ✅ Match |
| 5 | If Truth Gate triggers: `final_score=0.0` and >=1 override deduction | Both judges: `truth_gate_triggered=true`, `final_score=0.0`, 4 `is_truth_gate_override=true` deductions each | ✅ Match |
| 6 | If Truth Gate doesn't trigger: `final_score = raw_llm_score - LLM deductions` | N/A — Truth Gate triggered on both judges | N/A |
| 7 | `consensus_score` consistent with the two `final_score` values | Both 0.0 -> consensus 0.0, `outlier_judge_ids=[]` | ✅ Match |

**6/6 applicable expectations met** (expectation 6 didn't apply, since the Truth Gate triggered for both judges in this run). The pipeline's control flow held up correctly under real, unscripted model output.

## The real finding: a false Truth Gate trigger

The mechanically "correct" result above hides the actually interesting outcome. The Student's answer is **genuinely correct, idiomatic, thread-safe code**:

```python
def __init__(self):
    self._counter = 0
    self._lock = threading.Lock()

def increment(self):
    with self._lock:
        self._counter += 1
```

But the Teacher wrote rubric patterns hardcoded to one specific identifier name:

```json
"validation_pattern": "self\\.mutex\\s*=\\s*threading\\.Lock\\(\\)"
"validation_pattern": "def\\s+increment\\(self\\):\\s*with\\s+self\\.mutex:"
```

`self._lock` never matches `self\.mutex`, so all four critical patterns failed the static scan, and the Truth Gate zeroed a correct answer down to 0/100. **The Truth Gate did exactly what it was built to do** — it is a static, literal pattern check, not a semantic one — but the *rubric it was given* was too brittle to survive a trivially different (and equally valid) naming choice. This is a real calibration gap between two components (`QuestionGenerator`'s prompt for the Teacher, and `JudgeEngine`'s literal enforcement of whatever pattern it's handed) that only a live run with two independently-behaving real models could have surfaced — the mocked run 001 could not have shown this, because its mock Teacher and mock Student were both hand-written to agree on the exact pattern `pthread_mutex_lock`.

## A second finding: an unparsable judge silently defaults to "no problems found"

`judge-gemini-flash`'s `feedback_summary` is the fallback string `"Judge LLM response could not be parsed."` — Gemini's actual grading response failed `ResilienceUtils.clean_json_response` for a reason we could not diagnose after the fact (see `ISSUES.md`, Issue 5; the raw response wasn't captured). The consequence: `JudgeEngine.grade` treated the empty parse as *zero proposed deductions*, i.e., a perfect 100/100 `raw_llm_score` — indistinguishable, downstream, from a judge that actually reviewed the code and approved it. In this run that didn't matter, because the Truth Gate independently zeroed the score regardless of what either judge's LLM call produced or failed to produce. But this is a real, general risk: **on a TestCase where the Truth Gate does not trigger, a silently-failed judge would contribute a full, unearned score to the ensemble** rather than being excluded or flagged as "no data."

## Observations

- Both real findings above are about the *quality/robustness of the Teacher's generated rubric* and *the pipeline's handling of judge parse failures* — not about bugs in the six V14 modules' core logic, which behaved exactly per their contracts throughout.
- Getting to a clean run required two real code fixes (documented and committed): a genuine gap in `ResilienceUtils` (no recovery for raw control characters inside JSON string literals — a common LLM mistake when embedding multi-line code) and a stale default model reference (`gemini-2.0-flash`, no longer free-tier-eligible).
- The free-tier cost/rate-limit hardening from before this run held up without incident: no timeouts, no rate-limit errors, total wall-clock time for the whole run was well under a minute.

## Verdict

**Pass, with two follow-up items opened** (not yet fixed, by design — see `ISSUES.md` Issues 4 and 5 for the reasoning): (1) make `QuestionGenerator`'s Teacher prompt produce naming-agnostic `validation_pattern` regexes, and (2) decide how an unparsable judge response should be represented in the ensemble (silent full score vs. explicit abstention/exclusion) and add raw-response capture for future diagnosability. This run is a good baseline to compare the next live run against once those two items are addressed.
