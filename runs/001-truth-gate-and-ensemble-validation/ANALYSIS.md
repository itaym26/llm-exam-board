# Run 001: Analysis

**Executed against:** commit `ce4dc27`
**Environment:** Python 3.12.6, offline (mock LLM clients only, no real API calls)
**Script exit code:** 0
**Raw output:** [`output.txt`](output.txt)

## Result vs. expectation

| # | Expectation | Actual | Verdict |
|---|---|---|---|
| 1 | Scenario 1: `raw_llm_score` = 15.0 for both judges | 15.0 for both | ✅ Match |
| 2 | Scenario 1: `truth_gate_triggered` = `True` for both judges | `True` for both | ✅ Match |
| 3 | Scenario 1: `final_score` = 0.0 for both judges | 0.0 for both | ✅ Match |
| 4 | Scenario 1: override deduction present with justification | Present: "Truth Gate: critical pattern ... was not found ... via static scan." (10.0 pts) on both | ✅ Match |
| 5 | Scenario 1: `consensus_score` = 0.0 | 0.0 | ✅ Match |
| 6 | Scenario 2: `truth_gate_triggered` = `False` for both judges | `False` for both | ✅ Match |
| 7 | Scenario 2: `final_score` = 15.0 for both judges | 15.0 for both | ✅ Match |
| 8 | Scenario 2: `consensus_score` = 15.0 | 15.0 | ✅ Match |
| 9 | Scenario 3: `outlier_judge_ids` = `["judge-erratic"]` only | `['judge-erratic']` | ✅ Match |
| 10 | Scenario 3: `consensus_score` = 15.0 (majority, unaffected by outlier) | 15.0 | ✅ Match |
| 11 | Scenario 3: `score_std_dev` nonzero but not the outlier-decision basis | 6.600 | ✅ Match |

**11 / 11 expectations met.** All in-script assertions (`assert` statements in `run_demo.py`) also passed, and the process exited with code 0.

## Observations

- **The Truth Gate did exactly what it exists to do**: both mock judges hallucinated a perfect, unconditionally positive review ("Looks correct to me!", "Great job!") of code that was missing its one critical requirement, and the static scan caught it anyway, overriding the score to 0.0. This is the pipeline's core value proposition working as intended — without the Truth Gate, this run would have silently reported a passing score of 15.0 for broken code.
- **The override is fully traceable**: the deduction attached to each `GradedResponse` names the exact rubric criterion that failed the static check, so a human reviewing this result later doesn't have to guess why the score dropped to zero.
- **The outlier detection isolated the correct judge**, not the agreeing majority. This is worth calling out because a naive mean/standard-deviation approach was tried first during development and failed this exact scenario (it flagged all three judges as outliers, because the erratic judge's score dragged the mean and inflated the std-dev enough that the two agreeing judges also appeared to deviate). The current median/MAD-based implementation (see `orchestrator.py`) does not have that failure mode, and this run is the regression check for it.
- **`score_std_dev` (6.600) is reported for transparency only.** A reader skimming just that field might assume the ensemble is unreliable here; the point of this run is to confirm that the *consensus_score* is unaffected by that dispersion, because the outlier is excluded before the mean is taken.

## Caveats

This run uses mock LLM clients with fixed, scripted responses — it validates the pipeline's *control flow and decision logic*, not the quality of any real LLM's grading judgment. It does not exercise `ResilienceUtils`' recovery paths beyond the one markdown-fence-wrapped response the Teacher mock returns, nor does it test the Audit Gate's rejection path (every generated `TestCase` here has a non-empty golden reference and rubric, so the Audit Gate always passes silently). Those are reasonable candidates for a follow-up run once real LLM clients are wired in.

## Verdict

**Pass.** No deviations from expectations were observed. This run is a suitable baseline snapshot to compare future runs against, especially after any change to `JudgeEngine` or `EnsembleManager`.
