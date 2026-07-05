"""
End-to-end demo of the llm_exam_board pipeline using mock LLM clients.

No real API keys are required: this script simulates the Teacher, Student,
and Judge LLM calls with plain Python functions, so the full Teacher ->
Audit Gate -> Student -> Judge(s) -> Ensemble flow can be exercised offline.

It walks through three scenarios:
  1. A student who omits critical logic (a mutex lock) -> the Truth Gate
     should zero out the score regardless of what the judges claim.
  2. A student who includes the critical logic -> the judges' scores stand.
  3. Three judges grading the same (correct) answer, one of whom is a wild
     outlier -> the Ensemble Consensus should isolate that outlier rather
     than letting it drag down the majority's agreement.
"""

import json
import sys
from pathlib import Path

# Allow running this script directly from the examples/ directory without
# installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_exam_board import (
    ConfigManager,
    EnsembleManager,
    EvaluationOrchestrator,
    JudgeEngine,
    QuestionGenerator,
    StudentResponder,
)


def teacher_llm(prompt: str) -> str:
    """Simulates the Teacher LLM: returns a deliberately noisy JSON response.

    Args:
        prompt: The generation prompt (ignored by this mock).

    Returns:
        str: A JSON payload wrapped in markdown fences and surrounding
        prose, to exercise ResilienceUtils' Micro-Resilience parsing.
    """
    payload = {
        "question": "Implement a thread-safe increment function for a shared counter in C using pthreads.",
        "golden_reference": "pthread_mutex_lock(&lock); counter++; pthread_mutex_unlock(&lock);",
        "rubric": [
            {
                "description": "Acquires a mutex lock before touching the shared counter",
                "max_points": 10,
                "is_critical": True,
                "validation_pattern": "pthread_mutex_lock",
            },
            {
                "description": "Uses clear, descriptive variable names",
                "max_points": 5,
                "is_critical": False,
            },
        ],
    }
    return "Sure, here you go:\n```json\n" + json.dumps(payload) + "\n```\nLet me know if you need anything else!"


def student_llm_missing_lock(prompt: str) -> str:
    """Simulates a Student that forgets the mutex lock entirely."""
    return "int increment(int counter) { counter++; return counter; }"


def student_llm_correct(prompt: str) -> str:
    """Simulates a Student that correctly guards the counter with a mutex."""
    return "pthread_mutex_lock(&lock); counter++; pthread_mutex_unlock(&lock);"


def make_judge_llm(points_deducted: float, feedback: str):
    """Builds a mock Judge LLM that always awards a fixed, hallucinated deduction.

    Args:
        points_deducted: The (non-critical) deduction this mock judge will
            always claim, regardless of what the student actually wrote.
        feedback: The free-text feedback summary this mock judge will return.

    Returns:
        Callable[[str], str]: A mock LLM client suitable for JudgeEngine.
    """

    def _judge_llm(prompt: str) -> str:
        deductions = (
            [{"rubric_item_id": "style", "points_deducted": points_deducted, "justification": "minor style nit"}]
            if points_deducted > 0
            else []
        )
        return json.dumps({"deductions": deductions, "feedback_summary": feedback})

    return _judge_llm


def print_ensemble_result(label: str, result) -> None:
    """Prints an EnsembleResult in a readable format for the demo output."""
    print(f"\n--- {label} ---")
    print(f"Consensus score: {result.consensus_score}")
    print(f"Score std dev:   {result.score_std_dev:.3f}")
    print(f"Outlier judges:  {result.outlier_judge_ids or 'none'}")
    for graded in result.graded_responses:
        print(
            f"  judge={graded.judge_id:<14} raw_llm_score={graded.raw_llm_score:<5} "
            f"final_score={graded.final_score:<5} truth_gate_triggered={graded.truth_gate_triggered}"
        )
        for deduction in graded.deductions:
            print(f"    - ({deduction.points_deducted} pts) {deduction.justification}")


def main() -> None:
    """Runs all three demo scenarios and prints their results."""
    config_manager = ConfigManager({"Multithreading": ["mutex locks", "deadlock avoidance"]})
    question_generator = QuestionGenerator(teacher_llm, config_manager)

    # Two judges that both praise the answer unconditionally (they will be
    # fooled by the missing mutex lock unless the Truth Gate intervenes).
    lenient_judges = [
        JudgeEngine(make_judge_llm(0, "Looks correct to me!"), "judge-gpt"),
        JudgeEngine(make_judge_llm(0, "Great job!"), "judge-claude"),
    ]
    ensemble_manager = EnsembleManager()

    # --- Scenario 1: missing critical logic ---
    orchestrator_bad = EvaluationOrchestrator(
        question_generator, StudentResponder(student_llm_missing_lock), lenient_judges, ensemble_manager
    )
    result_bad = orchestrator_bad.run_single_evaluation("Multithreading", "hard")
    print_ensemble_result("Scenario 1: student omits the mutex lock", result_bad)
    assert result_bad.consensus_score == 0.0, "Truth Gate should have zeroed the score."

    # --- Scenario 2: correct critical logic ---
    orchestrator_good = EvaluationOrchestrator(
        question_generator, StudentResponder(student_llm_correct), lenient_judges, ensemble_manager
    )
    result_good = orchestrator_good.run_single_evaluation("Multithreading", "hard")
    print_ensemble_result("Scenario 2: student includes the mutex lock", result_good)
    assert result_good.consensus_score == 15.0, "Full score should stand when critical logic is present."

    # --- Scenario 3: ensemble outlier detection ---
    judges_with_outlier = [
        JudgeEngine(make_judge_llm(0, "Great job!"), "judge-gpt"),
        JudgeEngine(make_judge_llm(0, "Great job!"), "judge-claude"),
        JudgeEngine(make_judge_llm(14, "Terrible, everything is wrong."), "judge-erratic"),
    ]
    orchestrator_outlier = EvaluationOrchestrator(
        question_generator, StudentResponder(student_llm_correct), judges_with_outlier, ensemble_manager
    )
    result_outlier = orchestrator_outlier.run_single_evaluation("Multithreading", "hard")
    print_ensemble_result("Scenario 3: one judge is a statistical outlier", result_outlier)
    assert result_outlier.outlier_judge_ids == ["judge-erratic"], "Only the erratic judge should be isolated."
    assert result_outlier.consensus_score == 15.0, "Consensus should reflect the agreeing majority, not the outlier."

    print("\nAll demo scenarios completed and passed their assertions.")


if __name__ == "__main__":
    main()
