"""
A larger live run: two different models each act as Teacher, generating
several test cases apiece, every one of which is graded end-to-end by the
full pipeline (Audit Gate -> Student -> two Judges -> Ensemble Consensus)
against real free-tier APIs.

A fixed delay follows every real API call, regardless of provider, so this
run stays comfortably within each free tier's per-minute request and token
limits without needing to track exact budgets. Each test case is evaluated
independently and wrapped in its own error boundary: one bad generation or
a single unparsable response is recorded as an error entry rather than
aborting the whole batch, since a run of this size is expected to
occasionally hit a rough edge on at least one item.

Requires GROQ_API_KEY and GEMINI_API_KEY to be set in the environment.
"""

import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemini_client import make_gemini_client
from groq_client import make_groq_client

from llm_exam_board import (
    ConfigManager,
    EnsembleManager,
    EvaluationOrchestrator,
    JudgeEngine,
    QuestionGenerator,
    StudentResponder,
    TestCase,
)

TOPIC = "Multithreading"
DIFFICULTY = "medium"
QUESTIONS_PER_TEACHER = 5
STUDENT_MODEL = "llama-3.1-8b-instant"  # Groq

# A short pause after every real API call, regardless of provider, keeps
# this run comfortably within each free tier's per-minute limits without
# needing to track exact token/request budgets.
CALL_DELAY_SECONDS = 2.0


def throttled(client: Callable[[str], str]) -> Callable[[str], str]:
    """Wraps an LLM client so every call is followed by a fixed pause.

    Args:
        client: The underlying pipeline-compatible LLM client to wrap.

    Returns:
        Callable[[str], str]: A client with identical behavior, except it
        sleeps for CALL_DELAY_SECONDS after every call completes.
    """

    def _wrapped(prompt: str) -> str:
        result = client(prompt)
        time.sleep(CALL_DELAY_SECONDS)
        return result

    return _wrapped


def run_audit_gate(test_case: TestCase) -> None:
    """Applies the same solvability check EvaluationOrchestrator applies internally.

    Args:
        test_case: The TestCase to audit.

    Raises:
        RuntimeError: If the TestCase fails the solvability check.
    """
    if not test_case.golden_reference.strip():
        raise RuntimeError(f"Audit Gate rejected TestCase {test_case.test_case_id}: empty golden_reference.")
    if not test_case.rubric:
        raise RuntimeError(f"Audit Gate rejected TestCase {test_case.test_case_id}: empty rubric.")
    test_case.mark_audited()


def evaluate_test_case(
    test_case: TestCase,
    teacher_label: str,
    student_responder: StudentResponder,
    judges: List[JudgeEngine],
    ensemble_manager: EnsembleManager,
) -> Dict:
    """Runs one TestCase through the Audit Gate, Student, Judges, and Ensemble.

    Args:
        test_case: The TestCase to evaluate.
        teacher_label: A human-readable label identifying which Teacher
            model generated this TestCase.
        student_responder: The Student component that will answer it.
        judges: The Judge components that will grade the answer.
        ensemble_manager: The manager that reduces judges' scores to a consensus.

    Returns:
        Dict: A structured record of the TestCase, the Student's answer,
        every judge's GradedResponse, and the ensemble result.
    """
    run_audit_gate(test_case)
    student_answer = student_responder.answer(test_case)
    graded_responses = [judge.grade(test_case, student_answer) for judge in judges]
    ensemble_result = ensemble_manager.build_consensus(graded_responses)

    return {
        "teacher": teacher_label,
        "test_case": {
            "test_case_id": test_case.test_case_id,
            "prompt": test_case.prompt,
            "golden_reference": test_case.golden_reference,
            "rubric": [
                {
                    "item_id": item.item_id,
                    "description": item.description,
                    "max_points": item.max_points,
                    "is_critical": item.is_critical,
                    "validation_pattern": item.validation_pattern,
                }
                for item in test_case.rubric
            ],
        },
        "student_answer": {
            "answer_id": student_answer.answer_id,
            "answer_text": student_answer.answer_text,
        },
        "graded_responses": [
            {
                "judge_id": gr.judge_id,
                "raw_llm_score": gr.raw_llm_score,
                "final_score": gr.final_score,
                "truth_gate_triggered": gr.truth_gate_triggered,
                "feedback_summary": gr.feedback_summary,
                "deductions": [
                    {
                        "rubric_item_id": d.rubric_item_id,
                        "points_deducted": d.points_deducted,
                        "justification": d.justification,
                        "is_truth_gate_override": d.is_truth_gate_override,
                    }
                    for d in gr.deductions
                ],
            }
            for gr in graded_responses
        ],
        "ensemble_result": {
            "consensus_score": ensemble_result.consensus_score,
            "score_std_dev": ensemble_result.score_std_dev,
            "outlier_judge_ids": ensemble_result.outlier_judge_ids,
        },
    }


def main() -> None:
    """Generates QUESTIONS_PER_TEACHER test cases from each of two Teacher models and grades all of them."""
    config_manager = ConfigManager(
        {TOPIC: ["mutex locks", "race conditions", "deadlock avoidance", "thread pools", "producer-consumer queues"]}
    )

    teachers = [
        ("groq-llama-3.3-70b", QuestionGenerator(throttled(make_groq_client("llama-3.3-70b-versatile")), config_manager)),
        ("gemini-2.5-flash", QuestionGenerator(throttled(make_gemini_client("gemini-2.5-flash")), config_manager)),
    ]
    student_responder = StudentResponder(throttled(make_groq_client(STUDENT_MODEL)))
    judges = [
        JudgeEngine(throttled(make_groq_client("llama-3.3-70b-versatile")), "judge-groq-llama"),
        JudgeEngine(throttled(make_gemini_client("gemini-2.5-flash")), "judge-gemini-flash"),
    ]
    ensemble_manager = EnsembleManager()

    results: List[Dict] = []
    for teacher_label, teacher in teachers:
        for question_index in range(QUESTIONS_PER_TEACHER):
            try:
                test_case = teacher.generate_test_case(TOPIC, DIFFICULTY)
                record = evaluate_test_case(test_case, teacher_label, student_responder, judges, ensemble_manager)
                results.append(record)
                consensus = record["ensemble_result"]["consensus_score"]
                triggered = [gr["truth_gate_triggered"] for gr in record["graded_responses"]]
                print(
                    f"[{teacher_label} #{question_index + 1}] consensus={consensus} "
                    f"truth_gate_triggered={triggered}",
                    file=sys.stderr,
                )
            except Exception as exc:  # noqa: BLE001 - a batch run must not abort on one bad item
                results.append({"teacher": teacher_label, "question_index": question_index, "error": str(exc)})
                print(f"[{teacher_label} #{question_index + 1}] ERROR: {exc}", file=sys.stderr)

    successful = [r for r in results if "error" not in r]
    summary = {
        "total": len(results),
        "successful": len(successful),
        "errors": len(results) - len(successful),
        "truth_gate_trigger_count": sum(
            1 for r in successful for gr in r["graded_responses"] if gr["truth_gate_triggered"]
        ),
        "average_consensus_score": (
            sum(r["ensemble_result"]["consensus_score"] for r in successful) / len(successful) if successful else None
        ),
    }
    print(json.dumps({"summary": summary, "results": results}, indent=2))


if __name__ == "__main__":
    main()
