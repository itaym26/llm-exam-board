"""
A real, live run of the llm_exam_board pipeline against the Anthropic API.

Unlike examples/run_demo.py (which uses scripted mock clients), this script
makes real API calls and incurs real (small) API cost. It requires
ANTHROPIC_API_KEY to be set in the environment.

This is deliberately a *small* run: one topic, one generated test case, and
two judges built from two different models (a cheaper model and a stronger
one), so the Ensemble Consensus has genuine model diversity to reconcile
rather than two calls to the same model.

The pipeline steps are driven explicitly here (rather than through
EvaluationOrchestrator.run_single_evaluation) so that every intermediate
artifact -- the generated question, the golden reference, the rubric, and
the student's raw answer -- can be captured and inspected, not just the
final EnsembleResult.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anthropic_client import make_anthropic_client

from llm_exam_board import ConfigManager, EnsembleManager, JudgeEngine, QuestionGenerator, StudentResponder

TOPIC = "Multithreading"
DIFFICULTY = "medium"

TEACHER_MODEL = "claude-haiku-4-5-20251001"
STUDENT_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODELS = {
    "judge-haiku": "claude-haiku-4-5-20251001",
    "judge-sonnet": "claude-sonnet-5",
}


def run_audit_gate(test_case) -> None:
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
    for item in test_case.rubric:
        if not item.is_critical or not item.validation_pattern:
            continue
        match_result = JudgeEngine.check_pattern(item.validation_pattern, test_case.golden_reference)
        if match_result is not True:
            reason = "is not a valid regular expression" if match_result is None else "does not match the golden reference"
            raise RuntimeError(
                f"Audit Gate rejected TestCase {test_case.test_case_id}: critical rubric item "
                f"'{item.description}' has a validation_pattern that {reason}."
            )
    test_case.mark_audited()


def main() -> None:
    """Runs one real, live evaluation and prints every intermediate artifact as JSON."""
    config_manager = ConfigManager({TOPIC: ["mutex locks", "race conditions", "deadlock avoidance"]})
    question_generator = QuestionGenerator(make_anthropic_client(TEACHER_MODEL), config_manager)
    student_responder = StudentResponder(make_anthropic_client(STUDENT_MODEL))
    judges = [JudgeEngine(make_anthropic_client(model), judge_id) for judge_id, model in JUDGE_MODELS.items()]
    ensemble_manager = EnsembleManager()

    test_case = question_generator.generate_test_case(TOPIC, DIFFICULTY)
    run_audit_gate(test_case)
    student_answer = student_responder.answer(test_case)
    graded_responses = [judge.grade(test_case, student_answer) for judge in judges]
    ensemble_result = ensemble_manager.build_consensus(graded_responses)

    artifact = {
        "test_case": {
            "test_case_id": test_case.test_case_id,
            "topic": test_case.topic,
            "difficulty": test_case.difficulty,
            "prompt": test_case.prompt,
            "golden_reference": test_case.golden_reference,
            "is_audited": test_case.is_audited,
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
            "is_context_isolated": student_answer.is_context_isolated,
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
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
