"""
Orchestration layer for the V14 LLM Evaluation Pipeline.

This module contains EnsembleManager, which aggregates multiple judges'
scores for the same student answer into a single consensus while isolating
statistical outliers, and EvaluationOrchestrator, which drives the full
Teacher -> Audit Gate -> Student -> Judge(s) -> Ensemble flow end to end.
"""

import statistics
from typing import List

from .generators import QuestionGenerator
from .interfaces import EnsembleResult, GradedResponse, TestCase
from .judge_engine import JudgeEngine
from .responders import StudentResponder


class EnsembleManager:
    """Aggregates multiple GradedResponse instances into a single consensus.

    For a given student answer, several JudgeEngine instances may each
    produce their own GradedResponse. EnsembleManager identifies the median
    final_score across those responses, then isolates any judge whose score
    yields a modified Z-score (based on the Median Absolute Deviation, which
    is robust to the outliers themselves) beyond a configurable threshold,
    excluding such outliers from the final consensus score.

    Attributes:
        outlier_threshold (float): The modified Z-score a judge's deviation
            from the median must exceed to be classified as an outlier.
    """

    def __init__(self, outlier_threshold_init: float = 3.5) -> None:
        """Initializes the EnsembleManager.

        Args:
            outlier_threshold_init: The modified Z-score (based on Median
                Absolute Deviation) a judge's score must exceed, relative to
                the group median, to be classified as an outlier. Defaults
                to 3.5, the threshold recommended by Iglewicz and Hoaglin.

        Raises:
            ValueError: If outlier_threshold_init is not positive.
        """
        if outlier_threshold_init <= 0:
            raise ValueError("outlier_threshold_init must be positive.")
        self.__outlier_threshold: float = outlier_threshold_init

    @property
    def outlier_threshold(self) -> float:
        """float: The modified Z-score threshold used for outlier detection."""
        return self.__outlier_threshold

    def build_consensus(self, graded_responses: List[GradedResponse]) -> EnsembleResult:
        """Builds a consensus EnsembleResult from multiple judges' GradedResponses.

        Args:
            graded_responses: The GradedResponse instances produced by each
                judge for the same student answer. Must be non-empty.

        Returns:
            EnsembleResult: The aggregated consensus, including the mean
            score across inlier judges, the standard deviation across all
            judges, and the identifiers of any outlier judges.

        Raises:
            ValueError: If graded_responses is empty.
        """
        if not graded_responses:
            raise ValueError(
                "Cannot build consensus from an empty list of graded responses."
            )

        scores = [response.final_score for response in graded_responses]
        # Population standard deviation is used since these are the entire
        # set of judges consulted for this answer, not a sample of a larger
        # population. A single judge has zero deviation by definition. This
        # value is reported for transparency but, deliberately, is NOT used
        # to decide outliers below: the mean and this std_dev are both
        # highly sensitive to the very outliers we are trying to detect,
        # which can make an outlier drag a tight majority group over the
        # threshold as well.
        std_dev = statistics.pstdev(scores) if len(scores) > 1 else 0.0

        # Outlier detection instead uses the median and Median Absolute
        # Deviation (MAD), which are robust to the presence of the outliers
        # themselves (a modified Z-score, per Iglewicz & Hoaglin).
        median_score = statistics.median(scores)
        absolute_deviations = [abs(score - median_score) for score in scores]
        mad = statistics.median(absolute_deviations)

        outlier_judge_ids: List[str] = []
        inlier_scores: List[float] = []
        for response, deviation in zip(graded_responses, absolute_deviations):
            if mad > 0:
                modified_z_score = 0.6745 * deviation / mad
                is_outlier = modified_z_score > self.__outlier_threshold
            else:
                # Every score but this one agrees exactly on the median; any
                # nonzero deviation from that unanimous value is an outlier.
                is_outlier = deviation > 0
            if is_outlier:
                outlier_judge_ids.append(response.judge_id)
            else:
                inlier_scores.append(response.final_score)

        # Recomputing the consensus from inliers only ensures an outlier
        # judge cannot skew the final reported score.
        consensus_score = statistics.mean(inlier_scores) if inlier_scores else median_score

        first_response = graded_responses[0]
        return EnsembleResult(
            test_case_id_init=first_response.test_case_id,
            answer_id_init=first_response.answer_id,
            graded_responses_init=graded_responses,
            consensus_score_init=consensus_score,
            score_std_dev_init=std_dev,
            outlier_judge_ids_init=outlier_judge_ids,
        )


class EvaluationOrchestrator:
    """Coordinates the full evaluation pipeline for a batch of tasks.

    The orchestrated flow is: QuestionGenerator produces a TestCase, the
    Audit Gate confirms it is solvable, StudentResponder answers it under
    Context Isolation, every configured JudgeEngine grades the answer, and
    EnsembleManager reduces those grades to a single consensus result.

    Attributes:
        question_generator (QuestionGenerator): The Teacher component.
        student_responder (StudentResponder): The Student component.
        judges (List[JudgeEngine]): The Judge components consulted for
            every answer.
        ensemble_manager (EnsembleManager): The consensus/outlier aggregator.
    """

    def __init__(
        self,
        question_generator_init: QuestionGenerator,
        student_responder_init: StudentResponder,
        judges_init: List[JudgeEngine],
        ensemble_manager_init: EnsembleManager,
    ) -> None:
        """Initializes the EvaluationOrchestrator.

        Args:
            question_generator_init: The Teacher component used to generate tasks.
            student_responder_init: The Student component used to answer tasks.
            judges_init: The Judge components consulted for every answer.
                Must contain at least one judge.
            ensemble_manager_init: The manager used to reduce multiple
                judges' scores to a single consensus.

        Raises:
            ValueError: If judges_init is empty.
        """
        if not judges_init:
            raise ValueError("EvaluationOrchestrator requires at least one JudgeEngine.")
        self.__question_generator: QuestionGenerator = question_generator_init
        self.__student_responder: StudentResponder = student_responder_init
        self.__judges: List[JudgeEngine] = list(judges_init)
        self.__ensemble_manager: EnsembleManager = ensemble_manager_init

    @property
    def question_generator(self) -> QuestionGenerator:
        """QuestionGenerator: The Teacher component."""
        return self.__question_generator

    @property
    def student_responder(self) -> StudentResponder:
        """StudentResponder: The Student component."""
        return self.__student_responder

    @property
    def judges(self) -> List[JudgeEngine]:
        """List[JudgeEngine]: A defensive copy of the configured judges."""
        return list(self.__judges)

    @property
    def ensemble_manager(self) -> EnsembleManager:
        """EnsembleManager: The consensus/outlier aggregator."""
        return self.__ensemble_manager

    def run_single_evaluation(self, topic: str, difficulty: str) -> EnsembleResult:
        """Runs the full pipeline for a single topic/difficulty pair.

        Args:
            topic: The high-level topic to generate a task for.
            difficulty: A difficulty label for the generated task.

        Returns:
            EnsembleResult: The ensemble consensus across all configured judges.

        Raises:
            RuntimeError: If the Audit Gate rejects the generated TestCase
                as unsolvable.
        """
        test_case = self.__question_generator.generate_test_case(topic, difficulty)
        self.__run_audit_gate(test_case)

        student_answer = self.__student_responder.answer(test_case)

        graded_responses = [
            judge.grade(test_case, student_answer) for judge in self.__judges
        ]
        return self.__ensemble_manager.build_consensus(graded_responses)

    def run_batch_evaluation(
        self, topic: str, difficulty: str, count: int
    ) -> List[EnsembleResult]:
        """Runs the full pipeline repeatedly for the same topic and difficulty.

        Args:
            topic: The high-level topic to generate tasks for.
            difficulty: A difficulty label applied to every generated task.
            count: The number of independent evaluations to run.

        Returns:
            List[EnsembleResult]: One EnsembleResult per evaluation run.

        Raises:
            ValueError: If count is not a positive integer.
        """
        if count <= 0:
            raise ValueError("count must be a positive integer.")
        return [self.run_single_evaluation(topic, difficulty) for _ in range(count)]

    def __run_audit_gate(self, test_case: TestCase) -> None:
        """Confirms a generated TestCase is solvable before releasing it to the Student.

        A task is considered solvable, for the purposes of this gate, when
        it carries a non-empty golden reference solution and at least one
        rubric item to grade against. This is a deterministic structural
        check rather than a re-invocation of an LLM, keeping the gate fast
        and reproducible.

        Args:
            test_case: The TestCase to audit.

        Raises:
            RuntimeError: If the TestCase fails the solvability check.
        """
        if not test_case.golden_reference.strip():
            raise RuntimeError(
                f"Audit Gate rejected TestCase {test_case.test_case_id}: "
                "empty golden_reference."
            )
        if not test_case.rubric:
            raise RuntimeError(
                f"Audit Gate rejected TestCase {test_case.test_case_id}: empty rubric."
            )
        test_case.mark_audited()
