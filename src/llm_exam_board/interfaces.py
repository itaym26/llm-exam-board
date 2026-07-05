"""
Foundational data models for the V14 LLM Evaluation Pipeline.

This module defines the strictly encapsulated data contracts shared across
every other module in the system (generators, responders, judge_engine,
orchestrator). No other module may define its own data-carrying classes;
they must all consume the models defined here.
"""

import uuid
from dataclasses import InitVar, dataclass
from typing import List, Optional


@dataclass
class RubricItem:
    """Represents a single, gradable criterion within a dynamic rubric.

    A RubricItem is the atomic unit of grading. When `is_critical` is set,
    the JudgeEngine's Truth Gate will statically verify `validation_pattern`
    against the student's raw answer text before trusting any LLM-generated
    score for this criterion.

    Attributes:
        item_id (str): Unique identifier for this rubric item.
        description (str): Human-readable explanation of what is being graded.
        max_points (float): Maximum number of points this item is worth.
        is_critical (bool): Marks this item as subject to Truth Gate enforcement.
        validation_pattern (Optional[str]): A regex pattern used by the Truth
            Gate to statically confirm the presence of critical logic
            (e.g., a mutex lock acquisition) in the student's answer.
    """

    description_init: InitVar[str]
    max_points_init: InitVar[float]
    is_critical_init: InitVar[bool] = False
    validation_pattern_init: InitVar[Optional[str]] = None
    item_id_init: InitVar[Optional[str]] = None

    def __post_init__(
        self,
        description_init: str,
        max_points_init: float,
        is_critical_init: bool,
        validation_pattern_init: Optional[str],
        item_id_init: Optional[str],
    ) -> None:
        """Validates constructor arguments and populates private attributes.

        Args:
            description_init: Raw description passed to the constructor.
            max_points_init: Raw max point value passed to the constructor.
            is_critical_init: Raw critical flag passed to the constructor.
            validation_pattern_init: Raw regex pattern passed to the constructor.
            item_id_init: Raw identifier passed to the constructor, or None
                to auto-generate one.

        Raises:
            ValueError: If description is empty or max_points is negative.
        """
        if not description_init:
            raise ValueError("RubricItem description must not be empty.")
        if max_points_init < 0:
            raise ValueError("RubricItem max_points must not be negative.")
        # A critical item without a validation pattern would make the Truth
        # Gate unable to perform its static scan, silently defeating it.
        if is_critical_init and not validation_pattern_init:
            raise ValueError(
                "Critical rubric items must define a validation_pattern "
                "for the Truth Gate to statically enforce."
            )

        self.__description: str = description_init
        self.__max_points: float = max_points_init
        self.__is_critical: bool = is_critical_init
        self.__validation_pattern: Optional[str] = validation_pattern_init
        self.__item_id: str = item_id_init or str(uuid.uuid4())

    @property
    def item_id(self) -> str:
        """str: Unique identifier for this rubric item."""
        return self.__item_id

    @property
    def description(self) -> str:
        """str: Human-readable explanation of what is being graded."""
        return self.__description

    @property
    def max_points(self) -> float:
        """float: Maximum number of points this item is worth."""
        return self.__max_points

    @property
    def is_critical(self) -> bool:
        """bool: Whether this item is enforced by the Truth Gate."""
        return self.__is_critical

    @property
    def validation_pattern(self) -> Optional[str]:
        """Optional[str]: Regex pattern used for static Truth Gate validation."""
        return self.__validation_pattern

    def __repr__(self) -> str:
        """Returns a concise, debug-friendly representation."""
        return (
            f"RubricItem(item_id={self.__item_id!r}, "
            f"max_points={self.__max_points!r}, is_critical={self.__is_critical!r})"
        )


@dataclass
class DeductionItem:
    """Represents a single, itemized point deduction applied to a graded answer.

    Deduction items are the only mechanism through which points may be
    removed from a student's score. This enforces the Itemized Deductions
    principle: the Judge must justify every point lost against a specific
    rubric item rather than returning unstructured free-text feedback.

    Attributes:
        rubric_item_id (str): The RubricItem.item_id this deduction applies to.
        points_deducted (float): Number of points removed for this criterion.
        justification (str): Explanation of why points were deducted.
        is_truth_gate_override (bool): True if this deduction was forcibly
            applied by the Truth Gate rather than proposed by the LLM judge.
    """

    rubric_item_id_init: InitVar[str]
    points_deducted_init: InitVar[float]
    justification_init: InitVar[str]
    is_truth_gate_override_init: InitVar[bool] = False

    def __post_init__(
        self,
        rubric_item_id_init: str,
        points_deducted_init: float,
        justification_init: str,
        is_truth_gate_override_init: bool,
    ) -> None:
        """Validates constructor arguments and populates private attributes.

        Args:
            rubric_item_id_init: Identifier of the rubric item being deducted from.
            points_deducted_init: Raw point deduction amount.
            justification_init: Raw human-readable justification text.
            is_truth_gate_override_init: Whether the Truth Gate forced this deduction.

        Raises:
            ValueError: If points_deducted is negative or justification is empty.
        """
        if points_deducted_init < 0:
            raise ValueError("points_deducted must not be negative.")
        if not justification_init:
            raise ValueError("DeductionItem justification must not be empty.")

        self.__rubric_item_id: str = rubric_item_id_init
        self.__points_deducted: float = points_deducted_init
        self.__justification: str = justification_init
        self.__is_truth_gate_override: bool = is_truth_gate_override_init

    @property
    def rubric_item_id(self) -> str:
        """str: The RubricItem.item_id this deduction applies to."""
        return self.__rubric_item_id

    @property
    def points_deducted(self) -> float:
        """float: Number of points removed for this criterion."""
        return self.__points_deducted

    @property
    def justification(self) -> str:
        """str: Explanation of why points were deducted."""
        return self.__justification

    @property
    def is_truth_gate_override(self) -> bool:
        """bool: Whether this deduction was forcibly applied by the Truth Gate."""
        return self.__is_truth_gate_override

    def __repr__(self) -> str:
        """Returns a concise, debug-friendly representation."""
        return (
            f"DeductionItem(rubric_item_id={self.__rubric_item_id!r}, "
            f"points_deducted={self.__points_deducted!r}, "
            f"is_truth_gate_override={self.__is_truth_gate_override!r})"
        )


@dataclass
class TestCase:
    """Represents a single evaluation task generated by the Teacher.

    A TestCase bundles the question prompt, the Teacher's own golden
    reference solution, and the dynamic rubric that the Judge will use
    to score Student answers. `is_audited` records whether the Audit Gate
    has already confirmed this task is solvable.

    Attributes:
        test_case_id (str): Unique identifier for this test case.
        topic (str): The subject area this task belongs to.
        difficulty (str): A difficulty label (e.g., "easy", "medium", "hard").
        prompt (str): The question text presented to the Student.
        golden_reference (str): The Teacher's own reference solution, used
            by the Audit Gate to confirm solvability.
        rubric (List[RubricItem]): The dynamic rubric used to grade answers.
        is_audited (bool): Whether the Audit Gate has verified solvability.
    """

    topic_init: InitVar[str]
    difficulty_init: InitVar[str]
    prompt_init: InitVar[str]
    golden_reference_init: InitVar[str]
    rubric_init: InitVar[Optional[List[RubricItem]]] = None
    is_audited_init: InitVar[bool] = False
    test_case_id_init: InitVar[Optional[str]] = None

    def __post_init__(
        self,
        topic_init: str,
        difficulty_init: str,
        prompt_init: str,
        golden_reference_init: str,
        rubric_init: Optional[List[RubricItem]],
        is_audited_init: bool,
        test_case_id_init: Optional[str],
    ) -> None:
        """Validates constructor arguments and populates private attributes.

        Args:
            topic_init: Subject area label for this task.
            difficulty_init: Difficulty label for this task.
            prompt_init: The question text presented to the Student.
            golden_reference_init: The Teacher's own reference solution.
            rubric_init: List of RubricItem objects, or None for an empty rubric.
            is_audited_init: Whether the Audit Gate has already run.
            test_case_id_init: Explicit identifier, or None to auto-generate one.

        Raises:
            ValueError: If prompt or golden_reference is empty.
        """
        if not prompt_init:
            raise ValueError("TestCase prompt must not be empty.")
        if not golden_reference_init:
            raise ValueError("TestCase golden_reference must not be empty.")

        self.__topic: str = topic_init
        self.__difficulty: str = difficulty_init
        self.__prompt: str = prompt_init
        self.__golden_reference: str = golden_reference_init
        # Defensive copy prevents external mutation of the internal rubric list.
        self.__rubric: List[RubricItem] = list(rubric_init) if rubric_init else []
        self.__is_audited: bool = is_audited_init
        self.__test_case_id: str = test_case_id_init or str(uuid.uuid4())

    @property
    def test_case_id(self) -> str:
        """str: Unique identifier for this test case."""
        return self.__test_case_id

    @property
    def topic(self) -> str:
        """str: The subject area this task belongs to."""
        return self.__topic

    @property
    def difficulty(self) -> str:
        """str: Difficulty label for this task."""
        return self.__difficulty

    @property
    def prompt(self) -> str:
        """str: The question text presented to the Student."""
        return self.__prompt

    @property
    def golden_reference(self) -> str:
        """str: The Teacher's own reference solution."""
        return self.__golden_reference

    @property
    def rubric(self) -> List[RubricItem]:
        """List[RubricItem]: A defensive copy of the dynamic rubric."""
        # Returning a copy preserves encapsulation: callers cannot mutate
        # the internal list through the returned reference.
        return list(self.__rubric)

    @property
    def is_audited(self) -> bool:
        """bool: Whether the Audit Gate has verified this task is solvable."""
        return self.__is_audited

    def mark_audited(self) -> None:
        """Marks this test case as having passed the Audit Gate.

        This is the only permitted mutation on a TestCase instance, modeling
        a one-way state transition once the Orchestrator's Audit Gate
        confirms the golden reference actually solves the task.
        """
        self.__is_audited = True

    def __repr__(self) -> str:
        """Returns a concise, debug-friendly representation."""
        return (
            f"TestCase(test_case_id={self.__test_case_id!r}, "
            f"topic={self.__topic!r}, is_audited={self.__is_audited!r})"
        )


@dataclass
class StudentAnswer:
    """Represents a Student model's answer to a single TestCase.

    Attributes:
        answer_id (str): Unique identifier for this answer.
        test_case_id (str): The TestCase.test_case_id this answer responds to.
        answer_text (str): The raw text/code submitted by the Student.
        is_context_isolated (bool): Confirms the Student produced this answer
            under Context Isolation (no prior conversation memory).
    """

    test_case_id_init: InitVar[str]
    answer_text_init: InitVar[str]
    is_context_isolated_init: InitVar[bool] = True
    answer_id_init: InitVar[Optional[str]] = None

    def __post_init__(
        self,
        test_case_id_init: str,
        answer_text_init: str,
        is_context_isolated_init: bool,
        answer_id_init: Optional[str],
    ) -> None:
        """Validates constructor arguments and populates private attributes.

        Args:
            test_case_id_init: Identifier of the TestCase being answered.
            answer_text_init: The raw text/code submitted by the Student.
            is_context_isolated_init: Whether Context Isolation was enforced.
            answer_id_init: Explicit identifier, or None to auto-generate one.

        Raises:
            ValueError: If test_case_id is empty.
        """
        if not test_case_id_init:
            raise ValueError("StudentAnswer test_case_id must not be empty.")

        self.__test_case_id: str = test_case_id_init
        # An empty answer is valid (a non-answer/timeout) and must still be
        # gradable, so no emptiness check is applied to answer_text.
        self.__answer_text: str = answer_text_init
        self.__is_context_isolated: bool = is_context_isolated_init
        self.__answer_id: str = answer_id_init or str(uuid.uuid4())

    @property
    def answer_id(self) -> str:
        """str: Unique identifier for this answer."""
        return self.__answer_id

    @property
    def test_case_id(self) -> str:
        """str: The TestCase.test_case_id this answer responds to."""
        return self.__test_case_id

    @property
    def answer_text(self) -> str:
        """str: The raw text/code submitted by the Student."""
        return self.__answer_text

    @property
    def is_context_isolated(self) -> bool:
        """bool: Whether Context Isolation was enforced when generating this answer."""
        return self.__is_context_isolated

    def __repr__(self) -> str:
        """Returns a concise, debug-friendly representation."""
        return (
            f"StudentAnswer(answer_id={self.__answer_id!r}, "
            f"test_case_id={self.__test_case_id!r})"
        )


@dataclass
class GradedResponse:
    """Represents a single Judge's evaluation of one StudentAnswer.

    The `final_score` is always the authoritative score: it is derived from
    `raw_llm_score` but is subject to being zeroed out or overridden by the
    JudgeEngine's Truth Gate when critical logic is statically found to be
    missing, regardless of what the underlying LLM claimed in its feedback.

    Attributes:
        response_id (str): Unique identifier for this graded response.
        test_case_id (str): The TestCase.test_case_id being graded.
        answer_id (str): The StudentAnswer.answer_id being graded.
        judge_id (str): Identifier of the judge model/instance that produced this.
        raw_llm_score (float): The score as originally proposed by the LLM judge.
        final_score (float): The authoritative score after Truth Gate enforcement.
        deductions (List[DeductionItem]): Itemized point deductions.
        truth_gate_triggered (bool): Whether the Truth Gate overrode the LLM score.
        feedback_summary (str): Free-text summary, informational only and never
            authoritative over the itemized deductions or final_score.
    """

    test_case_id_init: InitVar[str]
    answer_id_init: InitVar[str]
    judge_id_init: InitVar[str]
    raw_llm_score_init: InitVar[float]
    final_score_init: InitVar[float]
    deductions_init: InitVar[Optional[List[DeductionItem]]] = None
    truth_gate_triggered_init: InitVar[bool] = False
    feedback_summary_init: InitVar[str] = ""
    response_id_init: InitVar[Optional[str]] = None

    def __post_init__(
        self,
        test_case_id_init: str,
        answer_id_init: str,
        judge_id_init: str,
        raw_llm_score_init: float,
        final_score_init: float,
        deductions_init: Optional[List[DeductionItem]],
        truth_gate_triggered_init: bool,
        feedback_summary_init: str,
        response_id_init: Optional[str],
    ) -> None:
        """Validates constructor arguments and populates private attributes.

        Args:
            test_case_id_init: Identifier of the TestCase being graded.
            answer_id_init: Identifier of the StudentAnswer being graded.
            judge_id_init: Identifier of the judge model/instance.
            raw_llm_score_init: The score as originally proposed by the LLM judge.
            final_score_init: The authoritative score after Truth Gate enforcement.
            deductions_init: List of DeductionItem objects, or None for an empty list.
            truth_gate_triggered_init: Whether the Truth Gate overrode the LLM score.
            feedback_summary_init: Free-text, non-authoritative feedback summary.
            response_id_init: Explicit identifier, or None to auto-generate one.

        Raises:
            ValueError: If either score is negative, or if a Truth Gate override
                is claimed without any corresponding override deduction present.
        """
        if raw_llm_score_init < 0 or final_score_init < 0:
            raise ValueError("Scores must not be negative.")
        resolved_deductions = list(deductions_init) if deductions_init else []
        # Guards against silent inconsistency: a Truth Gate override must always
        # be traceable to at least one deduction flagged as its cause.
        if truth_gate_triggered_init and not any(
            d.is_truth_gate_override for d in resolved_deductions
        ):
            raise ValueError(
                "truth_gate_triggered is True but no deduction is marked "
                "as is_truth_gate_override; the override would be untraceable."
            )

        self.__test_case_id: str = test_case_id_init
        self.__answer_id: str = answer_id_init
        self.__judge_id: str = judge_id_init
        self.__raw_llm_score: float = raw_llm_score_init
        self.__final_score: float = final_score_init
        self.__deductions: List[DeductionItem] = resolved_deductions
        self.__truth_gate_triggered: bool = truth_gate_triggered_init
        self.__feedback_summary: str = feedback_summary_init
        self.__response_id: str = response_id_init or str(uuid.uuid4())

    @property
    def response_id(self) -> str:
        """str: Unique identifier for this graded response."""
        return self.__response_id

    @property
    def test_case_id(self) -> str:
        """str: The TestCase.test_case_id being graded."""
        return self.__test_case_id

    @property
    def answer_id(self) -> str:
        """str: The StudentAnswer.answer_id being graded."""
        return self.__answer_id

    @property
    def judge_id(self) -> str:
        """str: Identifier of the judge model/instance that produced this response."""
        return self.__judge_id

    @property
    def raw_llm_score(self) -> float:
        """float: The score as originally proposed by the LLM judge."""
        return self.__raw_llm_score

    @property
    def final_score(self) -> float:
        """float: The authoritative score after Truth Gate enforcement."""
        return self.__final_score

    @property
    def deductions(self) -> List[DeductionItem]:
        """List[DeductionItem]: A defensive copy of the itemized deductions."""
        return list(self.__deductions)

    @property
    def truth_gate_triggered(self) -> bool:
        """bool: Whether the Truth Gate overrode the LLM's original score."""
        return self.__truth_gate_triggered

    @property
    def feedback_summary(self) -> str:
        """str: Free-text feedback summary; never authoritative over final_score."""
        return self.__feedback_summary

    def __repr__(self) -> str:
        """Returns a concise, debug-friendly representation."""
        return (
            f"GradedResponse(response_id={self.__response_id!r}, "
            f"judge_id={self.__judge_id!r}, final_score={self.__final_score!r}, "
            f"truth_gate_triggered={self.__truth_gate_triggered!r})"
        )


@dataclass
class EnsembleResult:
    """Represents the aggregated consensus of multiple judges grading one answer.

    Produced by the EnsembleManager, this model captures the mean score across
    all non-outlier GradedResponse instances for a given StudentAnswer, along
    with which judges (if any) were isolated as statistical outliers.

    Attributes:
        test_case_id (str): The TestCase.test_case_id being graded.
        answer_id (str): The StudentAnswer.answer_id being graded.
        graded_responses (List[GradedResponse]): All individual judge responses.
        consensus_score (float): The mean final_score across non-outlier judges.
        outlier_judge_ids (List[str]): Identifiers of judges excluded as outliers.
        score_std_dev (float): Standard deviation of final_score across all judges.
    """

    test_case_id_init: InitVar[str]
    answer_id_init: InitVar[str]
    graded_responses_init: InitVar[List[GradedResponse]]
    consensus_score_init: InitVar[float]
    score_std_dev_init: InitVar[float]
    outlier_judge_ids_init: InitVar[Optional[List[str]]] = None

    def __post_init__(
        self,
        test_case_id_init: str,
        answer_id_init: str,
        graded_responses_init: List[GradedResponse],
        consensus_score_init: float,
        score_std_dev_init: float,
        outlier_judge_ids_init: Optional[List[str]],
    ) -> None:
        """Validates constructor arguments and populates private attributes.

        Args:
            test_case_id_init: Identifier of the TestCase being graded.
            answer_id_init: Identifier of the StudentAnswer being graded.
            graded_responses_init: All individual GradedResponse instances collected.
            consensus_score_init: The mean final_score across non-outlier judges.
            score_std_dev_init: Standard deviation of final_score across all judges.
            outlier_judge_ids_init: Identifiers of judges excluded as outliers,
                or None if no outliers were detected.

        Raises:
            ValueError: If graded_responses_init is empty, since consensus
                cannot be computed over zero judges.
        """
        if not graded_responses_init:
            raise ValueError(
                "EnsembleResult requires at least one GradedResponse to "
                "compute a consensus."
            )

        self.__test_case_id: str = test_case_id_init
        self.__answer_id: str = answer_id_init
        self.__graded_responses: List[GradedResponse] = list(graded_responses_init)
        self.__consensus_score: float = consensus_score_init
        self.__score_std_dev: float = score_std_dev_init
        self.__outlier_judge_ids: List[str] = (
            list(outlier_judge_ids_init) if outlier_judge_ids_init else []
        )

    @property
    def test_case_id(self) -> str:
        """str: The TestCase.test_case_id being graded."""
        return self.__test_case_id

    @property
    def answer_id(self) -> str:
        """str: The StudentAnswer.answer_id being graded."""
        return self.__answer_id

    @property
    def graded_responses(self) -> List[GradedResponse]:
        """List[GradedResponse]: A defensive copy of all individual judge responses."""
        return list(self.__graded_responses)

    @property
    def consensus_score(self) -> float:
        """float: The mean final_score across non-outlier judges."""
        return self.__consensus_score

    @property
    def score_std_dev(self) -> float:
        """float: Standard deviation of final_score across all judges."""
        return self.__score_std_dev

    @property
    def outlier_judge_ids(self) -> List[str]:
        """List[str]: A defensive copy of the identifiers of excluded outlier judges."""
        return list(self.__outlier_judge_ids)

    def __repr__(self) -> str:
        """Returns a concise, debug-friendly representation."""
        return (
            f"EnsembleResult(answer_id={self.__answer_id!r}, "
            f"consensus_score={self.__consensus_score!r}, "
            f"outlier_judge_ids={self.__outlier_judge_ids!r})"
        )
