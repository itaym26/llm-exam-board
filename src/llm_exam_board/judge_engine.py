"""
Judge-side component for the V14 LLM Evaluation Pipeline.

This module contains JudgeEngine, which grades a StudentAnswer against a
TestCase's dynamic rubric using an LLM, then applies the Truth Gate: a
static logic filter that overrides the LLM's own score whenever critical
logic (as defined by the rubric's validation patterns) is provably absent
from the student's answer.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .interfaces import DeductionItem, GradedResponse, RubricItem, StudentAnswer, TestCase
from .resilience import ResilienceUtils

# A generic LLM client contract: a callable that sends a prompt string and
# returns the model's raw text response. Kept local to this module so that
# judge_engine.py remains self-contained per the Clean Separation guideline.
LLMClient = Callable[[str], str]


class JudgeEngine:
    """The Judge component: grades a StudentAnswer and enforces the Truth Gate.

    Grading proceeds in two stages. First, an LLM judge is prompted to
    return itemized point deductions against the TestCase's dynamic rubric.
    Second, the Truth Gate statically scans the student's raw answer text
    against every critical RubricItem's validation_pattern; if any required
    pattern is missing, the entire score is zeroed out regardless of what
    the LLM claimed, preventing hallucinated positive feedback from masking
    a fundamentally broken answer.

    Attributes:
        llm_client (LLMClient): The callable used to send prompts to the Judge LLM.
        judge_id (str): The identifier of this judge instance/model.
    """

    def __init__(self, llm_client_init: LLMClient, judge_id_init: str) -> None:
        """Initializes the JudgeEngine.

        Args:
            llm_client_init: A callable that sends a prompt string to an LLM
                and returns its raw text response.
            judge_id_init: A unique identifier for this judge (e.g., the
                underlying model name), used to attribute scores in an
                ensemble and to identify outliers.

        Raises:
            ValueError: If judge_id_init is empty.
        """
        if not judge_id_init:
            raise ValueError("judge_id_init must not be empty.")
        self.__llm_client: LLMClient = llm_client_init
        self.__judge_id: str = judge_id_init

    @property
    def llm_client(self) -> LLMClient:
        """LLMClient: The callable used to send prompts to the Judge LLM."""
        return self.__llm_client

    @property
    def judge_id(self) -> str:
        """str: The identifier of this judge instance/model."""
        return self.__judge_id

    def grade(self, test_case: TestCase, student_answer: StudentAnswer) -> GradedResponse:
        """Grades a single StudentAnswer against a TestCase's dynamic rubric.

        Args:
            test_case: The TestCase whose rubric defines the grading criteria.
            student_answer: The StudentAnswer to be graded.

        Returns:
            GradedResponse: The graded result, including the LLM's raw score,
            the Truth-Gate-enforced final score, itemized deductions, and
            whether the Truth Gate overrode the LLM's assessment.
        """
        rubric = test_case.rubric

        prompt = self.__build_grading_prompt(test_case, student_answer)
        raw_response = self.__llm_client(prompt)
        parsed = ResilienceUtils.clean_json_response(raw_response)

        if parsed is None:
            # An unparsable judge response carries no usable information;
            # it is treated as a zero-deduction, zero-confidence response
            # rather than crashing the grading pipeline.
            llm_deductions: List[DeductionItem] = []
            feedback_summary = "Judge LLM response could not be parsed."
        else:
            llm_deductions = self.__parse_deductions(parsed.get("deductions", []))
            feedback_summary = str(parsed.get("feedback_summary", ""))

        raw_llm_score = self.__compute_score(rubric, llm_deductions)

        # Truth Gate: independently, statically verify every critical rubric
        # item regardless of what the LLM's deductions claimed.
        truth_gate_deductions, truth_gate_triggered = self.__apply_truth_gate(
            rubric, student_answer.answer_text
        )

        if truth_gate_triggered:
            # The Truth Gate strictly overrides the LLM: the score is zeroed
            # out entirely and the override deductions are appended so the
            # cause remains fully traceable.
            final_deductions = llm_deductions + truth_gate_deductions
            final_score = 0.0
        else:
            final_deductions = llm_deductions
            final_score = raw_llm_score

        return GradedResponse(
            test_case_id_init=test_case.test_case_id,
            answer_id_init=student_answer.answer_id,
            judge_id_init=self.__judge_id,
            raw_llm_score_init=raw_llm_score,
            final_score_init=final_score,
            deductions_init=final_deductions,
            truth_gate_triggered_init=truth_gate_triggered,
            feedback_summary_init=feedback_summary,
        )

    def grade_batch(
        self, test_case: TestCase, student_answers: List[StudentAnswer]
    ) -> List[GradedResponse]:
        """Grades multiple StudentAnswer instances against the same TestCase.

        Args:
            test_case: The TestCase whose rubric defines the grading criteria.
            student_answers: The StudentAnswer instances to grade.

        Returns:
            List[GradedResponse]: One GradedResponse per input StudentAnswer,
            in the same order.
        """
        return [self.grade(test_case, student_answer) for student_answer in student_answers]

    def __build_grading_prompt(self, test_case: TestCase, student_answer: StudentAnswer) -> str:
        """Builds the prompt instructing the Judge LLM to return itemized deductions.

        Args:
            test_case: The TestCase providing the rubric and golden reference.
            student_answer: The StudentAnswer being graded.

        Returns:
            str: The complete prompt text to send to the Judge LLM.
        """
        rubric_description = "\n".join(
            f'- id="{item.item_id}" ({item.max_points} pts): {item.description}'
            for item in test_case.rubric
        )
        return (
            "You are grading a student's answer against a fixed rubric.\n"
            f"Task:\n{test_case.prompt}\n\n"
            f"Golden reference solution:\n{test_case.golden_reference}\n\n"
            f"Rubric items:\n{rubric_description}\n\n"
            f"Student answer:\n{student_answer.answer_text}\n\n"
            "Respond with a single JSON object containing exactly these keys:\n"
            '  "deductions": an array of objects, each with keys '
            '"rubric_item_id", "points_deducted", and "justification", '
            "one entry per rubric item where points were lost (omit items "
            "with no deduction),\n"
            '  "feedback_summary": a short free-text summary for human review.\n'
            "Return only the JSON object, with no additional commentary."
        )

    def __parse_deductions(self, raw_deductions: List[Dict[str, Any]]) -> List[DeductionItem]:
        """Converts raw deduction dictionaries from the LLM response into DeductionItem instances.

        Args:
            raw_deductions: The raw list of deduction dictionaries as parsed
                from the Judge LLM's JSON response.

        Returns:
            List[DeductionItem]: The successfully constructed deductions.
            Malformed entries are silently skipped rather than aborting
            grading of the entire answer, consistent with the
            Micro-Resilience principle.
        """
        deductions: List[DeductionItem] = []
        for raw_item in raw_deductions:
            try:
                deductions.append(
                    DeductionItem(
                        rubric_item_id_init=str(raw_item["rubric_item_id"]),
                        points_deducted_init=float(raw_item["points_deducted"]),
                        justification_init=str(raw_item["justification"]),
                        is_truth_gate_override_init=False,
                    )
                )
            except (KeyError, TypeError, ValueError):
                # Skip malformed deduction entries rather than crashing the
                # entire grading pipeline over one bad item.
                continue
        return deductions

    def __compute_score(
        self, rubric: List[RubricItem], deductions: List[DeductionItem]
    ) -> float:
        """Computes a score by subtracting itemized deductions from the total possible points.

        Args:
            rubric: The full list of rubric items for this test case.
            deductions: The itemized deductions to subtract.

        Returns:
            float: The resulting score, floored at zero.
        """
        total_possible = sum(item.max_points for item in rubric)
        total_deducted = sum(deduction.points_deducted for deduction in deductions)
        return max(0.0, total_possible - total_deducted)

    @staticmethod
    def check_pattern(pattern: str, text: str) -> Optional[bool]:
        """Safely checks whether a regex pattern matches within a text, without ever raising.

        `validation_pattern` values are authored by an LLM (the Teacher),
        not a human, so they cannot be trusted to always be syntactically
        valid regular expressions. This method is the single place that
        risk is contained, so neither the Truth Gate nor the Audit Gate
        (which reuses this same check against the golden reference) can be
        crashed by a malformed pattern.

        Args:
            pattern: The regex pattern to search for.
            text: The text to search within.

        Returns:
            Optional[bool]: True if the pattern matches somewhere in the
            text, False if the pattern is valid but does not match, or
            None if `pattern` is not itself a valid regular expression --
            in which case the check is inconclusive rather than a
            confirmed absence of the underlying logic.
        """
        try:
            # re.DOTALL is essential here: these patterns check for a
            # construct's presence anywhere in a multi-line code snippet,
            # and a Teacher-authored pattern spanning a line break (e.g.
            # matching a method signature followed by part of its body)
            # would otherwise silently fail to match even a fully correct
            # answer, since "." does not match "\n" by default.
            return re.search(pattern, text, re.DOTALL) is not None
        except re.error:
            return None

    def __apply_truth_gate(
        self, rubric: List[RubricItem], answer_text: str
    ) -> Tuple[List[DeductionItem], bool]:
        """Statically scans the student's answer for every critical rubric item's pattern.

        This is the Truth Gate: it operates entirely independently of the
        LLM's own assessment, using deterministic regex matching against the
        raw answer text. Any critical rubric item whose validation_pattern
        is confirmed absent (not merely unverifiable) triggers an override.

        Args:
            rubric: The full list of rubric items for this test case.
            answer_text: The raw text/code submitted by the student.

        Returns:
            Tuple[List[DeductionItem], bool]: A list of override deduction
            items explaining each missing critical pattern (empty if none
            were missing), and a boolean indicating whether the Truth Gate
            was triggered at all.
        """
        override_deductions: List[DeductionItem] = []

        for item in rubric:
            if not item.is_critical or not item.validation_pattern:
                continue
            match_result = self.check_pattern(item.validation_pattern, answer_text)
            # match_result is False only when the pattern is valid and
            # confirmed absent -- a real missing-logic finding. When it is
            # None, the pattern itself is malformed and the check is
            # inconclusive, so the student must not be penalized for a
            # rubric-authoring defect that isn't theirs.
            if match_result is False:
                override_deductions.append(
                    DeductionItem(
                        rubric_item_id_init=item.item_id,
                        points_deducted_init=item.max_points,
                        justification_init=(
                            f"Truth Gate: critical pattern for '{item.description}' "
                            "was not found in the student's answer via static scan."
                        ),
                        is_truth_gate_override_init=True,
                    )
                )

        return override_deductions, bool(override_deductions)
