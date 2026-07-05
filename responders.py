"""
Student-side component for the V14 LLM Evaluation Pipeline.

This module contains StudentResponder, which answers TestCase prompts using
an LLM while strictly enforcing Context Isolation: every answer is produced
from a brand-new prompt with no memory of previously answered tasks.
"""

from typing import Callable, List

from interfaces import StudentAnswer, TestCase

# A generic LLM client contract: a callable that sends a prompt string and
# returns the model's raw text response. Kept local to this module so that
# responders.py remains self-contained per the Clean Separation guideline.
LLMClient = Callable[[str], str]


class StudentResponder:
    """The Student component: answers TestCase prompts under Context Isolation.

    Context Isolation is enforced structurally rather than by convention:
    this class holds no conversational state between calls, and every
    invocation of `answer` builds a fresh, self-contained prompt containing
    only the current TestCase's text. This prevents information from one
    task (or its rubric/golden reference) from leaking into the answer of
    another.

    Attributes:
        llm_client (LLMClient): The callable used to send prompts to the Student LLM.
    """

    def __init__(self, llm_client_init: LLMClient) -> None:
        """Initializes the StudentResponder.

        Args:
            llm_client_init: A callable that sends a prompt string to an LLM
                and returns its raw text response. This callable must not
                itself retain conversation history across invocations,
                otherwise Context Isolation cannot be guaranteed end-to-end.
        """
        self.__llm_client: LLMClient = llm_client_init

    @property
    def llm_client(self) -> LLMClient:
        """LLMClient: The callable used to send prompts to the Student LLM."""
        return self.__llm_client

    def answer(self, test_case: TestCase) -> StudentAnswer:
        """Produces a StudentAnswer for the given TestCase under Context Isolation.

        Args:
            test_case: The TestCase to answer.

        Returns:
            StudentAnswer: The Student's answer, marked as having been
            produced under Context Isolation.
        """
        # A fresh, self-contained prompt is built for every call; no state
        # from prior calls is read or referenced here.
        isolated_prompt = self.__build_isolated_prompt(test_case)
        raw_answer = self.__llm_client(isolated_prompt)

        return StudentAnswer(
            test_case_id_init=test_case.test_case_id,
            answer_text_init=raw_answer,
            is_context_isolated_init=True,
        )

    def answer_batch(self, test_cases: List[TestCase]) -> List[StudentAnswer]:
        """Produces a StudentAnswer for each TestCase in the given batch.

        Each call to `answer` within this batch remains independently
        isolated; no context is carried over between test cases.

        Args:
            test_cases: The TestCase instances to answer.

        Returns:
            List[StudentAnswer]: One StudentAnswer per input TestCase, in
            the same order.
        """
        return [self.answer(test_case) for test_case in test_cases]

    def __build_isolated_prompt(self, test_case: TestCase) -> str:
        """Builds a self-contained prompt referencing only the current task.

        Args:
            test_case: The TestCase whose prompt text will be answered.

        Returns:
            str: A prompt containing exclusively the current task's
            question text, with explicit isolation instructions and no
            reference to any other task.
        """
        return (
            "You are answering a single, standalone task. "
            "You have no memory of any previous task or conversation; "
            "treat the following as the entirety of the available context.\n\n"
            f"Task:\n{test_case.prompt}"
        )
