"""
Teacher-side components for the V14 LLM Evaluation Pipeline.

This module contains ConfigManager, which manages topic breakdown
configuration, and QuestionGenerator, the Teacher component that prompts an
LLM to produce TestCase instances complete with a golden reference solution
and a dynamic, itemized rubric.
"""

import random
from typing import Any, Callable, Dict, List, Optional

from .interfaces import RubricItem, TestCase
from .resilience import ResilienceUtils

# A generic LLM client contract: a callable that sends a prompt string and
# returns the model's raw text response. Kept local to this module so that
# generators.py remains self-contained per the Clean Separation guideline.
LLMClient = Callable[[str], str]


class ConfigManager:
    """Manages the topic breakdown configuration used to diversify generated test cases.

    The ConfigManager holds a mapping from high-level topics (e.g.,
    "Multithreading") to a list of granular subtopics (e.g., "mutex locks",
    "deadlock avoidance", "thread pools"). QuestionGenerator consults this
    mapping to pick a specific focus area for each generated TestCase.

    Attributes:
        topic_breakdown (Dict[str, List[str]]): A defensive copy of the
            current topic-to-subtopics mapping.
    """

    def __init__(self, topic_breakdown_init: Optional[Dict[str, List[str]]] = None) -> None:
        """Initializes the ConfigManager with an optional initial topic breakdown.

        Args:
            topic_breakdown_init: Initial mapping of topic name to a list of
                subtopics. Defaults to an empty configuration if omitted.
        """
        # Defensive copy: both the outer dict and each inner list are copied
        # so that external mutation of the caller's structures cannot
        # silently corrupt this instance's private state.
        self.__topic_breakdown: Dict[str, List[str]] = (
            {topic: list(subtopics) for topic, subtopics in topic_breakdown_init.items()}
            if topic_breakdown_init
            else {}
        )

    @property
    def topic_breakdown(self) -> Dict[str, List[str]]:
        """Dict[str, List[str]]: A defensive copy of the topic-to-subtopics mapping."""
        return {topic: list(subtopics) for topic, subtopics in self.__topic_breakdown.items()}

    def get_subtopics(self, topic: str) -> List[str]:
        """Returns the configured subtopics for a given high-level topic.

        Args:
            topic: The high-level topic to look up.

        Returns:
            List[str]: The subtopics registered for `topic`, or an empty
            list if the topic has no registered breakdown.
        """
        return list(self.__topic_breakdown.get(topic, []))

    def list_topics(self) -> List[str]:
        """Returns all high-level topics currently registered.

        Returns:
            List[str]: The list of registered topic names.
        """
        return list(self.__topic_breakdown.keys())

    def register_topic(self, topic: str, subtopics: List[str]) -> None:
        """Registers or replaces the subtopic breakdown for a given topic.

        Args:
            topic: The high-level topic name.
            subtopics: The list of granular subtopics belonging to `topic`.

        Raises:
            ValueError: If `topic` is empty or `subtopics` is empty.
        """
        if not topic:
            raise ValueError("topic must not be empty.")
        if not subtopics:
            raise ValueError("subtopics must not be empty.")
        self.__topic_breakdown[topic] = list(subtopics)


class QuestionGenerator:
    """The Teacher component: generates TestCase instances via an LLM.

    QuestionGenerator prompts an LLM to produce a question, a golden
    reference solution, and a dynamic rubric, then resiliently parses the
    response using ResilienceUtils and assembles a validated TestCase.

    Attributes:
        llm_client (LLMClient): The callable used to send prompts to the Teacher LLM.
        config_manager (ConfigManager): The topic breakdown configuration in use.
    """

    def __init__(self, llm_client_init: LLMClient, config_manager_init: ConfigManager) -> None:
        """Initializes the QuestionGenerator.

        Args:
            llm_client_init: A callable that sends a prompt string to an LLM
                and returns its raw text response.
            config_manager_init: The ConfigManager providing topic breakdown data.
        """
        self.__llm_client: LLMClient = llm_client_init
        self.__config_manager: ConfigManager = config_manager_init

    @property
    def llm_client(self) -> LLMClient:
        """LLMClient: The callable used to send prompts to the Teacher LLM."""
        return self.__llm_client

    @property
    def config_manager(self) -> ConfigManager:
        """ConfigManager: The topic breakdown configuration in use."""
        return self.__config_manager

    def generate_test_case(self, topic: str, difficulty: str) -> TestCase:
        """Generates a single TestCase for the given topic and difficulty.

        Args:
            topic: The high-level topic for the generated task.
            difficulty: A difficulty label (e.g., "easy", "medium", "hard").

        Returns:
            TestCase: The fully assembled test case, including its golden
            reference solution and dynamic rubric.

        Raises:
            RuntimeError: If the Teacher LLM's response cannot be parsed
                into valid JSON, or is missing required keys, even after
                Micro-Resilience recovery.
        """
        subtopics = self.__config_manager.get_subtopics(topic)
        # Picking a random subtopic (when available) diversifies generated
        # tasks across repeated calls for the same high-level topic.
        focus_subtopic = random.choice(subtopics) if subtopics else topic

        prompt = self.__build_generation_prompt(topic, difficulty, focus_subtopic)
        raw_response = self.__llm_client(prompt)
        parsed = ResilienceUtils.clean_json_response(raw_response)

        if parsed is None:
            raise RuntimeError(
                f"QuestionGenerator: Teacher LLM returned unparsable output for topic '{topic}'."
            )

        try:
            question_text = str(parsed["question"])
            golden_reference = str(parsed["golden_reference"])
        except KeyError as exc:
            raise RuntimeError(
                f"QuestionGenerator: Teacher LLM response missing required key {exc}."
            ) from exc

        rubric_items = self.__parse_rubric(parsed.get("rubric", []))

        return TestCase(
            topic_init=topic,
            difficulty_init=difficulty,
            prompt_init=question_text,
            golden_reference_init=golden_reference,
            rubric_init=rubric_items,
        )

    def generate_batch(self, topic: str, difficulty: str, count: int) -> List[TestCase]:
        """Generates multiple TestCase instances for the same topic and difficulty.

        Args:
            topic: The high-level topic for the generated tasks.
            difficulty: A difficulty label applied to every generated task.
            count: The number of test cases to generate.

        Returns:
            List[TestCase]: The generated test cases.

        Raises:
            ValueError: If `count` is not a positive integer.
        """
        if count <= 0:
            raise ValueError("count must be a positive integer.")
        return [self.generate_test_case(topic, difficulty) for _ in range(count)]

    def __build_generation_prompt(self, topic: str, difficulty: str, focus_subtopic: str) -> str:
        """Builds the prompt instructing the Teacher LLM to produce structured JSON.

        Args:
            topic: The high-level topic for the generated task.
            difficulty: The requested difficulty label.
            focus_subtopic: The specific subtopic the question should focus on.

        Returns:
            str: The complete prompt text to send to the Teacher LLM.
        """
        return (
            "You are an expert technical instructor designing an evaluation task.\n"
            f"Topic: {topic}\n"
            f"Focus subtopic: {focus_subtopic}\n"
            f"Difficulty: {difficulty}\n\n"
            "Respond with a single JSON object containing exactly these keys:\n"
            '  "question": a self-contained task prompt for a student,\n'
            '  "golden_reference": a correct reference solution to the task,\n'
            '  "rubric": an array of objects, each with keys '
            '"description", "max_points", "is_critical", and "validation_pattern".\n\n'
            "Critical rules for validation_pattern (this is a regex matched literally, "
            "with no understanding of code semantics, against a student's answer):\n"
            "- NEVER hardcode a specific variable, attribute, or method name the student "
            "is free to choose (e.g. do not require the literal name \"mutex\" or \"lock\", "
            "and do not require a specific method name like \"wait_for_completion\" unless "
            "the question text explicitly tells the student to name it exactly that). A "
            "student who names their lock `self._lock` instead of `self.mutex`, or their "
            "method `wait_for_tasks` instead of `wait_for_completion`, is just as correct if "
            "the question never mandated a specific name -- and a pattern bound to one "
            "specific name will wrongly fail them.\n"
            "- If a rubric item depends on a specific method or class name existing, you "
            "must also state that exact required name explicitly in the \"question\" text "
            "itself, so the requirement is unambiguous to the student; otherwise, write the "
            "name portion of the pattern as a wildcard (\\w+) too.\n"
            "- Match the underlying language feature or API being exercised using short, "
            "distinctive fragments, not a rigid whole-signature or whole-call match: prefer "
            '"threading\\.Lock\\(\\)" over trying to match an entire "def __init__(self):" '
            "line (which breaks the moment a student adds another parameter), and prefer "
            '"\\.submit\\(" over matching an exact, fixed argument list (which breaks the '
            "moment a student passes *args/**kwargs or an extra argument).\n"
            "- A pattern is only worth marking is_critical=true if it can be written "
            "this way; if a requirement cannot be checked without assuming a specific "
            "name or exact call shape, mark it is_critical=false and describe it in free "
            "text instead.\n"
            "- Before finalizing your answer, mentally check every is_critical pattern "
            "against your own golden_reference: if a pattern would not match your own "
            "correct solution, the pattern is broken -- fix it (or the golden_reference) "
            "before responding, since a broken pattern would fail every student regardless "
            "of the quality of their answer.\n\n"
            "Also, if golden_reference or any other string value contains a line break, "
            "escape it as the two characters backslash-n (\\n) inside the JSON string, "
            "never as a literal newline, since a literal newline inside a JSON string "
            "is invalid JSON.\n\n"
            "Return only the JSON object, with no additional commentary."
        )

    def __parse_rubric(self, raw_rubric_items: List[Dict[str, Any]]) -> List[RubricItem]:
        """Converts raw rubric dictionaries from the LLM response into RubricItem instances.

        Args:
            raw_rubric_items: The raw list of rubric dictionaries as parsed
                from the Teacher LLM's JSON response.

        Returns:
            List[RubricItem]: The successfully constructed rubric items.
            Malformed entries are silently skipped rather than aborting
            generation of the entire test case, consistent with the
            Micro-Resilience principle.
        """
        rubric_items: List[RubricItem] = []
        for raw_item in raw_rubric_items:
            try:
                description = str(raw_item["description"])
                max_points = float(raw_item["max_points"])
                is_critical = bool(raw_item.get("is_critical", False))
                validation_pattern = raw_item.get("validation_pattern")

                # A critical item cannot be constructed without a validation
                # pattern (see RubricItem's own invariant), so it is safely
                # downgraded to non-critical instead of discarding the item.
                if is_critical and not validation_pattern:
                    is_critical = False

                rubric_items.append(
                    RubricItem(
                        description_init=description,
                        max_points_init=max_points,
                        is_critical_init=is_critical,
                        validation_pattern_init=validation_pattern,
                    )
                )
            except (KeyError, TypeError, ValueError):
                # Skip malformed rubric entries rather than crashing the
                # entire generation pipeline over one bad item.
                continue
        return rubric_items
