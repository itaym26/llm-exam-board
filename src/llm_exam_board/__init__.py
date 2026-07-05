"""
llm_exam_board: a V14 Teacher-Student-Judge LLM evaluation pipeline.

This package re-exports the public API of every module in the pipeline so
that consumers can simply write `from llm_exam_board import ...` rather
than reaching into individual submodules.
"""

from .generators import ConfigManager, LLMClient, QuestionGenerator
from .interfaces import (
    DeductionItem,
    EnsembleResult,
    GradedResponse,
    RubricItem,
    StudentAnswer,
    TestCase,
)
from .judge_engine import JudgeEngine
from .orchestrator import EnsembleManager, EvaluationOrchestrator
from .resilience import ResilienceUtils
from .responders import StudentResponder

__all__ = [
    "ConfigManager",
    "LLMClient",
    "QuestionGenerator",
    "DeductionItem",
    "EnsembleResult",
    "GradedResponse",
    "RubricItem",
    "StudentAnswer",
    "TestCase",
    "JudgeEngine",
    "EnsembleManager",
    "EvaluationOrchestrator",
    "ResilienceUtils",
    "StudentResponder",
]
