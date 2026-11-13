"""Offline invalid-grade regressions. Run: PYTHONPATH=src python3 test_invalid_grades.py"""
import unittest
from llm_exam_board import RubricItem, TestCase, StudentAnswer, JudgeEngine, EnsembleManager, GradedResponse

class InvalidGradeTests(unittest.TestCase):
    def setUp(self):
        self.task = TestCase('testing','easy','Answer correctly','correct',[RubricItem('Correctness',10)])
        self.answer = StudentAnswer(self.task.test_case_id,'answer')
    def grade(self, text, name='judge'):
        return JudgeEngine(lambda _:text,name).grade(self.task,self.answer)
    def test_unparseable_is_not_scored(self):
        for text in ('not JSON at all','', '```json\n{"broken":'):
            with self.subTest(text=text):
                result=self.grade(text)
                self.assertFalse(result.is_valid)
                self.assertIsNone(result.raw_llm_score)
                self.assertIsNone(result.final_score)
                self.assertTrue(result.invalid_reason)
    def test_valid_empty_deductions_remain_full_marks(self):
        result=self.grade('{"deductions":[],"feedback_summary":"correct"}')
        self.assertTrue(result.is_valid)
        self.assertEqual(result.final_score,10)
    def test_invalid_excluded_but_retained_in_audit(self):
        bad=self.grade('not JSON','bad')
        good=self.grade('{"deductions":[],"feedback_summary":"correct"}','good')
        result=EnsembleManager().build_consensus([bad,good])
        self.assertEqual(result.consensus_score,10)
        self.assertEqual(result.score_std_dev,0)
        self.assertEqual(len(result.graded_responses),2)
        self.assertEqual(result.outlier_judge_ids,[])
    def test_all_invalid_raises_instead_of_zero_or_full_marks(self):
        with self.assertRaisesRegex(ValueError,'no valid judge grades'):
            EnsembleManager().build_consensus([self.grade('bad')])
    def test_invalid_model_rejects_numeric_score(self):
        with self.assertRaisesRegex(ValueError,'must not contain a score'):
            GradedResponse('test','answer','judge',0,0,is_valid_init=False,invalid_reason_init='bad')
if __name__=='__main__': unittest.main()
