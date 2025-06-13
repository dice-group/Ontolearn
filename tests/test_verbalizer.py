import unittest
from unittest.mock import patch, MagicMock
from ontolearn.verbalizer import verbalize_learner_prediction

class TestVerbalizeLearnerPrediction(unittest.TestCase):
    def test_verbalize_learner_prediction_prints_three_verbalizations_with_string(self):
        mock_response = "Mocked verbalization"
        with patch("ontolearn.verbalizer.LLMVerbalizer", return_value=lambda text: mock_response), \
             patch("builtins.print") as mock_print:
            verbalize_learner_prediction("Some prediction")
            mock_print.assert_called_once_with([mock_response, mock_response, mock_response])

    def test_verbalize_learner_prediction_prints_three_verbalizations_with_object(self):
        mock_response = "Mocked verbalization"
        mock_obj = MagicMock()  # Simulate a class expression object
        with patch("ontolearn.verbalizer.LLMVerbalizer", return_value=lambda text: mock_response), \
             patch("builtins.print") as mock_print:
            verbalize_learner_prediction(mock_obj)
            mock_print.assert_called_once_with([mock_response, mock_response, mock_response])

    def test_verbalize_learner_prediction_raises_on_none(self):
        with self.assertRaisesRegex(AssertionError, "Learner prediction cannot be None"):
            verbalize_learner_prediction(None)

if __name__ == "__main__":
    unittest.main()