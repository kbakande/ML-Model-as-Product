import unittest
from unittest.mock import patch
import os
import numpy as np
import pandas as pd
from src.mlpipeline import MLPipeline

class TestMLPipeline(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        # Define model directory and test CSV file path
        self.model_dir = "artifacts/model"
        self.test_csv = "test_data.csv"
        
        # Sample data for testing
        data = {
            "Type": ["Cat", "Cat"],
            "Age": [3, 1],
            "Breed1": ["Tabby", "Domestic Medium Hair"],
            "Gender": ["Male", "Male"],
            "Color1": ["Black", "Black"],
            "Color2": ["White", "Brown"],
            "MaturitySize": ["Small", "Medium"],
            "FurLength": ["Short", "Medium"],
            "Vaccinated": ["No", "Not Sure"],
            "Sterilized": ["No", "Not Sure"],
            "Health": ["Healthy", "Healthy"],
            "Fee": [100, 0],
            "PhotoAmt": [1, 2],
            "Adopted": ["Yes", "Yes"]
        }

        # Save the sample data to a CSV file
        pd.DataFrame(data).to_csv(self.test_csv, index=False)
        
        # Initialize the MLPipeline
        self.pipeline = MLPipeline(model_directory=self.model_dir)

    def test_score_data_success(self):
        """
        Test the score_data method for successful execution.
        """
        with patch('src.mlpipeline.MLPipeline.download_gcs_csv', return_value=pd.read_csv(self.test_csv)), \
            patch('xgboost.Booster.predict', return_value=np.array([0.6, 0.4])):
            
            try:
                result = self.pipeline.score_data(self.test_csv)
                self.assertIsInstance(result, pd.DataFrame)
            except Exception as e:
                self.fail(f"score_data() raised Exception unexpectedly: {e}")

    def test_predictions_length(self):
        """
        Test the length of predictions from the score_data method.
        """
        with patch('src.mlpipeline.MLPipeline.download_gcs_csv', return_value=pd.read_csv(self.test_csv)), \
            patch('xgboost.Booster.predict', return_value=np.array([0.6, 0.4])):
            
            predictions = self.pipeline.score_data(self.test_csv)
            self.assertEqual(len(predictions), 2)

    def test_predictions_values(self):
        """
        Test the values of predictions from the score_data method.
        """
        with patch('src.mlpipeline.MLPipeline.download_gcs_csv', return_value=pd.read_csv(self.test_csv)), \
            patch('xgboost.Booster.predict', return_value=np.array([0.6, 0.4])):
            
            predictions = self.pipeline.score_data(self.test_csv)['Adopted_prediction'].map({'Yes': 1, 'No': 0}).values
            self.assertTrue(np.all(predictions >= 0))
            self.assertTrue(np.all(predictions <= 1))

    def tearDown(self):
        """
        Clean up resources after each test.
        """
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

if __name__ == '__main__':
    unittest.main()
