import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from src.mlpipeline import MLPipeline

# Setup and Teardown using pytest fixtures
@pytest.fixture(scope="module")
def setup_teardown():
    model_dir = "artifacts/model"
    test_csv = os.path.abspath("test_data.csv")
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
    pd.DataFrame(data).to_csv(test_csv, index=False)
    pipeline = MLPipeline(model_directory=model_dir)
    yield test_csv, pipeline
    if os.path.exists(test_csv):
        os.remove(test_csv)

def test_score_data_success(setup_teardown):
    test_csv, pipeline = setup_teardown
    with patch('src.mlpipeline.MLPipeline.download_gcs_csv', return_value=pd.read_csv(test_csv)):
        with patch('xgboost.Booster.predict', return_value=np.array([0.6, 0.4])):
            result = pipeline.score_data(test_csv)
            assert isinstance(result, pd.DataFrame)

def test_predictions_length(setup_teardown):
    test_csv, pipeline = setup_teardown
    with patch('src.mlpipeline.MLPipeline.download_gcs_csv', return_value=pd.read_csv(test_csv)):
        with patch('xgboost.Booster.predict', return_value=np.array([0.6, 0.4])):
            predictions = pipeline.score_data(test_csv)
            assert len(predictions) == 2

def test_predictions_values(setup_teardown):
    test_csv, pipeline = setup_teardown
    with patch('src.mlpipeline.MLPipeline.download_gcs_csv', return_value=pd.read_csv(test_csv)):
        with patch('xgboost.Booster.predict', return_value=np.array([0.6, 0.4])):
            predictions = pipeline.score_data(test_csv)['Adopted_prediction'].map({'Yes': 1, 'No': 0}).values
            assert np.all(predictions >= 0)
            assert np.all(predictions <= 1)