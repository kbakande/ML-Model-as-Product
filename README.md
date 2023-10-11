# MLPipeline Project
The `ml_pipeline_project` is a machine learning project that provides a pipeline for preprocessing, training, and predicting using the XGBoost model. The project is structured to facilitate easy setup, training, and prediction using a Makefile.

## Project Structure

```graphql
ml_pipeline_project/
│
├── Makefile
│
├── src/
│   ├── __init__.py
│   └── mlpipeline.py  # code file containing the MLPipeline class
│
├── tests/
│   ├── __init__.py
│   └── test_mlpipeline.py  # unittest cases file
│
├── pyproject.toml
├── poetry.lock
|
├── mlpipeline_demonstration.ipynb
│
└── artifacts/
    └── model/
```

## Installation
1. Ensure you have Python version between 3.11 and 3.13 installed.
2. Install the required dependencies using [Poetry](https://python-poetry.org/):

```bash
poetry install
```

## Usage
The project provides a Makefile with several commands to facilitate the execution of the pipeline:

* Preprocess Data: Downloads and preprocesses the data from a Google Cloud Storage URL.

```bash
make preprocess
```

* Train Model: Trains the XGBoost model using the preprocessed data.

```bash
make train
```

* Predict: Uses the trained model to make predictions on new data.

```bash
make predict
```

* Run Tests: Executes the unittest cases.

```bash
make test
```

* Run Pytest: Executes the pytest cases.

```bash
pytest pytest_mlpipeline.py
```

* Run All: Executes the preprocess, train, and predict commands in sequence.

```bash
make all
```

* Help: Displays available commands and their descriptions.

```bash
make help
```

## Dependencies
The project uses several Python libraries, including:

* numpy
* pandas
* xgboost
* google-cloud-storage
* joblib
* matplotlib
* scikit-learn
* pytest
 

## Author
[Kabeer Akande](https://www.linkedin.com/in/koakande/)