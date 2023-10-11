# Define all functions
import os
from io import BytesIO
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from google.cloud import storage
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, accuracy_score, recall_score,
                             precision_score, cohen_kappa_score)

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


class MLPipeline:
    def __init__(self, model_directory: str = 'artifacts/model', 
                 output_directory: str = 'output'):
        """
        Initialize the MLPipeline.

        Args:
            model_directory (str): The directory where the model artifacts are stored.
            output_directory (str): The directory where the model predictions are stored.
        """
        self.model_directory = model_directory
        self.output_directory= output_directory

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        self.output_path = os.path.join(self.output_directory, 'results.csv')
        
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        self.encoder_path = os.path.join(self.model_directory, 'encoder.pkl')
        self.model_path = os.path.join(self.model_directory, 'xgb_model.model')

    def download_gcs_csv(self, gcs_url: str) -> pd.DataFrame:
        """
        Downloads a CSV file from the specified Google Cloud Storage URL.

        Args:
        - url (str): The GCS URL to download from, e.g., "gs://bucket-name/path/to/file.csv"

        Returns:
        - df: Pandas dataframe with the downloaded csv file.

        Raises:
        - ValueError: If the provided URL does not have a valid GCS format.
        """
        # Extract bucket and blob info from GCS URL
        if not gcs_url.startswith("gs://"):
            raise ValueError("Invalid GCS URL format")

        parts = gcs_url[5:].split("/", 1)

        if len(parts) != 2:
            raise ValueError("Invalid GCS URL format")

        bucket_name, blob_name = parts

        # Create a client
        client = storage.Client()

        # Access the bucket and blob
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Read the contents into a BytesIO stream and then load into Pandas
        data = blob.download_as_bytes()
        df = pd.read_csv(BytesIO(data))

        return df

    def encode_and_save_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes categorical columns of a dataframe using OrdinalEncoder and saves the encoder to a directory.

        Args:
        - df (pandas.DataFrame): The input dataframe.
        - save_directory (str): The directory to save the encoder.

        Returns:
        - pandas.DataFrame: The dataframe with encoded columns.
        """
        
        # Selecting categorical columns and excluding target
        cat_columns = df.select_dtypes(['object']).columns.tolist()[:-1]
  
        # Initialize and fit the OrdinalEncoder
        encoder = OrdinalEncoder()
        encoded_data = encoder.fit_transform(df[cat_columns])

        # Replace the original columns with encoded data
        df_encoded = df.copy()
        df_encoded[cat_columns] = encoded_data

        # Save the encoder
        joblib.dump(encoder, self.encoder_path)

        return df_encoded


    def split_data(self, df: pd.DataFrame, target: str = 'Adopted', 
                   train_ratio: float = 0.6, test_ratio: float = 0.2, 
                   val_ratio: float = 0.2) -> tuple:
        """
        Splits the dataframe into training, validation, and test sets.

        Parameters:
        - df: DataFrame containing the data.
        - train_ratio: Proportion of the dataset to be used for training.
        - test_ratio: Proportion of the dataset to be used for testing.
        - val_ratio: Proportion of the dataset to be used for validation.

        Returns:
        - X_train, X_val, X_test, y_train, y_val, y_test: Split data sets.
        """

        # Ensure ratios sum to 1
        assert train_ratio + test_ratio + val_ratio == 1, "Ratios must sum to 1."

        # Split data into features and target
        # Convert 'yes' to 1 and 'no' to 0, required by xgboost
        df[target] = df[target].replace({'Yes': 1, 'No': 0})

        X = df.drop(target, axis=1)
        y = df[target]

        # Split data into training and temp sets (test + validation)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_ratio + val_ratio, random_state=42)

        # Calculate the ratio for the test set relative to the temp set
        relative_test_ratio = test_ratio / (test_ratio + val_ratio)

        # Split temp data into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=relative_test_ratio, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, 
                           y_test: pd.Series, max_depth: int, eta: float, num_round: int, 
                           early_stopping_rounds: int, model_path: str) -> tuple:
        """
        Train an XGBoost model, evaluate its performance on the test set, and save the trained model.

        Parameters:
        - X_train, y_train: Training data and labels.
        - X_val, y_val: Validation data and labels.
        - X_test, y_test: Test data and labels.
        - max_depth: Maximum depth of the trees.
        - eta: Learning rate.
        - num_round: Number of boosting rounds.
        - early_stopping_rounds: Activates early stopping. Validation error needs to decrease at least every <early_stopping_rounds> round(s) to continue training.
        - model_path: Path to save the trained model.

        Returns:
        - Trained model.
        """

        # Convert the datasets into DMatrix format
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, enable_categorical=True)

        # Define training parameters
        param = {
            'max_depth': max_depth,
            'eta': eta,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }

        # Train the model
        evals_result = {}  # This dictionary will store evaluation results
        bst = xgb.train(
            param, 
            dtrain, 
            num_round, 
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=False
        )

        # Predict on the test set and convert to binary
        y_pred_prob = bst.predict(dtest)
        y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]

        # Log performance metrics
        logging.info("Performance Metrics on Test Set:")
        logging.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        logging.info(f"Recall: {recall_score(y_test, y_pred):.4f}")
        logging.info(f"Precision: {precision_score(y_test, y_pred):.4f}")
        logging.info(f"Cohen’s kappa: {cohen_kappa_score(y_test, y_pred):.4f}")

        # Save the model
        bst.save_model(self.model_path)

        return bst, evals_result

    def load_model_and_encoder(self) -> tuple:
        """
        Load the XGBoost model and the encoder from the specified paths.

        Returns:
            tuple: A tuple containing the loaded XGBoost model and the encoder.
        """
        # Load Model
        bst = xgb.Booster()
        bst.load_model(self.model_path)

        # Load Encoder
        encoder = joblib.load(self.encoder_path)
        return bst, encoder

    def score_data(self, gcs_url: str) -> pd.DataFrame:
        """
        Score the data from the given Google Cloud Storage URL using the loaded model.

        Args:
            gcs_url (str): The URL of the data in Google Cloud Storage.

        Returns:
            pd.DataFrame: The original data with an additional column for the model's predictions.
        """
        # Load Model and Encoder
        bst, encoder = self.load_model_and_encoder()
        
        # Fetch the data from GCS URL
        df = self.download_gcs_csv(gcs_url)

        # Backup the original dataframe before any transformations
        original_df = df.copy()

        # Drop the target column if it exists
        if 'Adopted' in df.columns:
            df.drop(columns=['Adopted'], inplace=True)

        # Use the encoder to transform the categorical columns
        cat_columns = df.select_dtypes(['object']).columns.tolist()
        df[cat_columns] = encoder.transform(df[cat_columns])

        # Convert to DMatrix and Predict
        data = xgb.DMatrix(df)
        probabilities = bst.predict(data)

        # Convert probabilities to 'Yes' or 'No'
        predictions = ['Yes' if prob > 0.5 else 'No' for prob in probabilities]
        original_df['Adopted_prediction'] = predictions

        return original_df
    
        # set up functions to plot training evolution
    def plot_evaluation(self, evals_result: dict, metric: str = 'logloss', 
                        xlabel: str = 'Boosting Round', ylabel: str = 'Log Loss', 
                        title: str = 'XGBoost Log Loss') -> None:
        """
        Plot training and validation evaluation results.

        Parameters:
        - evals_result: Evaluation results from XGBoost training.
        - metric: Name of the metric to be plotted.
        - xlabel: Label for the x-axis.
        - ylabel: Label for the y-axis.
        - title: Title for the plot.

        Returns:
        - None
        """

        # Extract epochs and metrics from the results
        epochs = len(evals_result['train'][metric])
        x_axis = range(0, epochs)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(x_axis, evals_result['train'][metric], label='Train')
        ax.plot(x_axis, evals_result['val'][metric], label='Validation')
        ax.legend()
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.show()
    
    # refactor above methods into preprocess, train and predict
    def preprocess(self, gcs_url: str) -> pd.DataFrame:
        """
        Preprocess the data from the given Google Cloud Storage URL.

        Args:
            gcs_url (str): The URL of the data in Google Cloud Storage.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        df = self.download_gcs_csv(gcs_url)
        df_encoded = self.encode_and_save_categorical_columns(df)
        return df_encoded

    def train(self, df: pd.DataFrame, **kwargs) -> tuple:
        """
        Train the model using the provided dataframe.

        Args:
            df (pd.DataFrame): The training data.

        Returns:
            tuple: The trained model and evaluation results.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
        max_depth = kwargs.get('max_depth', 6)
        eta = kwargs.get('eta', 0.3)
        num_round = kwargs.get('num_round', 1000)
        early_stopping_rounds = kwargs.get('early_stopping_rounds', 10)
        model_path = os.path.join(self.model_directory, 'xgb_model.model')
        self.model, self.evals_result = self.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, max_depth, eta, num_round, early_stopping_rounds, model_path)
        return self.model, self.evals_result
    
    def predict(self, gcs_url: str) -> pd.DataFrame:
        """
        Predict using the model on data from the given Google Cloud Storage URL.

        Args:
            gcs_url (str): The URL of the data in Google Cloud Storage.

        Returns:
            pd.DataFrame: The predictions.
        """
        predictions_df = self.score_data(gcs_url)
        # save to a file
        predictions_df.to_csv(self.output_path, index=False)
        return predictions_df