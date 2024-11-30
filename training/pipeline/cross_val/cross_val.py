import os
import numpy as np
from sklearn.model_selection import KFold
import joblib
from training.constants import RANDOM_STATE
import json
from joblib import dump
from sklearn.metrics import classification_report
from training.exception import DataIngestionError, handle_exception
from training.custom_logging import info_logger, error_logger
from training.entity.config_entity import DataIngestionConfig
from training.configuration_manager.configuration import ConfigurationManager


class CrossValidation:
    def __init__(self, config):
        self.config = config

    def get_data_labels(self):
        """
        Load the dataset and separate features (X) and target (y).
        """
        # Assuming data is loaded as NumPy arrays
        data = np.load(self.config.data_file)
        X, y = data['X'], data['y']
        return X, y

    def split_data(self, X, y):
        """
        Perform a single train-test split for cross-validation.
        """
        kf = KFold(n_splits=self.config.k_folds, shuffle=True, random_state=RANDOM_STATE)
        train_idx, test_idx = next(kf.split(X))  # Take the first fold as a test split
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        return X_train, X_test, y_train, y_test

    def save_train_test_data(self, X_train, X_test, y_train, y_test):
        """
        Save the train and test data for the next stages of the pipeline.
        """
        os.makedirs(self.config.root_dir, exist_ok=True)
        train_path = self.config.train_data_path
        test_path = self.config.test_data_path

        # Save as .npz files
        np.savez(train_path, X_train=X_train, y_train=y_train)
        np.savez(test_path, X_test=X_test, y_test=y_test)
        print(f"Train data saved to {train_path}")
        print(f"Test data saved to {test_path}")

    def train_kfold_models(self, X, y):
        """
        Perform K-Fold cross-validation training and save models for each fold.
        """
        kf = KFold(n_splits=self.config.k_folds, shuffle=True, random_state=RANDOM_STATE)
        fold_num = 1
        metrics = {}

        for train_idx, val_idx in kf.split(X):
            # Split data for current fold
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Initialize model
            model = self.initialize_model()

            # Train the model
            model.fit(X_train, y_train)

            # Save the model for the current fold
            model_path = self.config.model_cache_rf.format(fold=fold_num)
            joblib.dump(model, model_path)
            print(f"Fold {fold_num} model saved to {model_path}")

            # Evaluate on validation data
            val_score = model.score(X_val, y_val)
            metrics[f"fold_{fold_num}"] = {"val_score": val_score}
            fold_num += 1

        # Save metrics for all folds
        metrics_path = self.config.metric_file_name_rf
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        print(f"K-Fold metrics saved to {metrics_path}")

    def initialize_model(self):
        """
        Initialize the RandomForest model. Replace or extend for different models.
        """
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=RANDOM_STATE)

    def main(self):
        """
        Execute the cross-validation pipeline.
        """
        # Load data
        X, y = self.get_data_labels()

        # Perform train-test split
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Save train and test data
        self.save_train_test_data(X_train, X_test, y_train, y_test)

        # Perform K-Fold training and save models
        self.train_kfold_models(X_train, y_train)


  


## Checking the code 

class Config:
    """
    Configuration class to provide paths and parameters for the CrossValidation pipeline.
    """
    def __init__(self):
        # Set paths and other configuration parameters
        self.data_file = "data/dataset.npz"  # Path to the .npz file containing 'X' and 'y'
        self.k_folds = 5  # Number of K-Folds
        self.root_dir = "artifacts/cross_val"  # Root directory for saving outputs
        self.train_data_path = "artifacts/cross_val/Train.npz"  # Path to save train data
        self.test_data_path = "artifacts/cross_val/Test.npz"  # Path to save test data
        self.model_cache_rf = "artifacts/cross_val/models/fold_{fold}_model.joblib"  # Model cache format
        self.metric_file_name_rf = "artifacts/cross_val/metrics.json"  # Path to save metrics


def test_cross_validation_initialization():
    """
    Test the CrossValidation class initialization and basic attribute setup.
    """
    try:
        # Step 1: Initialize the configuration
        config = Config()

        # Step 2: Create CrossValidation instance
        cross_validation = CrossValidation(config)

        # Log the initialization details
        info_logger.info("CrossValidation class initialized successfully.")
        info_logger.info(f"Data File: {cross_validation.config.data_file}")
        info_logger.info(f"Root Directory: {cross_validation.config.root_dir}")
        info_logger.info(f"Train Data Path: {cross_validation.config.train_data_path}")
        info_logger.info(f"Test Data Path: {cross_validation.config.test_data_path}")
        info_logger.info(f"K-Folds: {cross_validation.config.k_folds}")
        info_logger.info(f"Model Cache Path: {cross_validation.config.model_cache_rf}")

    except Exception as e:
        info_logger.error(f"An error occurred during testing: {e}")


if __name__ == "__main__":
    test_cross_validation_initialization()
