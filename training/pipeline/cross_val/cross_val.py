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
