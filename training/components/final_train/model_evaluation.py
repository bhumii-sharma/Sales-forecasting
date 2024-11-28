import os
import json
import numpy as np
from joblib import load
from sklearn.metrics import classification_report
from training.exception import ModelEvaluationError, handle_exception
from training.custom_logging import info_logger, error_logger
from training.entity.config_entity import ModelEvaluationConfig
from training.configuration_manager.configuration import ConfigurationManager

class ModelEvaluation:
    def _init_(self, config: ModelEvaluationConfig):
        """
        Initializes the ModelEvaluation class with the configuration provided.
        Args:
            config (ModelEvaluationConfig): Configuration containing paths and other parameters.
        """
        self.config = config

    def load_test_data(self) -> tuple:
        """
        Loads the test data required for model evaluation.
        Returns:
            X_test (np.ndarray): Features for testing.
            y_test (np.ndarray): True labels for testing.
        """
        try:
            info_logger.info("Loading the test data for Model Evaluation...")

            test_data_path = os.path.join(self.config.test_data_path, "Test.npz")

            # Ensure the file exists
            if not os.path.exists(test_data_path):
                raise FileNotFoundError(f"Test data file not found at {test_data_path}.")

            # Loading the .npz file
            test_data = np.load(test_data_path)

            X_test, y_test = test_data["X_test"], test_data["y_test"]

            info_logger.info(f"Successfully loaded the test data: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
            return X_test, y_test

        except Exception as e:
            handle_exception(e, ModelEvaluationError)

    def load_final_model(self):
        """
        Loads the final trained model from the specified path.
        Returns:
            final_model (model): The trained machine learning model.
        """
        try:
            final_model_path = self.config.model_path

            # Ensure the model file exists
            if not os.path.exists(final_model_path):
                raise FileNotFoundError(f"Model file not found at {final_model_path}.")

            final_model = load(final_model_path)

            info_logger.info(f"Successfully loaded the final model from {final_model_path}")
            return final_model

        except Exception as e:
            handle_exception(e, ModelEvaluationError)
    
    def evaluate_final_model(self, final_model, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluates the final model and saves the evaluation metrics.
        Args:
            final_model (model): The trained model.
            X_test (np.ndarray): Features for the test set.
            y_test (np.ndarray): True labels for the test set.
        """
        try:
            info_logger.info("Evaluating the final model...")

            # Making predictions using the final model
            y_pred = final_model.predict(X_test)

            # Generating the classification report
            report = classification_report(y_test, y_pred)

            # Log the classification report
            info_logger.info(f"Classification Report:\n{report}")

            # Saving the classification report to a file
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(report)

            # Saving the classification report as a JSON file
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            metric_file_name = os.path.join(self.config.metric_file, "metrics.json")
            with open(metric_file_name, 'w') as f:
                json.dump(report_dict, f, indent=4)

            info_logger.info("Successfully evaluated the final model and saved the metrics.")

        except Exception as e:
            handle_exception(e, ModelEvaluationError)