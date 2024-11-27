import os
import json
from joblib import dump
from sklearn.metrics import classification_report
from training.exception import DataIngestionError, handle_exception
from training.custom_logging import info_logger, error_logger
from training.entity.config_entity import DataIngestionConfig
from training.configuration_manager.configuration import ConfigManager
import numpy as np


class CrossValidation:
    def __init__(self, config: CrossValConfig):
        self.config = config

    def save_train_test_data_for_final_train(self, X_train, X_test, y_train, y_test, groups_train):
        try:
            # Ensure the directories for train and test data exist
            if not os.path.exists(self.config.train_data_path):
                os.makedirs(self.config.train_data_path)
            if not os.path.exists(self.config.test_data_path):
                os.makedirs(self.config.test_data_path)

            # Save X_train, y_train, and groups_train to train.npz
            np.savez(os.path.join(self.config.train_data_path, 'Train.npz'),
                     X_train=X_train, y_train=y_train, groups_train=groups_train)
            np.savez(os.path.join(self.config.test_data_path, 'Test.npz'),
                     X_test=X_test, y_test=y_test)

            info_logger.info(f"Train.npz and Test.npz saved at {self.config.train_data_path} and {self.config.test_data_path}")
        except Exception as e:
            handle_exception(e, CrossValError)

    def save_model(self, model, fold_number: int):
        try:
            # Ensure the directory exists for saving models
            if not os.path.exists(self.config.random_search_models_rf):
                os.makedirs(self.config.random_search_models_rf)

            # Save the model for this fold
            model_path = os.path.join(self.config.random_search_models_rf, f"model_fold_{fold_number}.joblib")
            dump(model, model_path)
            info_logger.info(f"Model for fold {fold_number} saved at {model_path}")
        except Exception as e:
            handle_exception(e, CrossValError)

    def save_best_model(self, model):
        try:
            # Ensure the directory exists for saving the best model
            if not os.path.exists(self.config.random_search_models_rf):
                os.makedirs(self.config.random_search_models_rf)

            # Save the best model
            best_model_path = os.path.join(self.config.random_search_models_rf, "best_model.joblib")
            dump(model, best_model_path)
            info_logger.info(f"Best model saved at {best_model_path}")
        except Exception as e:
            handle_exception(e, CrossValError)

    def save_metrics(self, report_dict, fold_number: int):
        try:
            # Ensure the metrics directory exists
            if not os.path.exists(self.config.metric_file_name_rf):
                os.makedirs(self.config.metric_file_name_rf)

            # Save metrics for this fold as a JSON file
            metrics_path = os.path.join(self.config.metric_file_name_rf, f"metrics_fold_{fold_number}.json")
            with open(metrics_path, 'w') as f:
                json.dump(report_dict, f, indent=4)
            info_logger.info(f"Metrics for fold {fold_number} saved at {metrics_path}")
        except Exception as e:
            handle_exception(e, CrossValError)

    def save_best_model_params(self, best_model, fold_number: int):
        try:
            # Ensure the best model parameters directory exists
            if not os.path.exists(self.config.best_model_params_rf):
                os.makedirs(self.config.best_model_params_rf)

            # Save the best model parameters as a JSON file
            best_model_params = best_model.get_params()
            serializable_params = {k: v for k, v in best_model_params.items() if self.is_json_serializable(v)}
            best_model_params_path = os.path.join(self.config.best_model_params_rf, f"best_params_fold_{fold_number}.json")

            with open(best_model_params_path, 'w') as f:
                json.dump(serializable_params, f, indent=4)

            info_logger.info(f"Best model parameters for fold {fold_number} saved at {best_model_params_path}")
        except Exception as e:
            handle_exception(e, CrossValError)

    def is_json_serializable(self, value):
        """
        Check if a value is JSON serializable.
        """
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False
