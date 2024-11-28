import os
import json
from joblib import dump
from sklearn.metrics import classification_report
from pathlib import Path
from training.exception import CrossValError, handle_exception
from training.custom_logging import info_logger, error_logger
from training.entity.config_entity import CrossValConfig
import numpy as np


class CrossValidation:
    def __init__(self, config: CrossValConfig):
        self.config = config

    def save_train_test_data_for_final_train(self, X_train, X_test, y_train, y_test, groups_train):
        try:
            # Ensure the directories for train and test data exist
            train_data_path = Path(self.config.train_data_path['train'])
            test_data_path = Path(self.config.test_data_path['test'])
            train_data_path.parent.mkdir(parents=True, exist_ok=True)
            test_data_path.parent.mkdir(parents=True, exist_ok=True)

            # Save X_train, y_train, and groups_train to Train.npz
            np.savez(train_data_path, X_train=X_train, y_train=y_train, groups_train=groups_train)
            np.savez(test_data_path, X_test=X_test, y_test=y_test)

            info_logger.info(f"Train.npz saved at {train_data_path} and Test.npz saved at {test_data_path}")
        except Exception as e:
            handle_exception(e, CrossValError)

    def save_model(self, model, fold_number: int):
        try:
            # Save the model for this fold
            model_path = Path(self.config.random_search_models_rf[f"model_fold_{fold_number}"])
            model_path.parent.mkdir(parents=True, exist_ok=True)
            dump(model, model_path)
            info_logger.info(f"Model for fold {fold_number} saved at {model_path}")
        except Exception as e:
            handle_exception(e, CrossValError)

    def save_best_model(self, model):
        try:
            # Save the best model
            best_model_path = Path(self.config.random_search_models_rf['best_model'])
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            dump(model, best_model_path)
            info_logger.info(f"Best model saved at {best_model_path}")
        except Exception as e:
            handle_exception(e, CrossValError)

    def save_metrics(self, report_dict, fold_number: int):
        try:
            # Save metrics for this fold as a JSON file
            metrics_path = Path(self.config.random_search_models_rf['metrics'][f"fold_{fold_number}"])
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(report_dict, f, indent=4)
            info_logger.info(f"Metrics for fold {fold_number} saved at {metrics_path}")
        except Exception as e:
            handle_exception(e, CrossValError)

    def save_best_model_params(self, best_model, fold_number: int):
        try:
            # Save the best model parameters as a JSON file
            best_model_params = best_model.get_params()
            serializable_params = {k: v for k, v in best_model_params.items() if self.is_json_serializable(v)}
            params_path = Path(self.config.random_search_models_rf['best_model_params'][f"fold_{fold_number}"])
            params_path.parent.mkdir(parents=True, exist_ok=True)
            with open(params_path, 'w') as f:
                json.dump(serializable_params, f, indent=4)

            info_logger.info(f"Best model parameters for fold {fold_number} saved at {params_path}")
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
