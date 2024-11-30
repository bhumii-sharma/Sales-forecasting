import os
import json
from joblib import dump
from sklearn.metrics import classification_report
from pathlib import Path
from training.exception import CrossValError, handle_exception
from training.custom_logging import info_logger, error_logger
from training.entity.config_entity import CrossValConfig
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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


if __name__ == "__main__":
    try:
        info_logger.info("Starting CrossValidation testing process...")

        # Initialize the configuration with updated paths
        config = CrossValConfig(
            root_dir="artifacts/cross_val",
            extracted_features="artifacts/cross_val/features",
            random_search_models_rf={
                'model_fold_1': 'artifacts/cross_val/kfold_models/random_forest/model_fold_1.joblib',
                'model_fold_2': 'artifacts/cross_val/kfold_models/random_forest/model_fold_2.joblib',
                'best_model': 'artifacts/cross_val/kfold_models/random_forest/best_model.joblib',
                'metrics': {
                    'fold_1': 'artifacts/cross_val/kfold_models/random_forest/metrics/metrics_fold_1.json',
                    'fold_2': 'artifacts/cross_val/kfold_models/random_forest/metrics/metrics_fold_2.json',
                },
                'best_model_params': {
                    'fold_1': 'artifacts/cross_val/kfold_models/random_forest/tuned_params/tuned_params_fold_1.json',
                    'fold_2': 'artifacts/cross_val/kfold_models/random_forest/tuned_params/tuned_params_fold_2.json',
                },
            },
            model_cache_rf="artifacts/cross_val/model_cache_rf",
            train_data_path={
                'train': 'artifacts/cross_val/data_for_final_train/Train.npz',
            },
            test_data_path={
                'test': 'artifacts/cross_val/data_for_final_train/Test.npz',
            },
            model_name="random_forest",
            STATUS_FILE="artifacts/cross_val/status.txt",
            metric_file_name_rf="artifacts/cross_val/kfold_models/random_forest/metrics.json",
            best_model_params_rf="artifacts/cross_val/kfold_models/random_forest/tuned_params.json",
        )

        # Initialize CrossValidation class
        cross_validation = CrossValidation(config=config)

        # Mock data for testing
        X_train = np.random.rand(100, 10)
        X_test = np.random.rand(20, 10)
        y_train = np.random.randint(0, 2, 100)
        y_test = np.random.randint(0, 2, 20)
        groups_train = np.random.randint(0, 5, 100)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Test individual methods
        info_logger.info("Testing save_train_test_data_for_final_train...")
        cross_validation.save_train_test_data_for_final_train(X_train, X_test, y_train, y_test, groups_train)

        info_logger.info("Testing save_model...")
        cross_validation.save_model(model, fold_number=1)

        info_logger.info("Testing save_best_model...")
        cross_validation.save_best_model(model)

        info_logger.info("Testing save_metrics...")
        cross_validation.save_metrics({"accuracy": 0.95}, fold_number=1)

        info_logger.info("Testing save_best_model_params...")
        cross_validation.save_best_model_params(model, fold_number=1)

        info_logger.info("CrossValidation testing completed successfully.")

    except Exception as e:
        error_logger.error("An error occurred during CrossValidation testing.")
        handle_exception(e, CrossValError)
