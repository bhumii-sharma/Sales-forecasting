from training.constants import *
from training.entity.config_entity import DataIngestionConfig
from training.entity.config_entity import DataValidationConfig
from training.entity.config_entity import FeatureEngineeringConfig
from training.entity.config_entity import ModelTrainerConfig
from training.entity.config_entity import ModelEvaluationConfig
from training.entity.config_entity import CrossValConfig

import os

class CrossValConfig:
    """
    Configuration class for managing K-Fold Cross-Validation settings and paths.
    """
    def __init__(self, root_dir, extracted_features, model_cache_rf, train_data_path, 
                 test_data_path, model_name, STATUS_FILE, metric_file_name_rf, 
                 best_model_params_rf, k_folds=5):
        self.root_dir = root_dir
        self.extracted_features = extracted_features
        self.model_cache_rf = model_cache_rf
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.model_name = model_name
        self.STATUS_FILE = STATUS_FILE
        self.metric_file_name_rf = metric_file_name_rf
        self.best_model_params_rf = best_model_params_rf
        self.k_folds = k_folds


def create_directories(directories):
    """
    Creates directories if they don't already exist.
    :param directories: List of directory paths to create.
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def get_cross_val_config(config) -> CrossValConfig:
    """
    Retrieves the cross-validation configuration, creates required directories, and sets up paths for K-Fold Cross-Validation.

    :param config: The base configuration object containing paths and parameters.
    :return: CrossValConfig object containing configuration for K-Fold cross-validation.
    """
    # Create required directories for cross-validation
    create_directories([config.root_dir])
    create_directories([config.extracted_features, config.model_cache_rf])
    create_directories([config.train_data_path, config.test_data_path])
    create_directories([config.metric_file_name_rf, config.best_model_params_rf])

    # Configure for K-Fold Cross-Validation
    cross_val_config = CrossValConfig(
        root_dir=config.root_dir,
        extracted_features=config.extracted_features,
        model_cache_rf=config.model_cache_rf,
        train_data_path=config.train_data_path,
        test_data_path=config.test_data_path,
        model_name=config.model_name,
        STATUS_FILE=config.STATUS_FILE,
        metric_file_name_rf=config.metric_file_name_rf,
        best_model_params_rf=config.best_model_params_rf,
        k_folds=config.k_folds  # New field specific to K-Fold CV
    )

    return cross_val_config
