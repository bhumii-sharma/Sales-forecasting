from training.constants import *
from training.entity.config_entity import DataIngestionConfig
from training.entity.config_entity import DataValidationConfig
from training.entity.config_entity import FeatureEngineeringConfig
from training.entity.config_entity import ModelTrainerConfig
from training.entity.config_entity import ModelEvaluationConfig
from training.entity.config_entity import CrossValConfig

import os

        create_directories([self.config.artifacts_root])
#1
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        

        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source=config.source,
            data_dir=config.data_dir,
            STATUS_FILE=config.STATUS_FILE
        )
        return data_ingestion_config
#2    
    def get_data_validation_config(self) -> DataValidationConfig:
        config= self.config.data_validation
        #schema = self.schema.COLUMNS - Not needed

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir= config.root_dir,
            data_dir= config.source,
            STATUS_FILE= config.STATUS_FILE
        )

        return data_validation_config


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
