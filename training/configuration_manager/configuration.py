import os
from training.constants import *  # Ensure constants are well-defined
from training.configuration_manager.configuration
from training.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    FeatureEngineeringConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    CrossValConfig,
)


def create_directories(directories):
    """
    Creates directories if they don't already exist.
    :param directories: List of directory paths to create.
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


class ConfigManager:
    """
    Configuration Manager to handle different configurations for data ingestion,
    validation, feature engineering, training, and evaluation.
    """

    def __init__(self, config):
        """
        Initialize with a config object.
        :param config: Configuration object containing paths and parameters.
        """
        self.config = config

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves the Data Ingestion configuration and ensures directories exist.
        :return: DataIngestionConfig object.
        """
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source=config.source,
            data_dir=config.data_dir,
            STATUS_FILE=config.STATUS_FILE
        )
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Retrieves the Data Validation configuration and ensures directories exist.
        :return: DataValidationConfig object.
        """
        config = self.config.data_validation
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            data_dir=config.source,
            STATUS_FILE=config.STATUS_FILE
        )
        return data_validation_config

    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        """
        Retrieves the Feature Engineering configuration and ensures directories exist.
        :return: FeatureEngineeringConfig object.
        """
        config = self.config.feature_engineering
        create_directories([config.root_dir])

        feature_engineering_config = FeatureEngineeringConfig(
            root_dir=config.root_dir,
            pipeline_path=config.pipeline_path,
            STATUS_FILE=config.STATUS_FILE
        )
        return feature_engineering_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Retrieves the Model Trainer configuration and ensures directories exist.
        :return: ModelTrainerConfig object.
        """
        config = self.config.model_trainer
        create_directories([config.root_dir, config.saved_models_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            saved_models_dir=config.saved_models_dir,
            hyperparameters=config.hyperparameters,
            STATUS_FILE=config.STATUS_FILE
        )
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Retrieves the Model Evaluation configuration and ensures directories exist.
        :return: ModelEvaluationConfig object.
        """
        config = self.config.model_evaluation
        create_directories([config.root_dir, config.evaluation_reports_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            evaluation_reports_dir=config.evaluation_reports_dir,
            comparison_metrics=config.comparison_metrics,
            STATUS_FILE=config.STATUS_FILE
        )
        return model_evaluation_config

    def get_cross_val_config(self) -> CrossValConfig:
        """
        Retrieves the cross-validation configuration, creates required directories, and sets up paths for K-Fold Cross-Validation.
        :return: CrossValConfig object containing configuration for K-Fold cross-validation.
        """
        config = self.config.cross_val

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
