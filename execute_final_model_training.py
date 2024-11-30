from training.pipeline.final_train.feature_engineering import FeatureEngineeringTrainingPipeline
from training.pipeline.final_train.model_trainer import ModelTrainingPipeline
from training.pipeline.final_train.model_evaluation import ModelEvaluationTrainingPipeline

from training.custom_logging import info_logger, error_logger
import sys
import os


PIPELINE = "Feature Engineering Training Pipeline"
try:
    info_logger.info(f">>>>> {PIPELINE} started <<<<")
    feature_engineering = FeatureEngineeringTrainingPipeline()
    feature_engineering.main()
    info_logger.info(f">>>>> {PIPELINE} completed successfully <<<<")
except Exception as e:
    info_logger.error(f"Pipeline execution failed: {e}")


PIPELINE = "Model Evaluation Training Pipeline"
try:
    info_logger.info(f">>>>> {PIPELINE} started <<<<")
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.main()
    info_logger.info(f">>>>> {PIPELINE} completed successfully <<<<")
except Exception as e:
    info_logger.error(f"Pipeline execution failed: {e}")


PIPELINE = "Model Trainer Training Pipeline"
try:
    info_logger.info(f">>>>> {PIPELINE} started <<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    info_logger.info(f">>>>> {PIPELINE} completed successfully <<<<")
except Exception as e:
    info_logger.error(f"Pipeline execution failed: {e}")
