from training.pipeline.final_train.feature_engineering import FeatureEngineeringTrainingPipeline
from training.pipeline.final_train.model_trainer import ModelTrainingPipeline
from training.pipeline.final_train.model_evaluation import ModelEvaluationTrainingPipeline

from training.custom_logging import info_logger, error_logger
import sys
import os


PIPELINE = "Feature Engineering Training Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<")
feature_engineering = FeatureEngineeringTrainingPipeline()
feature_engineering.main()
info_logger.info(f">>>>>>> {PIPELINE} completed <<<<<<<<")


PIPELINE = "Model Evaluation Training Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<")
model_evaluation = ModelEvaluationTrainingPipeline()
model_evaluation.main()
info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")


PIPELINE = "Model Trainer Training Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<")
model_trainer = ModelTrainingPipeline()
model_trainer.main()
info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")