from training.pipeline.common.data_ingestion import DataIngestionPipeline
from training.pipeline.common.data_validation import DataValidationPipeline
from training.pipeline.cross_val.cross_val import CrossValPipeline

from training.custom_logging import info_logger, error_logger
import sys
import os


PIPELINE = "Data Ingestion Training Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<")
try:
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
except Exception as e:
    info_logger.error(f"Pipeline execution failed: {e}")
    info_logger.info(f">>>>> {PIPELINE} completed <<<<")



PIPELINE = "Data Validation Training Pipeline"
info_logger.info(f">>>>>>>> {PIPELINE} started <<<<<<<<<")
try:
    data_validation = DataValidationPipeline()
    data_validation.main()
except Exception as e:
    info_logger.error(f"Pipeline execution failed: {e}")
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")


PIPELINE = "Cross Validition Training Pipeline"
try:
    info_logger.info(f">>>>> {PIPELINE} started <<<<")
    cross_val = CrossValPipeline()
    cross_val.main()
    info_logger.info(f">>>>> {PIPELINE} completed <<<<")
except Exception as e:
        info_logger.error(f"Pipeline failed: {e}")




