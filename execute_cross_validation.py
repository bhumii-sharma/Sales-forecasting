from training.pipeline.common.data_ingestion import DataIngestionPipeline
from training.pipeline.common.data_validation import DataValidationPipeline
from training.pipeline.cross_val.cross_val import CrossValPipeline

from training.custom_logging import info_logger, error_logger
import sys
import os


PIPELINE = "Data Ingestion Training Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<")
data_ingestion = DataIngestionPipeline()
data_ingestion.main()
info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")



PIPELINE = "Data Validition Training Pipeline"
info_logger.info(f">>>>>>>> {PIPELINE} started <<<<<<<<<")
data_validation = DataValidationPipeline()
data_validation.main()
info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")



PIPELINE = "Cross Validition Training Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<")
cross_val = CrossValPipeline()
cross_val.main()
info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")




