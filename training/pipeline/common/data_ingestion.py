from training.configuration_manager.configuration import ConfigurationManager
from training.components.common.data_ingestion import DataIngestion
from training.custom_logging import info_logger


PIPELINE = "Data Ingestion Training Pipeline"


class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Log pipeline start
            info_logger.info(f"Starting {PIPELINE}...")

            # Load the data ingestion configuration object
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()

            # Pass the data ingestion configuration object to the Data Ingestion component
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.save_data()

            # Log pipeline completion
            info_logger.info(f"{PIPELINE} completed successfully.")
        except Exception as e:
            # Log exceptions if any
            info_logger.error(f"Error occurred in {PIPELINE}: {e}")
            raise


if __name__ == "__main__":
    # Initialize and run the pipeline
    info_logger.info(f">>>>> {PIPELINE} started <<<<")
    try:
        obj = DataIngestionPipeline()
        obj.main()
    except Exception as e:
        info_logger.error(f"Pipeline execution failed: {e}")
    info_logger.info(f">>>>> {PIPELINE} completed <<<<")
