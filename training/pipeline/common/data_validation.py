from training.configuration_manager.configuration import ConfigurationManager
from training.components.common.data_validation import DataValidation
from training.custom_logging import info_logger


PIPELINE = "Data Validation Training Pipeline"


class DataValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Log pipeline start
            info_logger.info(f"Starting {PIPELINE}...")

            # Load the data validation configuration object
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()

            # Passing the data validation configuration object to the component
            data_validation = DataValidation(config=data_validation_config)

            # Execute the data validation process
            data_validation.validate_data()

            # Log successful completion
            info_logger.info(f"{PIPELINE} completed successfully.")
        except Exception as e:
            # Log any exceptions and re-raise them
            info_logger.error(f"Error occurred in {PIPELINE}: {e}")
            raise


if __name__ == "__main__":
    # Initialize and run the pipeline
    info_logger.info(f">>>>>>>> {PIPELINE} started <<<<<<<<<")
    try:
        obj = DataValidationPipeline()
        obj.main()
    except Exception as e:
        info_logger.error(f"Pipeline execution failed: {e}")
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")
