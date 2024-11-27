import os
import shutil
from training.exception import DataIngestionError, handle_exception
from training.custom_logging import info_logger, error_logger
from training.entity.config_entity import DataIngestionConfig
from training.configuration_manager.configuration import ConfigManager


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        """
        Initializes the DataIngestion class with the provided configuration.
        """
        self.config = config

    def save_data(self):
        """
        Saves the data from the source directory to the target directory.
        Ensures that the data is copied only if the target directory is empty or doesn't exist.
        Writes the status of the operation to the STATUS_FILE.
        """
        try:
            status = None
            # Ensure the target directory exists
            if not os.path.exists(self.config.data_dir):
                info_logger.info(f"Target directory '{self.config.data_dir}' does not exist. Creating and copying data...")
                shutil.copytree(self.config.source, self.config.data_dir)
                status = True

            elif not os.listdir(self.config.data_dir):
                info_logger.info(f"Target directory '{self.config.data_dir}' exists but is empty. Proceeding with data copy...")
                shutil.copytree(self.config.source, self.config.data_dir)
                status = True

            else:
                info_logger.info(f"Target directory '{self.config.data_dir}' exists and is not empty. No action taken.")
                status = True  # Assuming no further action is a valid outcome

            # Write the status to the status file
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Data Ingestion status: {status}")

            info_logger.info(f"Data Ingestion completed with status: {status}")

        except Exception as e:
            # Write failure status to the status file
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Data Ingestion status: False")
            error_logger.error("An error occurred during data ingestion.")
            handle_exception(e, DataIngestionError)


# To test the component
if __name__ == "__main__":
    try:
        info_logger.info("Starting Data Ingestion process...")
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()

        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.save_data()
        info_logger.info("Data Ingestion process completed successfully.")
    except Exception as e:
        error_logger.error("An error occurred while testing the Data Ingestion process.")
        handle_exception(e, DataIngestionError)
