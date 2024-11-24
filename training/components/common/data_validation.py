import os
import pandas as pd
from training.entity.config_entity import DataValidationConfig
from training.configuration_manager.configuration import ConfigurationManager
from training.exception import DataValidationError, handle_exception
from training.custom_logging import info_logger, error_logger


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_file_exists(self):
        """
        Validates if the source file exists.
        """
        if not os.path.exists(self.config.source):
            raise FileNotFoundError(f"Source file not found at {self.config.source}")

    def validate_columns(self, df: pd.DataFrame):
        """
        Validates if all required columns are present in the dataset.
        """
        required_columns = ["Item_Identifier", "Item_Weight", "Item_Fat_Content", 
                            "Item_Visibility", "Item_Type", "Item_MRP", 
                            "Outlet_Identifier", "Outlet_Establishment_Year", 
                            "Outlet_Size", "Outlet_Location_Type", 
                            "Outlet_Type", "Item_Outlet_Sales"]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        info_logger.info(f"All required columns are present.")

    def validate_missing_values(self, df: pd.DataFrame):
        """
        Validates if there are missing values in critical columns.
        """
        if df.isnull().sum().any():
            info_logger.warning("Dataset contains missing values.")
        else:
            info_logger.info("No missing values detected.")

    def validate_data_types(self, df: pd.DataFrame):
        """
        Validates the data types of critical columns.
        """
        expected_types = {
            "Item_Identifier": str,
            "Item_Weight": float,
            "Item_Fat_Content": str,
            "Item_Visibility": float,
            "Item_Type": str,
            "Item_MRP": float,
            "Outlet_Identifier": str,
            "Outlet_Establishment_Year": int,
            "Outlet_Size": str,
            "Outlet_Location_Type": str,
            "Outlet_Type": str,
            "Item_Outlet_Sales": float,
        }

        mismatched_columns = []
        for column, expected_type in expected_types.items():
            if not pd.api.types.is_dtype_equal(df[column].dtype, expected_type):
                mismatched_columns.append(column)
        if mismatched_columns:
            raise TypeError(f"Data types mismatch for columns: {mismatched_columns}")
        info_logger.info("All data types match the expected schema.")

    def validate_data(self):
        """
        Orchestrates the data validation process.
        """
        try:
            # Check if the file exists
            self.validate_file_exists()

            # Load the dataset
            df = pd.read_csv(self.config.source)
            info_logger.info(f"Loaded data from {self.config.source} with shape {df.shape}.")

            # Perform validations
            self.validate_columns(df)
            self.validate_missing_values(df)
            self.validate_data_types(df)

            # Write validation status to the STATUS_FILE
            with open(self.config.STATUS_FILE, "w") as status_file:
                status_file.write("Data Validation completed: All checks passed successfully.")
            info_logger.info("Data Validation completed successfully.")

        except Exception as e:
            # Log and write failure status
            with open(self.config.STATUS_FILE, "w") as status_file:
                status_file.write(f"Data Validation failed due to an error: {str(e)}")
            handle_exception(e, DataValidationError)


# To check the component
if __name__ == "__main__":
    config_manager = ConfigurationManager()
    data_validation_config = config_manager.get_data_validation_config()

    data_validation = DataValidation(data_validation_config)
    data_validation.validate_data()
