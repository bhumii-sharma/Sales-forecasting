from training.configuration_manager.configuration import ConfigurationManager
from training.components.final_train.feature_engineering import FeatureEngineering
from training.custom_logging import info_logger

PIPELINE = "Feature Engineering Training Pipeline"

class FeatureEngineeringTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Log the start of the feature engineering process
            info_logger.info("Initializing the Feature Engineering process...")

            # Load the configuration for feature engineering
            config = ConfigurationManager()
            feature_engineering_config = config.get_feature_engineering_config()
            info_logger.info("Loaded feature engineering configuration.")

            # Initialize the FeatureEngineering component
            feature_engineering = FeatureEngineering(config=feature_engineering_config)
            info_logger.info("FeatureEngineering component initialized.")

            # Transform the data
            info_logger.info("Transforming the training and testing data...")
            X_train, X_test, y_train, y_test, groups_train = feature_engineering.transform_features()
            info_logger.info("Data transformation completed successfully.")

            # Save the transformed data
            info_logger.info("Saving the transformed data if it does not already exist...")
            feature_engineering.save_transformed_data(X_train, X_test, y_train, y_test, groups_train)
            info_logger.info("Transformed data saved successfully.")

        except Exception as e:
            info_logger.error(f"An error occurred in the {PIPELINE}: {e}")
            raise

if __name__ == "__main__":
    try:
        info_logger.info(f">>>>> {PIPELINE} started <<<<")
        obj = FeatureEngineeringTrainingPipeline()
        obj.main()
        info_logger.info(f">>>>> {PIPELINE} completed successfully <<<<")
    except Exception as e:
        info_logger.error(f"Pipeline execution failed: {e}")
