from training.configuration_manager.configuration import ConfigurationManager
from training.components.final_train.feature_engineering import FeatureEngineering
from training.custom_logging import info_logger
import sys

PIPELINE = "Feature Engineering Training Pipeline"

class FeatureEngineeringTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
<<<<<<< HEAD
        config = ConfigurationManager()
        feature_engineering_config = config.get_feature_engineering_config()

=======
        # Initialize configuration manager and load feature engineering configuration
        config = ConfigurationManager()
        feature_engineering_config = config.get_feature_engineering_config()

        # Initialize feature engineering with the configuration
>>>>>>> e30d18399790025029fbb074c579247ae38935d8
        feature_engineering = FeatureEngineering(config=feature_engineering_config)
        
        # Transforming the data
        X_train, X_test, y_train, y_test, groups_train = feature_engineering.transform_features()

<<<<<<< HEAD
        # Saving the transformed data only if if does not exist
=======
        # Save the transformed data if it does not exist
>>>>>>> e30d18399790025029fbb074c579247ae38935d8
        feature_engineering.save_transformed_data(X_train, X_test, y_train, y_test, groups_train)
        
if __name__ == "__main__":
    info_logger.info(f">>>>> {PIPELINE} started <<<<")
    obj = FeatureEngineeringTrainingPipeline()
<<<<<<< HEAD
    obj.main()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")
=======
    # obj.main()
    info_logger.info(f">>>>>>> {PIPELINE} completed <<<<<<<<")
>>>>>>> e30d18399790025029fbb074c579247ae38935d8
