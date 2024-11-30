from training.configuration_manager.configuration import ConfigurationManager
from training.components.final_train.model_training import ModelTraining
from training.custom_logging import info_logger

PIPELINE = "Final Model Training Pipeline"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Log the start of the model training process
            info_logger.info("Initializing the Model Training process...")

            # Load the configuration for model training
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            info_logger.info("Model training configuration loaded successfully.")

            # Initialize the ModelTraining component
            model_trainer = ModelTraining(config=model_trainer_config)
            info_logger.info("ModelTraining component initialized.")

            # Load the transformed train and test data
            info_logger.info("Loading transformed train and test data...")
            X_train, X_test, y_train, y_test, groups_train = model_trainer.load_transformed_data()
            info_logger.info("Transformed data loaded successfully.")

            # Load the best model's identifier from Nested Cross-Validation
            info_logger.info("Loading the best model identifier from Nested Cross-Validation...")
            best_model_count = model_trainer.load_best_model()
            info_logger.info(f"Best model identifier loaded: {best_model_count}")

            # Train the final model using the identified best model and train data
            info_logger.info("Training the final model...")
            final_model = model_trainer.train_final_model(best_model_count, X_train, y_train)
            info_logger.info("Final model training completed successfully.")

            # Save the final trained model
            info_logger.info("Saving the final trained model...")
            model_trainer.save_final_model(final_model)
            info_logger.info("Final model saved successfully.")

        except Exception as e:
            info_logger.error(f"An error occurred in the {PIPELINE}: {e}")
            raise

if __name__ == "__main__":
    try:
        info_logger.info(f">>>>> {PIPELINE} started <<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        info_logger.info(f">>>>> {PIPELINE} completed successfully <<<<")
    except Exception as e:
        info_logger.error(f"Pipeline execution failed: {e}")
