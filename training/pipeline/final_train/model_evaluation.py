from training.configuration_manager.configuration import ConfigurationManager
from training.components.final_train.model_evaluation import ModelEvaluation
from training.custom_logging import info_logger

PIPELINE = "Final Model Evaluation Pipeline"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Log the start of the model evaluation process
            info_logger.info("Initializing the Model Evaluation process...")

            # Load the configuration for model evaluation
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            info_logger.info("Model evaluation configuration loaded successfully.")

            # Initialize the ModelEvaluation component
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            info_logger.info("ModelEvaluation component initialized.")

            # Load the test data
            info_logger.info("Loading the test data for evaluation...")
            X_test, y_test = model_evaluation.load_test_data()
            info_logger.info("Test data loaded successfully.")

            # Load the final model
            info_logger.info("Loading the final trained model...")
            final_model = model_evaluation.load_final_model()
            info_logger.info("Final model loaded successfully.")

            # Evaluate the final model
            info_logger.info("Evaluating the final model on test data...")
            evaluation_metrics = model_evaluation.evaluate_final_model(final_model, X_test, y_test)
            info_logger.info(f"Model evaluation completed successfully. Metrics: {evaluation_metrics}")

        except Exception as e:
            info_logger.error(f"An error occurred in the {PIPELINE}: {e}")
            raise

if __name__ == "__main__":
    try:
        info_logger.info(f">>>>> {PIPELINE} started <<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        info_logger.info(f">>>>> {PIPELINE} completed successfully <<<<")
    except Exception as e:
        info_logger.error(f"Pipeline execution failed: {e}")
