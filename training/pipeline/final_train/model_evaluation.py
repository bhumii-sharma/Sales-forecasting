<<<<<<< HEAD
from training.configuration_manager.configuration import ConfigurationManager
from training.components.final_train.model_evaluation import ModelEvaluation
from training.custom_logging import info_logger
import sys

PIPELINE = "Final Model Evaluation Pipeline"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        #Load the data ingestion configuration object
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        model_evaluation = ModelEvaluation(config=model_evaluation_config)

        # Loading the test data for Model Evaluation
        X_test, y_test = model_evaluation.load_test_data()

        # Loading the final model
        final_model = model_evaluation.load_final_model()

        # Evluating the final_model
        model_evaluation.evaluate_final_model(final_model,X_test,y_test)

        


if __name__ == "__main__":  

        info_logger.info(f">>>>> {PIPELINE} started <<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")
=======
import joblib
import numpy as np
import json

class ModelEvaluation:
    def _init_(self, config):
        self.config = config

    def load_test_data(self):
        test_data = np.load(self.config.test_data_path)
        return test_data['X_test'], test_data['y_test']

    def load_model(self):
        # Load the trained model
        return joblib.load(self.config.model_path)

    def evaluate_model(self):
        # Load test data and model
        X_test, y_test = self.load_test_data()
        model = self.load_model()

        # Predict and calculate metrics
        predictions = model.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)

        # Save evaluation metrics
        metrics = {'MSE': mse}
        with open(self.config.metric_file, 'w') as f:
            json.dump(metrics, f)
        print(f"Evaluation metrics saved to {self.config.metric_file}")
>>>>>>> e30d18399790025029fbb074c579247ae38935d8
