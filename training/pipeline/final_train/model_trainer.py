import joblib
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import logging
from training.configuration_manager.configuration import ConfigurationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def load_train_test_data(self):
        train_data = np.load(self.config.train_data_path)
        test_data = np.load(self.config.test_data_path)
        return (
            train_data['X_train'], train_data['y_train'],
            test_data['X_test'], test_data['y_test']
        )

    def train_and_save_model(self):
        # Load train and test data
        X_train, y_train, X_test, y_test = self.load_train_test_data()

        # Initialize model with hyperparameters
        model = RandomForestRegressor(**self.config.hyperparameters)

        # Train the model
        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, self.config.final_model_name)
        print(f"Final model saved to {self.config.final_model_name}")

        # Evaluate the model
        self.evaluate_and_save_metrics(model, X_test, y_test)

    def evaluate_and_save_metrics(self, model, X_test, y_test):
        # Predict on test data
        predictions = model.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)

        # Save metrics
        metrics = {'MSE': mse}
        with open(self.config.metric_file_name_rf, 'w') as f:
            json.dump(metrics, f)
        print(f"Metrics saved to {self.config.metric_file_name_rf}")

  

PIPELINE = "Model Trainer Pipeline"

def main():
    try:
        logger.info(f"Starting {PIPELINE}...")

        # Initialize the configuration manager
        config_manager = ConfigurationManager()

        # Get the model training configuration
        model_trainer_config = config_manager.get_model_trainer_config()

        # Initialize the ModelTrainer class with the configuration
        model_trainer = ModelTrainer(config=model_trainer_config)

        # Train the model and save it along with metrics
        logger.info("Training the model...")
        model_trainer.train_and_save_model()

        logger.info(f"{PIPELINE} completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred in {PIPELINE}: {str(e)}")

if __name__ == "__main__":
    main()
