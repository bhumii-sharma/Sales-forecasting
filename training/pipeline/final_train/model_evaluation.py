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