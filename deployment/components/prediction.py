import joblib
import boto3
import os
from deployment.exception import PredictionError, handle_exception
from deployment.custom_logging import info_logger, error_logger  # Assuming you have a logging module

class Prediction:

    def __init__(self):
        # Initialize the S3 client and paths
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'sales_forecast-artifact-storage'  # Your S3 bucket name
        self.model_key = 'final_model.joblib'  # S3 key for the final model file
        self.local_model_path = '/tmp/final_model.joblib'  # Temporary local path to save the model

    def download_model_from_s3(self):
        """
        Download the trained model from S3 to a local path.
        """
        try:
            info_logger.info(f"Downloading model from S3 bucket: {self.bucket_name}, key: {self.model_key}")
            self.s3_client.download_file(self.bucket_name, self.model_key, self.local_model_path)
            info_logger.info(f"Model downloaded successfully to {self.local_model_path}")
        except Exception as e:
            handle_exception(e, PredictionError)
            error_logger.error(f"Error downloading model from S3: {str(e)}")

    def predict(self, transformed_features):
        """
        Predict the value of Item_Outlet_Sales based on the given set of transformed features.
        """
        try:
            # Ensure the transformed_features is not empty or None
            if transformed_features is None or len(transformed_features) == 0:
                raise ValueError("Transformed features are empty or None.")

            # Download the model from S3 before making a prediction
            self.download_model_from_s3()

            # Load the model from the local path
            model = joblib.load(self.local_model_path)
            info_logger.info("Model loaded successfully.")

            # Perform prediction
            prediction = model.predict(transformed_features)[0]  # Assuming it's a single sample prediction
            info_logger.info(f"Prediction result: {prediction}")

            # Return the predicted Item_Outlet_Sales value (numeric)
            return prediction

        except Exception as e:
            handle_exception(e, PredictionError)
            error_logger.error(f"Error during prediction: {str(e)}")
            return None  # In case of failure, return None or handle as per requirements
