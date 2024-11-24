import numpy as np
import joblib
import boto3
import sys
from deployment.exception import PredictionError,handle_exception

class Prediction:

    def _init_(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'fabric-artifact-storage'  # Your S3 bucket name
        self.model_key = 'final_model.joblib'  # S3 key for the final model file
        self.local_model_path = '/tmp/final_model.joblib'  # Temporary local path to save the model

    def download_model_from_s3(self):

        # Download the model from S3 to a local path
        self.s3_client.download_file(self.bucket_name, self.model_key, self.local_model_path)
        print(f"Model downloaded from S3 and saved to {self.local_model_path}")


    def predict(self, transformed_features):
        try:
            # Download the model from S3 before prediction
            self.download_model_from_s3()

            # Load the model from the local path
            model = joblib.load(self.local_model_path)

            # Perform prediction
            prediction = model.predict(transformed_features)[0]
            print(prediction)

            # Mapping prediction to category
            category_mapping = {1: 'corduroy', 2: 'denim'}
            predicted_label = category_mapping.get(prediction, "Unknown")

            return predicted_label

        except Exception as e:
            handle_exception(e, PredictionError)