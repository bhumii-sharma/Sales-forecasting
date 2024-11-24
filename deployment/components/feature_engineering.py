import numpy as np
import joblib
import boto3
import os
from deployment.exception import FeatureEngineeringError,handle_exception


class FeatureEngineering:
    
    def _init_(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'fabric-artifact-storage'
        self.pipeline_key = 'pipeline.joblib'
        self.local_pipeline_path = '/tmp/pipeline.joblib'

    def download_pipeline_from_s3(self):

        # Download the joblib pipeline from S3 to a local path
        self.s3_client.download_file(self.bucket_name, self.pipeline_key, self.local_pipeline_path)
        print(f"Pipeline downloaded from S3 and saved to {self.local_pipeline_path}")


    def transform_features(self, extracted_features):
        
        try:
            # Download pipeline from S3 before transformation
            self.download_pipeline_from_s3()

            # Load the pipeline from the local path after downloading
            transform_pipeline = joblib.load(self.local_pipeline_path)
            extracted_features = extracted_features.reshape((1, -1))
            transformed_features = transform_pipeline.transform(extracted_features)
            return transformed_features
        except Exception as e:
            handle_exception(e, FeatureEngineeringError)