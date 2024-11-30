from deployment.components.feature_engineering import FeatureEngineering
from deployment.components.prediction import Prediction
import sys
import os
import json

class PredictionPipeline:
    def _init_(self):
        pass

    def predict_label(self,extracted_features):

        feature_engineering = FeatureEngineering()
        transformed_features = feature_engineering.transform_features(extracted_features)

        prediction = Prediction()
        predicted_label = prediction.predict(transformed_features)
        
        return predicted_label
