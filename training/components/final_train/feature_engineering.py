import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from training.exception import FeatureEngineeringError, handle_exception
from training.custom_logging import info_logger, error_logger
from training.entity.config_entity import FeatureEngineeringConfig
from training.configuration_manager.configuration import ConfigurationManager

class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config



    def transform_features(self):
        """
        Transform features using preprocessing pipelines.
        """
        try:
            # Load and preprocess data
            df = self.load_and_preprocess_data()

            # Define feature types
            numerical_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
            categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

            if "target" in df.columns:
                y = df["target"]
                df.drop(columns=["target"], inplace=True)
            else:
                y = None

            X = df

            # Define transformations
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine transformations into a single pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )

            # Add PCA if needed
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('pca', PCA(n_components=None))  # Adjust n_components as required.
            ])

            # Fit the pipeline
            X_transformed = pipeline.fit_transform(X)
            info_logger.info(f"Feature transformation completed. Transformed shape: {X_transformed.shape}")

            # Save the pipeline
            pipeline_path = os.path.join(self.config.root_dir, "pipeline.joblib")
            joblib.dump(pipeline, pipeline_path)
            info_logger.info(f"Pipeline saved at {pipeline_path}")

            return X_transformed, y
        except Exception as e:
            handle_exception(e, FeatureEngineeringError)

    def save_transformed_data(self, X_transformed, y):
        """
        Save transformed data to .npz files.
        """
        try:
            transformed_data_path = self.config.root_dir
            np.savez(os.path.join(transformed_data_path, 'Transformed_Data.npz'), 
                     X_transformed=X_transformed, y=y)
            info_logger.info("Transformed data saved successfully.")
        except Exception as e:
            handle_exception(e, FeatureEngineeringError)
