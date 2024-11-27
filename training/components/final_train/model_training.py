import os
import sys
import json
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from training.exception import ModelTrainingError, handle_exception
from training.custom_logging import info_logger, error_logger
from training.entity.config_entity import ModelTrainerConfig
from training.configuration_manager.configuration import ConfigManager

class ModelTraining:
    def _init_(self, config: ModelTrainerConfig) -> None:
        self.config = config

    def load_transformed_data(self):
        """
        Loads the transformed training and test data for final training.
        Returns:
            X_train (np.ndarray), X_test (np.ndarray), y_train (np.ndarray), y_test (np.ndarray), groups_train (np.ndarray)
        """
        try:
            info_logger.info("Loading the train and test data for Final Training...")

            # Paths to the train and test .npz files
            train_data_path = os.path.join(self.config.train_data_path, "Train.npz")
            test_data_path = os.path.join(self.config.test_data_path, "Test.npz")

            # Loading the train and test .npz files
            train_data = np.load(train_data_path, allow_pickle=True)
            test_data = np.load(test_data_path, allow_pickle=True)

            # Access the arrays stored inside the .npz files
            X_train, y_train, groups_train = train_data["X_train"], train_data["y_train"], train_data["groups_train"]
            X_test, y_test = test_data["X_test"], test_data["y_test"]

            info_logger.info("Successfully Loaded the train data and test data for Final Training...")
            return X_train, X_test, y_train, y_test, groups_train
        except Exception as e:
            handle_exception(e, ModelTrainingError)

    def load_best_model(self):
        """
        Loads the best model based on the highest macro F1 score from the saved models.
        Returns:
            best_model (RandomForestClassifier): The best model loaded from the directory
            loop_count (int): The number of iterations through the models.
        """
        try:
            best_f1_score = -1  # Initialize with a very low value
            best_model_file = None
            loop_count = 0

            directory_path = self.config.metric_file_name_rf
            # Loop through all files in the directory
            for filename in os.listdir(directory_path):
                if filename.endswith('.json'):  # Assuming metrics are stored in .json files
                    file_path = os.path.join(directory_path, filename)
                    with open(file_path, 'r') as f:
                        metrics = json.load(f)  # Load the metrics from the JSON file

                    # Assuming 'macro avg' key contains the macro F1-score
                    macro_avg_f1_score = metrics['macro avg']['f1-score']
                    
                    loop_count += 1  # Increment the loop count

                    # Compare the current model's F1-score with the best one found so far
                    if macro_avg_f1_score > best_f1_score:
                        best_f1_score = macro_avg_f1_score
                        best_model_file = filename

            info_logger.info(f"Best model found with F1-score: {best_f1_score}")
            return loop_count
        except Exception as e:
            handle_exception(e, ModelTrainingError)

    def select_best_model(self, X_test, y_test):
        """
        Select the best performing model based on F1-score from saved models.
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): True labels for test data
        Returns:
            best_model (RandomForestClassifier): The best model based on F1-score
        """
        info_logger.info("Selecting the best model for Final Training...")

        best_f1 = -1  # Initialize the best F1 score
        best_model = None  # Initialize the best model

        # Get the directory where models are stored
        model_dir = self.config.best_cross_val_models_rf

        # List all joblib files in the directory
        model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.joblib')]

        # Loop through all model file paths
        for model_path in model_files:
            # Load the model using joblib
            model = load(model_path)
            
            # Make predictions on the validation data
            info_logger.info(f"shape of X_test: {X_test.shape}")
            info_logger.info(f"Type of X_test: {type(X_test)}")
            info_logger.info(f"Dtype of X_test: {X_test.dtype}")
            y_pred = model.predict(X_test)
            
            # Compute the macro average F1 score
            f1 = f1_score(y_test, y_pred, average='macro')
            
            # If the current model has the highest F1 score, update the best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = model

        info_logger.info(f"Successfully selected the best model for Final Training... with macro_avg F1-score: {best_f1}")
        return best_model.get_params()

    def train_final_model(self, loop_count, X_train, y_train):
        """
        Trains the final model using the best hyperparameters.
        Args:
            loop_count (int): The number of iterations for model selection
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        Returns:
            final_model (RandomForestClassifier): The trained RandomForest model
        """
        try:
            info_logger.info("Training the final model...")

            # Initialize a new RandomForestClassifier with the best hyperparameters
            hyperparams_file = f'best_params_rf_{loop_count}.json'
            hyperparams_directory = self.config.best_model_params_rf
        
            # Construct the full path to the hyperparameters file
            hyperparams_file_path = os.path.join(hyperparams_directory, hyperparams_file)
            
            # Load the hyperparameters from the JSON file
            with open(hyperparams_file_path, 'r') as f:
                hyperparams = json.load(f)

            hyperparams = self.filter_hyperparams(hyperparams)

            info_logger.info(f"Using hyperparameters: {hyperparams}")

            final_model = RandomForestClassifier(**hyperparams)
            final_model.fit(X_train, y_train)

            info_logger.info("Successfully trained the final model.")
            return final_model
        except Exception as e:
            handle_exception(e, ModelTrainingError)

    def save_final_model(self, final_model):
        """
        Saves the final trained model to disk.
        Args:
            final_model (RandomForestClassifier): The trained model to be saved
        """
        try:
            info_logger.info("Saving the final model...")

            # Save the final model using joblib
            dump(final_model, self.config.final_model_name)

            # Optionally, save the status of the model training
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Final Model status: {True}")

            info_logger.info("Successfully saved the final model.")
        except Exception as e:
            handle_exception(e, ModelTrainingError)

    @staticmethod
    def filter_hyperparams(params):
        """
        Filters the hyperparameters specific to the RandomForestClassifier.
        Args:
            params (dict): Hyperparameters dictionary
        Returns:
            dict: Filtered hyperparameters for RandomForestClassifier
        """
        # Extract only the parameters related to the classifier (RandomForestClassifier)
        rf_hyperparams = {key.replace('classifier_', ''): value for key, value in params.items() if key.startswith('classifier_')}
        return rf_hyperparams