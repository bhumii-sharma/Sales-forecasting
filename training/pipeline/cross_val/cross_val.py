from training.configuration_manager.configuration import ConfigurationManager
from training.components.cross_val.cross_val import CrossVal  # Import CrossVal instead of NestedCrossVal
from training.custom_logging import info_logger
from training.exception import CrossValError, handle_exception
import sys
import gc
import numpy as np

PIPELINE = "Cross Validation Training Pipeline"  # Updated pipeline name

class CrossValPipeline:

    def __init__(self):
        pass

    def main(self):
        try:
            # Load the data ingestion configuration object
            config = ConfigurationManager()
            cross_val_config = config.get_cross_val_config()  # Updated to load cross_val config

            # Pass the data ingestion configuration obj to the CrossVal component
            cross_val = CrossVal(config=cross_val_config)

            # Loading the extracted features, labels, and groups
            X, y, groups = cross_val.get_data_labels_groups()

            # Split the data into train and test sets
            X_train, X_test, y_train, y_test, groups_train = cross_val.train_test_split(X, y, groups)

            # Save X_train, X_test, y_train, y_test to be used by final_train
            cross_val.save_train_test_data_for_final_train(X_train, X_test, y_train, y_test, groups_train)

            # Initialize cross-validation loop (no nested loop in this case)
            cross_val_loop = cross_val.initialize_cross_val_loop()  # Updated to use cross_val_loop

            # The following loop runs based on the number of splits in cross_val_loop
            count = 1
            for train_idx, val_idx in cross_val_loop.split(X_train, y_train, groups=groups_train):
                # Train data for the current fold
                X_outer_train = X_train[train_idx]  # Training features
                y_outer_train = y_train[train_idx]  # Training labels
                groups_outer_train = groups_train[train_idx]  # Training groups

                # Validation data for the current fold
                X_outer_val = X_train[val_idx]  # Validation features
                y_outer_val = y_train[val_idx]  # Validation labels

                # Start training for the current fold
                cross_val.start_training_loop(count, X_outer_train, y_outer_train, groups_outer_train, X_outer_val, y_outer_val)
                
                # Clean up to free memory
                del X_outer_train
                del y_outer_train
                del groups_outer_train
                del X_outer_val
                del y_outer_val
                gc.collect()
                
                count += 1

            info_logger.info(f">>>>> {PIPELINE} completed <<<<")
        
        except Exception as e:
            info_logger.error(f"An error occurred during the {PIPELINE}.")
            handle_exception(e)  # You can define the error handling function as needed

if __name__ == "__main__":
    try:
        info_logger.info(f">>>>> {PIPELINE} started <<<<")
        obj = CrossValPipeline()
        obj.main()
    except Exception as e:
        info_logger.error(f"An error occurred during the {PIPELINE}.")
        handle_exception(e)
