from training.configuration_manager.configuration import ConfigManager
from training.components.cross_val.cross_val import CrossValidation  # Updated import to match the new class
from training.custom_logging import info_logger
import sys
import gc

PIPELINE = "Cross Validation Training Pipeline"  # Updated name

class CrossValPipeline:

    def __init__(self):
        pass

    def main(self):
        # Load the data ingestion configuration object
        config = ConfigurationManager()
        cross_val_config = config.get_cross_val_config()  # Updated config to use the new one

        # Pass the data ingestion configuration object to the CrossVal component
        cross_val = CrossValidation(config=cross_val_config)

        # Loading the extracted features, labels, and groups
        X, y, groups = cross_val.get_data_labels_groups()

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test, groups_train = cross_val.train_test_split(X, y, groups)

        # Save X_train, X_test, y_train, y_test to be used by final_train
        cross_val.save_train_test_data_for_final_train(X_train, X_test, y_train, y_test, groups_train)

        # Initialize outer loop (here outer_cv can be the cross-validation split)
        outer_cv = cross_val.initialize_outer_loop()

        # Loop over the outer cross-validation folds
        count = 1
        for outer_train_idx, outer_val_idx in outer_cv.split(X_train, y_train, groups=groups_train):
            # Train data for the current fold
            X_outer_train = X_train[outer_train_idx]  # Training features
            y_outer_train = y_train[outer_train_idx]  # Training labels
            groups_outer_train = groups_train[outer_train_idx]  # Training groups

            # Validation data for the current fold
            X_outer_val = X_train[outer_val_idx]  # Validation features
            y_outer_val = y_train[outer_val_idx]  # Validation labels

            # Train the model on the current fold
            cross_val.start_inner_loop(count, X_outer_train, y_outer_train, groups_outer_train, X_outer_val, y_outer_val)

            # Free up memory
            del X_outer_train
            del y_outer_train
            del groups_outer_train
            del X_outer_val
            del y_outer_val
            gc.collect()  # Collect garbage to free memory
            count += 1


if __name__ == "__main__":

    info_logger.info(f">>>>> {PIPELINE} started <<<<")
    obj = CrossValPipeline()  # Updated class name to CrossValPipeline
    obj.main()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<")
