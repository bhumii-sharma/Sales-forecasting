from training.configuration_manager.configuration import ConfigurationManager
from training.components.cross_val.cross_val import CrossValidation
from training.custom_logging import info_logger
import gc

PIPELINE = "Cross Validation Training Pipeline"

class CrossValPipeline:

    def __init__(self):
        pass

    def main(self):
        try:
            # Log the start of the pipeline
            info_logger.info(f"Starting the {PIPELINE}...")

            # Load the cross-validation configuration
            config = ConfigurationManager()
            cross_val_config = config.get_cross_val_config()

            # Initialize the CrossValidation component
            cross_val = CrossValidation(config=cross_val_config)

            # Load features, labels, and groups
            info_logger.info("Loading features, labels, and groups...")
            X, y, groups = cross_val.get_data_labels_groups()

            # Split data into training and testing sets
            info_logger.info("Splitting data into training and testing sets...")
            X_train, X_test, y_train, y_test, groups_train = cross_val.train_test_split(X, y, groups)

            # Save train-test splits for final training
            cross_val.save_train_test_data_for_final_train(X_train, X_test, y_train, y_test, groups_train)

            # Initialize cross-validation loop
            cross_val_loop = cross_val.initialize_cross_val_loop()

            # Perform cross-validation
            count = 1
            for train_idx, val_idx in cross_val_loop.split(X_train, y_train, groups=groups_train):
                info_logger.info(f"Processing fold {count}...")

                # Prepare training data for the current fold
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                groups_fold_train = groups_train[train_idx]

                # Prepare validation data for the current fold
                X_fold_val = X_train[val_idx]
                y_fold_val = y_train[val_idx]

                # Start training for the current fold
                cross_val.start_training_loop(
                    fold_number=count,
                    X_train=X_fold_train,
                    y_train=y_fold_train,
                    groups_train=groups_fold_train,
                    X_val=X_fold_val,
                    y_val=y_fold_val
                )

                # Clean up to free memory
                del X_fold_train, y_fold_train, groups_fold_train, X_fold_val, y_fold_val
                gc.collect()
                info_logger.info(f"Completed fold {count}.")
                count += 1

            info_logger.info(f"{PIPELINE} completed successfully.")

        except Exception as e:
            info_logger.error(f"An error occurred in the {PIPELINE}: {e}")
            raise

if __name__ == "__main__":
    try:
        info_logger.info(f">>>>> {PIPELINE} started <<<<")
        obj = CrossValPipeline()
        obj.main()
        info_logger.info(f">>>>> {PIPELINE} completed <<<<")
    except Exception as e:
        info_logger.error(f"Pipeline failed: {e}")
