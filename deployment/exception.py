from deployment.custom_logging import error_logger

# Custom exception handler function
def handle_exception(error, error_type):
    """
    Logs the error with details to the log file and prints a formatted message to the console.
    
    Args:
        error (Exception): The original exception instance.
        error_type (PipelineError): The specific pipeline error type (e.g., PredictionError).
    """
    # Log the complete stack trace and details to the log file
    error_logger.error("Exception occurred", exc_info=True)

    error_logger.error("\n\n")  # Leave blank lines for better readability in logs

    # Print only the formatted message to the console
    print(f"{error_type.__name__}: {error.__class__.__name__}: {error}")


# Base custom exception
class PipelineError(Exception):
    """Base class for custom pipeline exceptions"""
    def __init__(self, original_exception):
        super().__init__(str(original_exception))
        self.original_exception = original_exception

# Specific pipeline exceptions related to Sales Forecasting

class DataPreprocessingError(PipelineError):
    """Raised when an error occurs during data preprocessing"""
    pass

class FeatureEngineeringError(PipelineError):
    """Raised when an error occurs during feature engineering"""
    pass

class ModelTrainingError(PipelineError):
    """Raised when an error occurs during model training"""
    pass

class PredictionError(PipelineError):
    """Raised when an error occurs during prediction"""
    pass

if __name__ == "__main__":
    try:
        # Simulating a part of the sales forecasting pipeline
        # Replace with actual function calls like model training or prediction
        raise ValueError("Mock error during prediction")  # Example error
    except DataPreprocessingError as e:
        handle_exception(e, DataPreprocessingError)
    except FeatureEngineeringError as e:
        handle_exception(e, FeatureEngineeringError)
    except ModelTrainingError as e:
        handle_exception(e, ModelTrainingError)
    except PredictionError as e:
        handle_exception(e, PredictionError)
    except Exception as e:
        # Handle any unexpected errors
        print(f"Unexpected error: {str(e)}")
        error_logger.error("Unexpected error: ", exc_info=True)
