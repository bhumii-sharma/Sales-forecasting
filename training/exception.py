from training.custom_logging import error_logger


# Custom exception handler function
def handle_exception(error, error_type):
    """
    Logs the error with details to the log file and prints a formatted message to the console.
    
    Args:
        error (Exception): The original exception instance.
        error_type (PipelineError): The specific pipeline error type (e.g., DataIngestionError).
    """
    # Log the complete stack trace and details to the log file
    error_logger.error("Exception occurred", exc_info=True)

    error_logger.error("\n\n")  # To leave a few blank lines

    # Print only the formatted message to the console
    print(f"{error_type._name}: {error}")



# Base custom exception
class PipelineError(Exception):
    """Base class for custom pipeline exceptions"""
    def _init_(self, original_exception):
        super()._init_(str(original_exception))
        self.original_exception = original_exception

# Specific pipeline exceptions
class DataIngestionError(PipelineError):
    pass

class DataValidationError(PipelineError):
    pass

class CrossValError(PipelineError):
    pass

class FeatureEngineeringError(PipelineError):
    pass

class ModelTrainingError(PipelineError):
    pass

class ModelEvaluationError(PipelineError):
    pass



def my_func(value):
    # Example function logic
    if value < 0:
        raise ValueError("Value cannot be negative")
    print(f"Value is: {value}")

if __name__ == "__main__":
    try:
        my_func(23)  # Replace with the value you want to test
    except Exception as e:
        handle_exception(e, DataIngestionError)

# if __name__ == "__main__":
#     try:
#         my_func(23)
#     except Exception as e:
#         handle_exception(e, DataIngestionError)
