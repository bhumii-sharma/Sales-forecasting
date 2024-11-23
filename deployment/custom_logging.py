import logging
import sys
import os
from datetime import datetime

# Create logs directory if it doesn't exist
if not os.path.exists("deployment/logs"):
    os.makedirs("deployment/logs")


def setup_info_logger():
    """
    Sets up a logger that writes only INFO level logs to a dynamically named log file based on the current date and time.
    This logger can capture normal operations like model prediction, data processing steps, etc.
    """
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    
    # Create logs directory for info logs if it doesn't exist
    if not os.path.exists("deployment/logs/info_logs"):
        os.makedirs("deployment/logs/info_logs")
    
    log_filepath = os.path.join("deployment/logs/info_logs", log_filename)
    
    # Create and configure the info logger
    info_logger = logging.getLogger("info_logger")
    info_logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)

    # Log format
    formatter_info = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
    file_handler.setFormatter(formatter_info)
    
    info_logger.addHandler(file_handler)
    
    return info_logger

def setup_error_logger():
    """
    Sets up a logger that writes ERROR level logs to a dynamically named log file based on the current date and time.
    This logger can capture issues such as failed predictions, data errors, or model issues.
    """
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    
    # Create logs directory for error logs if it doesn't exist
    if not os.path.exists("deployment/logs/error_logs"):
        os.makedirs("deployment/logs/error_logs")
    
    log_filepath = os.path.join("deployment/logs/error_logs", log_filename)
    
    # Create and configure the error logger
    error_logger = logging.getLogger("error_logger")
    error_logger.setLevel(logging.ERROR)
    
    error_handler = logging.FileHandler(log_filepath)
    error_handler.setLevel(logging.ERROR)
    
    # Log format
    formatter_error = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s\n"
    )
    error_handler.setFormatter(formatter_error)
    
    error_logger.addHandler(error_handler)
    
    return error_logger

# Create loggers for info and error logging
info_logger = setup_info_logger()
error_logger = setup_error_logger()

if __name__ == "__main__":
    # Example info log for forecasting
    info_logger.info("Sales forecasting model initialized.")
    info_logger.info("Model prediction for product X completed. Predicted sales: 500 units.")
    
    # Example error log for sales forecasting issues
    error_logger.error("Error encountered while processing the sales data for product Y.")
    error_logger.error("Failed to load sales forecasting model. Check model configuration.")
