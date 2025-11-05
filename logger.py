import logging
from datetime import datetime
import os

# Set up logging with UTF-8 encoding for Windows compatibility
def setup_logger():
    # Ensure logs directory exists
    os.makedirs('./logs', exist_ok=True)
    
    # File path for the log file
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = f'./logs/dmcc_rag_dev_{current_time}.log'

    logger = logging.getLogger('MyLogger')
    logger.setLevel(logging.INFO)

    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)
    return logger
