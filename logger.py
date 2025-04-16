import os
import sys
import datetime
import logging

def setup_logger(log_dir, experiment_name):
    """Set up logger to track experiments and training progress"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique experiment ID with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{experiment_name}_{timestamp}"
    
    # Create experiment directory
    exp_dir = os.path.join(log_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Configure logger
    log_file = os.path.join(exp_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger()
    
    # Log basic system info
    logger.info(f"Starting experiment: {exp_id}")
    
    return logger, exp_dir
