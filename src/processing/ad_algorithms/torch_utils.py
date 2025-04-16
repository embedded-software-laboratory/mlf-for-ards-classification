
import logging
import os
import random
import torch

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Set the random seed for reproducibility.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed: {seed}")

def check_device() -> str:
    """
    Check if a GPU is available and set the device accordingly.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    return device

def check_directory(directory: str):
    """
    Check if the directory exists, if not, create it.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Directory {directory} created.")
    else:
        logger.info(f"Directory {directory} already exists.")