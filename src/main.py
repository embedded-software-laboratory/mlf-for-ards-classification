import framework
import logging
import os 
from datetime import datetime

# Define log directory for storing log files
LOG_DIR = "../Data/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create log filename with timestamp to ensure unique log files
log_filename = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Log framework startup
    logger.info("=" * 80)
    logger.info("Starting ML Framework")
    logger.info(f"Logs will be saved to: {log_filename}")
    logger.info("=" * 80)
    
    try:
        # Initialize and run the framework
        ml_framework = framework.Framework()
        
        # Execute the main workflow
        ml_framework.run()
        
        logger.info("=" * 80)
        logger.info("Framework execution completed successfully")
        logger.info("=" * 80)
        
    except Exception as e:
        # Log any errors that occur during execution
        logger.error("=" * 80)
        logger.error(f"Error occurred during framework execution: {str(e)}", exc_info=True)
        logger.error("=" * 80)
        raise
