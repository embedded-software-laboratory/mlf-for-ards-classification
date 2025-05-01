import framework
import logging
import os 
from datetime import datetime
LOG_DIR = "../Data/logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


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
    framework = framework.Framework()
    framework.run()
