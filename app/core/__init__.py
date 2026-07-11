from .app_info import APP_NAME, VERSION, AUTHOR, COMPANY
from .app_paths import app_paths
from .config import config
from .constants import UI, DATABASE, FILES, IMAGES, AI, CALIBRATION
from .logger import setup_logging

# Initialize logging when the core package is imported
setup_logging()
