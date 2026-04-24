# ============================================================
# config.py — Configuration constants
# AmmoniteID v1.0
# ============================================================

import os
from pathlib import Path

# Detect if running on Render or locally
IS_RENDER = os.getenv('RENDER') is not None

# Base directory
BASE_DIR = Path(__file__).parent

# Model file paths (same location locally and on Render)
MODEL_PATH = BASE_DIR / 'ammonite_model_v1.keras'
CLASS_INFO_PATH = BASE_DIR / 'class_info.json'

# Storage directories - different paths for Render vs local
if IS_RENDER:
    UPLOAD_DIR = Path('/tmp/uploads')
    REVIEW_DIR = Path('/tmp/review_queue')
else:
    UPLOAD_DIR = BASE_DIR / 'uploads'
    REVIEW_DIR = BASE_DIR / 'review_queue'

UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
REVIEW_DIR.mkdir(exist_ok=True, parents=True)

# Upload limits
MAX_PHOTOS  = 3
MAX_FILE_MB = 10

# Version info
MODEL_VERSION = "v1.0"
APP_VERSION   = "1.0.0"
