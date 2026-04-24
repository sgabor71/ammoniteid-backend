# ============================================================
# config.py — Backend configuration
# ============================================================
import os

# Detect if running on Render or locally
IS_RENDER = os.getenv('RENDER') is not None

from pathlib import Path

BASE_DIR   = Path(__file__).parent

# On Render, use /tmp for uploads and reviews
# (Render's filesystem is ephemeral but /tmp persists during runtime)
if IS_RENDER:
    UPLOAD_DIR = Path('/tmp/uploads')
    REVIEW_DIR = Path('/tmp/review_queue')
    # Database also goes in /tmp on Render
    # Will be recreated on each deploy - that's fine for testing
else:
    UPLOAD_DIR = BASE_DIR / 'uploads'
    REVIEW_DIR = BASE_DIR / 'review_queue'

UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
REVIEW_DIR.mkdir(exist_ok=True, parents=True)

UPLOAD_DIR.mkdir(exist_ok=True)
REVIEW_DIR.mkdir(exist_ok=True)

IMAGE_SIZE   = 224
MAX_PHOTOS   = 3
MAX_FILE_MB  = 10

FAMILY_LIKELY_THRESHOLD    = 0.75
FAMILY_POSSIBLE_THRESHOLD  = 0.55
GENUS_BEST_MATCH_THRESHOLD = 0.60
GENUS_POSSIBLE_THRESHOLD   = 0.30

FREE_DAILY_LIMIT = 5

# Smart cropping — finds fossil region automatically
# Set to False if results are worse with it on
SMART_CROP = True

MODEL_VERSION = 'v1'
APP_VERSION   = '1.0.0'