# ============================================================
# main.py — FastAPI web server
# AmmoniteID v1.0
# ============================================================
# Run with: py -3.11 -m uvicorn main:app --reload
# Then open: http://localhost:8000/docs
# Review portal: http://localhost:8000/static/review.html
# Test page: http://localhost:8000/static/test.html
# ============================================================

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

from fastapi import (
    FastAPI, File, UploadFile, HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import List
import uuid
import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from config import (
    UPLOAD_DIR, REVIEW_DIR,
    MAX_PHOTOS, MAX_FILE_MB,
    MODEL_VERSION, APP_VERSION
)
from identifier import identify_from_bytes_list

# ── Create the app ───────────────────────────────────────────
app = FastAPI(
    title="AmmoniteID API",
    description="Ammonite fossil identification backend",
    version=APP_VERSION
)

# ── Add CORS middleware ──────────────────────────────────────
# Must be added before mounting static files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount static files ───────────────────────────────────────
# Serves HTML files at /static/filename.html
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent)),
    name="static"
)

# ── Database setup ───────────────────────────────────────────
# Database location - /tmp on Render, local folder otherwise
if os.getenv('RENDER'):
    DB_PATH = Path('/tmp/ammonite.db')
else:
    DB_PATH = Path(__file__).parent / 'ammonite.db'

def init_db():
    """
    Creates the SQLite database and tables
    if they do not already exist.
    """
    conn = sqlite3.connect(str(DB_PATH))
    c    = conn.cursor()

    # Every identification is logged here
    c.execute('''
        CREATE TABLE IF NOT EXISTS identifications (
            id               TEXT PRIMARY KEY,
            timestamp        TEXT,
            num_photos       INTEGER,
            scenario         TEXT,
            top_family       TEXT,
            family_score     INTEGER,
            top_genus        TEXT,
            formatted_output TEXT,
            raw_result       TEXT
        )
    ''')

    # Images awaiting expert review
    c.execute('''
        CREATE TABLE IF NOT EXISTS review_queue (
            id                TEXT PRIMARY KEY,
            identification_id TEXT,
            timestamp         TEXT,
            ai_family         TEXT,
            ai_genus          TEXT,
            ai_confidence     INTEGER,
            status            TEXT DEFAULT 'pending',
            expert_family     TEXT,
            expert_genus      TEXT,
            expert_notes      TEXT,
            reviewed_at       TEXT,
            reviewed_by       TEXT
        )
    ''')

    conn.commit()
    conn.close()

# Initialize database when server starts
init_db()


# ── Helper functions ─────────────────────────────────────────

def save_identification(
    identification_id: str,
    result: dict,
    num_photos: int
):
    """Saves an identification result to the database."""
    top_genus = (
        result['genus_breakdown'][0]['genus']
        if result['genus_breakdown'] else None
    )

    conn = sqlite3.connect(str(DB_PATH))
    c    = conn.cursor()
    c.execute('''
        INSERT INTO identifications VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        identification_id,
        datetime.utcnow().isoformat(),
        num_photos,
        result['scenario'],
        result.get('top_family'),
        result.get('top_family_score'),
        top_genus,
        result['formatted_output'],
        json.dumps(result)
    ))
    conn.commit()
    conn.close()


def save_to_review_queue(
    identification_id: str,
    result: dict,
    photo_paths: list = None
):
    """
    Saves an identification to the expert review queue.
    Stores paths to the saved photos for display in
    the review portal.
    """
    review_id = str(uuid.uuid4())
    top_genus = (
        result['genus_breakdown'][0]['genus']
        if result['genus_breakdown'] else None
    )
    
    # Convert photo paths list to JSON string
    photo_paths_json = json.dumps(photo_paths) if photo_paths else None

    conn = sqlite3.connect(str(DB_PATH))
    c    = conn.cursor()
    
    # Check if photo_paths column exists, if not add it
    c.execute("PRAGMA table_info(review_queue)")
    columns = [col[1] for col in c.fetchall()]
    if 'photo_paths' not in columns:
        c.execute(
            "ALTER TABLE review_queue ADD COLUMN photo_paths TEXT"
        )
    # Check if it's a non-ammonite
    if result.get('scenario') == 'non_ammonite':
    ai_family = result.get('non_am_display', 'Non-ammonite')
    ai_genus = 'N/A'
    ai_confidence = result.get('non_am_total', 0)
    else:
    ai_family = result.get('top_family')
    ai_genus = top_genus
    ai_confidence = result.get('top_family_score')
    c.execute('''
        INSERT INTO review_queue
        (id, identification_id, timestamp,
         ai_family, ai_genus, ai_confidence, status, photo_paths)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
    review_id,
    identification_id,
    datetime.utcnow().isoformat(),
    result.get('top_family'),          # ← Change these 3 lines
    top_genus,                         # ← 
    result.get('top_family_score'),    # ←
    'pending',
    photo_paths_json
))
    conn.commit()
    conn.close()
    return review_id


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    """Health check — confirms the server is running."""
    return {
        "status":        "running",
        "app":           "AmmoniteID",
        "version":       APP_VERSION,
        "model_version": MODEL_VERSION,
    }


@app.post("/identify")
async def identify(
    photos: List[UploadFile] = File(...)
):
    """
    Main identification endpoint.
    Accepts 1 to 3 photos of the same specimen.
    Returns family, genus breakdown and
    formatted display text.
    """

    # ── Validate number of photos ────────────────────────────
    if len(photos) == 0:
        raise HTTPException(
            status_code=400,
            detail="No photos provided."
        )

    if len(photos) > MAX_PHOTOS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_PHOTOS} photos allowed."
        )

    # ── Read and validate each photo ─────────────────────────
    images_bytes = []
    max_bytes    = MAX_FILE_MB * 1024 * 1024

    for photo in photos:
        if photo.content_type not in (
            'image/jpeg', 'image/png', 'image/jpg'
        ):
            raise HTTPException(
                status_code=400,
                detail=f"{photo.filename} is not a"
                       f" JPG or PNG image."
            )

        contents = await photo.read()

        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"{photo.filename} is too large."
                       f" Maximum {MAX_FILE_MB}MB."
            )

        images_bytes.append(contents)

    # ── Run identification ────────────────────────────────────
    try:
        result = identify_from_bytes_list(
            images_bytes,
            num_photos=len(photos)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Identification failed: {str(e)}"
        )

    # ── Save to database ──────────────────────────────────────
    identification_id = str(uuid.uuid4())
    save_identification(
        identification_id, result, len(photos)
    )

    # ── Save photos to disk for review ───────────────────────
    # Create a folder for this identification
    review_folder = REVIEW_DIR / identification_id
    review_folder.mkdir(parents=True, exist_ok=True)
    
    saved_photo_paths = []
    for idx, img_bytes in enumerate(images_bytes):
        photo_path = review_folder / f"photo_{idx+1}.jpg"
        with open(str(photo_path), 'wb') as f:
            f.write(img_bytes)
        saved_photo_paths.append(str(photo_path))

    # ── Save to review queue with photo paths ────────────────
    review_id = save_to_review_queue(
        identification_id, result, saved_photo_paths
    )

    # ── Return result ─────────────────────────────────────────
    return {
        "identification_id": identification_id,
        "review_id":         review_id,
        "scenario":          result['scenario'],
        "num_photos":        result['num_photos'],
        "top_family":        result.get('top_family'),
        "family_confidence": result.get('top_family_score'),
        "genus_breakdown":   result.get('genus_breakdown'),
        "family_scores":     result.get('family_scores'),
        "non_am_total":      result.get('non_am_total'),
        "non_am_category":   result.get('non_am_category'),
        "non_am_display":    result.get('non_am_display'),
        "formatted_output":  result['formatted_output'],
        "model_version":     MODEL_VERSION,
    }


@app.get("/result/{identification_id}")
def get_result(identification_id: str):
    """
    Retrieves a previously saved identification
    by its ID. Used when the app needs to reload
    a result without rerunning the model.
    """
    conn = sqlite3.connect(str(DB_PATH))
    c    = conn.cursor()
    c.execute(
        "SELECT raw_result FROM identifications"
        " WHERE id=?",
        (identification_id,)
    )
    row = c.fetchone()
    conn.close()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="Identification not found."
        )

    return json.loads(row[0])


@app.get("/queue")
def get_review_queue(status: str = "pending"):
    """
    Returns the expert review queue.
    Filter by status: pending, reviewed,
    ambiguous, discarded.
    
    This endpoint is for the expert review portal.
    """
    conn = sqlite3.connect(str(DB_PATH))
    c    = conn.cursor()
    c.execute(
        "SELECT * FROM review_queue WHERE status=?"
        " ORDER BY timestamp DESC",
        (status,)
    )
    rows    = c.fetchall()
    columns = [d[0] for d in c.description]
    conn.close()

    return {
        "status": status,
        "count":  len(rows),
        "items":  [
            dict(zip(columns, row))
            for row in rows
        ]
    }


@app.post("/queue/{review_id}/update")
def update_review(
    review_id:     str,
    expert_family: str = None,
    expert_genus:  str = None,
    expert_notes:  str = None,
    status:        str = "reviewed",
    reviewed_by:   str = "expert"
):
    """
    Updates a review queue item with expert verdict.
    Status options: reviewed, discarded, ambiguous
    
    Called from the expert review portal when
    an expert submits their correction.
    """
    conn = sqlite3.connect(str(DB_PATH))
    c    = conn.cursor()
    c.execute('''
        UPDATE review_queue
        SET status=?, expert_family=?,
            expert_genus=?, expert_notes=?,
            reviewed_at=?, reviewed_by=?
        WHERE id=?
    ''', (
        status,
        expert_family,
        expert_genus,
        expert_notes,
        datetime.utcnow().isoformat(),
        reviewed_by,
        review_id
    ))
    conn.commit()
    conn.close()

    return {
        "review_id": review_id,
        "status":    status,
        "updated":   True
    }


@app.get("/stats")
def get_stats():
    """
    Returns basic usage statistics.
    Used by the admin panel and review portal.
    """
    conn = sqlite3.connect(str(DB_PATH))
    c    = conn.cursor()

    # Total identifications
    c.execute("SELECT COUNT(*) FROM identifications")
    total = c.fetchone()[0]

    # Scenario breakdown
    c.execute('''
        SELECT scenario, COUNT(*) as count
        FROM identifications
        GROUP BY scenario
        ORDER BY count DESC
    ''')
    scenarios = dict(c.fetchall())

    # Top families identified
    c.execute('''
        SELECT top_family, COUNT(*) as count
        FROM identifications
        WHERE top_family IS NOT NULL
        GROUP BY top_family
        ORDER BY count DESC
    ''')
    families = dict(c.fetchall())

    # Pending reviews
    c.execute('''
        SELECT COUNT(*) FROM review_queue
        WHERE status='pending'
    ''')
    pending_reviews = c.fetchone()[0]

    conn.close()

    return {
        "total_identifications": total,
        "pending_reviews":       pending_reviews,
        "scenarios":             scenarios,
        "top_families":          families,
        "model_version":         MODEL_VERSION,
    }


@app.get("/photo/{identification_id}/{photo_name}")
def get_photo(identification_id: str, photo_name: str):
    """
    Serves a photo from the review queue folder.
    Used by the expert review portal to display images.
    """
    from fastapi.responses import FileResponse
    
    photo_path = REVIEW_DIR / identification_id / photo_name
    
    if not photo_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Photo not found."
        )
    
    return FileResponse(str(photo_path))
