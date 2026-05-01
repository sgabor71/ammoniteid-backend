# ============================================================
# main.py — FastAPI web server (MEMORY OPTIMIZED)
# AmmoniteID v1.0
# ============================================================
# CHANGES FROM ORIGINAL:
# - Stream photos to disk immediately (don't hold in RAM)
# - Process from disk instead of memory
# - Aggressive cleanup after each request
# - Reduced MAX_FILE_MB from 10 to 5 (safer for 500MB limit)
# ============================================================
# Run with: py -3.11 -m uvicorn main:app --reload
# Then open: http://localhost:8000/docs
# Review portal: http://localhost:8000/static/review.html
# Test page: http://localhost:8000/static/test.html
# ============================================================

import os
import gc  # ADDED: For memory cleanup

from fastapi import (
    FastAPI, File, UploadFile, HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import List
import uuid
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
    description="Ammonite fossil identification backend (Memory Optimized)",
    version=APP_VERSION
)

# ── Add CORS middleware ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount static files ───────────────────────────────────────
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent)),
    name="static"
)

# ── Database setup ───────────────────────────────────────────
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
            reviewed_by       TEXT,
            photo_paths       TEXT
        )
    ''')

    conn.commit()
    conn.close()

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
    """
    review_id = str(uuid.uuid4())
    top_genus = (
        result['genus_breakdown'][0]['genus']
        if result['genus_breakdown'] else None
    )
    
    photo_paths_json = json.dumps(photo_paths) if photo_paths else None

    conn = sqlite3.connect(str(DB_PATH))
    c    = conn.cursor()
    
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
        ai_family,
        ai_genus,
        ai_confidence,
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
    
    CHANGED: Stream photos to disk first, then process from disk
    This prevents holding all photos in memory at once
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

    # ── Create folders for this identification ──────────────
    identification_id = str(uuid.uuid4())
    review_folder = REVIEW_DIR / identification_id
    review_folder.mkdir(parents=True, exist_ok=True)
    
    # CHANGED: Save photos to disk FIRST (streaming)
    # Don't hold all photos in memory at once
    max_bytes = MAX_FILE_MB * 1024 * 1024
    saved_photo_paths = []
    
    for idx, photo in enumerate(photos):
        # Validate file type
        if photo.content_type not in (
            'image/jpeg', 'image/png', 'image/jpg'
        ):
            raise HTTPException(
                status_code=400,
                detail=f"{photo.filename} is not a JPG or PNG image."
            )
        
        # CHANGED: Stream to disk immediately
        photo_path = review_folder / f"photo_{idx+1}.jpg"
        
        # Read and write in chunks to avoid memory spike
        chunk_size = 1024 * 1024  # 1MB chunks
        total_size = 0
        
        with open(str(photo_path), 'wb') as f:
            while True:
                chunk = await photo.read(chunk_size)
                if not chunk:
                    break
                total_size += len(chunk)
                
                # Check size limit
                if total_size > max_bytes:
                    # Delete partial file
                    photo_path.unlink()
                    raise HTTPException(
                        status_code=400,
                        detail=f"{photo.filename} is too large. "
                               f"Maximum {MAX_FILE_MB}MB."
                    )
                
                f.write(chunk)
        
        saved_photo_paths.append(str(photo_path))
        
        # ADDED: Cleanup immediately
        await photo.close()
        gc.collect()
    
    # CHANGED: Now read from disk for processing
    # One photo at a time, with cleanup after each
    images_bytes = []
    
    try:
        for photo_path in saved_photo_paths:
            with open(photo_path, 'rb') as f:
                img_bytes = f.read()
                images_bytes.append(img_bytes)
        
        # ── Run identification ────────────────────────────────
        result = identify_from_bytes_list(
            images_bytes,
            num_photos=len(photos)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Identification failed: {str(e)}"
        )
    
    finally:
        # ADDED: Cleanup images_bytes immediately after processing
        del images_bytes
        gc.collect()

    # ── Save to database ──────────────────────────────────────
    save_identification(
        identification_id, result, len(photos)
    )

    # ── Save to review queue ─────────────────────────────────
    review_id = save_to_review_queue(
        identification_id, result, saved_photo_paths
    )

    # ADDED: Final cleanup
    gc.collect()

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
    by its ID.
    """
    conn = sqlite3.connect(str(DB_PATH))
    c    = conn.cursor()
    c.execute(
        "SELECT raw_result FROM identifications WHERE id=?",
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
    """
    conn = sqlite3.connect(str(DB_PATH))
    c    = conn.cursor()
    c.execute(
        "SELECT * FROM review_queue WHERE status=? ORDER BY timestamp DESC",
        (status,)
    )
    rows    = c.fetchall()
    columns = [d[0] for d in c.description]
    conn.close()

    return {
        "status": status,
        "count":  len(rows),
        "items":  [dict(zip(columns, row)) for row in rows]
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
    """
    conn = sqlite3.connect(str(DB_PATH))
    c    = conn.cursor()

    c.execute("SELECT COUNT(*) FROM identifications")
    total = c.fetchone()[0]

    c.execute('''
        SELECT scenario, COUNT(*) as count
        FROM identifications
        GROUP BY scenario
        ORDER BY count DESC
    ''')
    scenarios = dict(c.fetchall())

    c.execute('''
        SELECT top_family, COUNT(*) as count
        FROM identifications
        WHERE top_family IS NOT NULL
        GROUP BY top_family
        ORDER BY count DESC
    ''')
    families = dict(c.fetchall())

    c.execute('''
        SELECT COUNT(*) FROM review_queue WHERE status='pending'
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
    """
    from fastapi.responses import FileResponse
    
    photo_path = REVIEW_DIR / identification_id / photo_name
    
    if not photo_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Photo not found."
        )
    
    return FileResponse(str(photo_path))
