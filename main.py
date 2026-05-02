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

@app.get("/export-corrections")
def export_corrections(
    status: str = "reviewed",
    format: str = "json"
):
    """
    Export expert corrections from the review queue.
    
    Parameters:
    - status: Filter by status (default: 'reviewed')
              Options: 'reviewed', 'all'
    - format: Output format (default: 'json')
              Options: 'json', 'csv'
    
    Returns:
    List of corrections with:
    - photo_paths: Paths to the photos
    - ai_prediction: What AI predicted (family/genus)
    - expert_correction: What expert said is correct
    - timestamp: When reviewed
    - notes: Expert's notes
    
    Example:
    GET /export-corrections
    GET /export-corrections?status=all
    GET /export-corrections?format=csv
    """
    
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    
    # Build query based on status filter
    if status == "all":
        query = "SELECT * FROM review_queue"
        c.execute(query)
    else:
        query = "SELECT * FROM review_queue WHERE status=?"
        c.execute(query, (status,))
    
    rows = c.fetchall()
    columns = [d[0] for d in c.description]
    conn.close()
    
    # Convert to list of dicts
    corrections = []
    for row in rows:
        item = dict(zip(columns, row))
        
        # Only include items where expert provided corrections
        if item.get('expert_family') or item.get('expert_genus'):
            
            # Parse photo paths from JSON
            photo_paths_json = item.get('photo_paths')
            if photo_paths_json:
                try:
                    photo_paths = json.loads(photo_paths_json)
                except:
                    photo_paths = []
            else:
                photo_paths = []
            
            correction = {
                'id': item['id'],
                'timestamp': item['timestamp'],
                'reviewed_at': item.get('reviewed_at'),
                'reviewed_by': item.get('reviewed_by'),
                
                # AI's prediction
                'ai_family': item.get('ai_family'),
                'ai_genus': item.get('ai_genus'),
                'ai_confidence': item.get('ai_confidence'),
                
                # Expert's correction
                'expert_family': item.get('expert_family'),
                'expert_genus': item.get('expert_genus'),
                'expert_notes': item.get('expert_notes'),
                
                # Photo information
                'photo_paths': photo_paths,
                'num_photos': len(photo_paths),
                
                # Was it correct or incorrect?
                'was_correct': (
                    item.get('ai_family') == item.get('expert_family') and
                    item.get('ai_genus') == item.get('expert_genus')
                ),
                
                # Identification ID (to get original photos if needed)
                'identification_id': item.get('identification_id')
            }
            
            corrections.append(correction)
    
    # Return in requested format
    if format == "csv":
        # Convert to CSV format
        import io
        import csv
        
        output = io.StringIO()
        if corrections:
            fieldnames = corrections[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for correction in corrections:
                # Flatten photo_paths for CSV
                row = correction.copy()
                row['photo_paths'] = ';'.join(correction['photo_paths'])
                writer.writerow(row)
        
        csv_content = output.getvalue()
        
        return JSONResponse(
            content={"csv": csv_content},
            headers={
                "Content-Disposition": "attachment; filename=corrections.csv"
            }
        )
    
    else:  # JSON format (default)
        return {
            "status": status,
            "count": len(corrections),
            "corrections": corrections,
            "model_version": MODEL_VERSION,
            "exported_at": datetime.utcnow().isoformat()
        }


@app.get("/correction-stats")
def get_correction_stats():
    """
    Get statistics about expert corrections.
    
    Shows:
    - How many corrections total
    - Accuracy by family/genus
    - Common mistakes the AI makes
    
    Example:
    GET /correction-stats
    """
    
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    
    # Get all reviewed items with expert corrections
    c.execute("""
        SELECT 
            ai_family, ai_genus, ai_confidence,
            expert_family, expert_genus,
            status
        FROM review_queue
        WHERE status = 'reviewed'
        AND expert_family IS NOT NULL
    """)
    
    rows = c.fetchall()
    conn.close()
    
    total = len(rows)
    if total == 0:
        return {
            "total_corrections": 0,
            "message": "No expert corrections yet"
        }
    
    # Calculate statistics
    correct_family = 0
    correct_genus = 0
    
    family_mistakes = {}
    genus_mistakes = {}
    
    for row in rows:
        ai_fam, ai_gen, ai_conf, exp_fam, exp_gen, status = row
        
        # Count correct predictions
        if ai_fam == exp_fam:
            correct_family += 1
            
            if ai_gen == exp_gen:
                correct_genus += 1
        
        # Track mistakes
        if ai_fam != exp_fam:
            mistake_key = f"{ai_fam} → {exp_fam}"
            family_mistakes[mistake_key] = family_mistakes.get(mistake_key, 0) + 1
        
        if ai_gen != exp_gen:
            mistake_key = f"{ai_gen} → {exp_gen}"
            genus_mistakes[mistake_key] = genus_mistakes.get(mistake_key, 0) + 1
    
    # Sort mistakes by frequency
    top_family_mistakes = sorted(
        family_mistakes.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    top_genus_mistakes = sorted(
        genus_mistakes.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    return {
        "total_corrections": total,
        
        "accuracy": {
            "family": {
                "correct": correct_family,
                "incorrect": total - correct_family,
                "percentage": round(correct_family / total * 100, 1)
            },
            "genus": {
                "correct": correct_genus,
                "incorrect": total - correct_genus,
                "percentage": round(correct_genus / total * 100, 1)
            }
        },
        
        "top_family_mistakes": [
            {"mistake": m[0], "count": m[1]}
            for m in top_family_mistakes
        ],
        
        "top_genus_mistakes": [
            {"mistake": m[0], "count": m[1]}
            for m in top_genus_mistakes
        ],
        
        "model_version": MODEL_VERSION
    }

        )
    
    return FileResponse(str(photo_path))
