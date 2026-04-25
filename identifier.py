# ============================================================
# identifier.py — Core identification logic
# AmmoniteID v1.0
# ============================================================
# Loads the model once at startup and provides the
# identify() function used by the API endpoints.
# Smart cropping automatically detects the fossil
# region when it occupies a small part of the frame.
# ============================================================
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
import io
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from config import (
    MODEL_PATH, CLASS_INFO, IMAGE_SIZE,
    FAMILY_LIKELY_THRESHOLD, FAMILY_POSSIBLE_THRESHOLD,
    GENUS_BEST_MATCH_THRESHOLD, GENUS_POSSIBLE_THRESHOLD
)

# ── Load class information ───────────────────────────────────
with open(str(CLASS_INFO), 'r') as f:
    class_info = json.load(f)

INDEX_TO_CLASS   = {
    int(k): v
    for k, v in class_info['index_to_class'].items()
}
GENUS_TO_FAMILY  = class_info['genus_to_family']
FAMILY_TO_GENERA = class_info['family_to_genera']
NON_AMMONITE_MAP = class_info['non_ammonite_map']
NUM_CLASSES      = class_info['num_classes']

# ── Non-ammonite display names ───────────────────────────────
NON_AM_DISPLAY = {
    'Not_Ammonite':     'a rock, pebble or non-fossil object',
    'Belemnite Fossil': 'a Belemnite',
    'Bivalve':          'a Bivalve',
    'Devils toenail':   'a Devils Toenail (Gryphaea)',
}

# ── Load model once at startup ───────────────────────────────
print("Loading ammonite identification model...")
model = tf.keras.models.load_model(str(MODEL_PATH))
print(f"Model loaded. Classes: {NUM_CLASSES}")


# ============================================================
# IMAGE LOADING AND PREPROCESSING
# ============================================================

def find_fossil_region(img: Image.Image) -> Image.Image:
    """
    Attempts to find and crop to the most likely
    fossil region in the image before resizing.

    Uses edge detection to find the region with
    the most visual detail — fossils typically have
    more edge complexity than surrounding rock or
    plain background.

    Falls back to the full image if no clearly
    distinct region is found.
    """
    # Work on a small thumbnail for speed
    thumb = img.copy()
    thumb.thumbnail((224, 224))
    thumb_array = np.array(
        thumb.convert('L'), dtype=np.float32
    )

    h, w = thumb_array.shape

    # Compute edge strength using horizontal and
    # vertical gradients — fossils have more edges
    # than plain rock or background
    grad_x = np.abs(
        np.diff(thumb_array, axis=1, append=0)
    )
    grad_y = np.abs(
        np.diff(thumb_array, axis=0, append=0)
    )
    edge_strength = grad_x + grad_y

    # Scan overlapping windows to find the region
    # with the highest edge density
    window_h = int(h * 0.6)
    window_w = int(w * 0.6)

    best_score = -1
    best_top   = 0
    best_left  = 0

    step = max(1, min(h, w) // 8)

    for top in range(0, h - window_h + 1, step):
        for left in range(0, w - window_w + 1, step):
            region = edge_strength[
                top:top + window_h,
                left:left + window_w
            ]
            score = float(np.mean(region))
            if score > best_score:
                best_score = score
                best_top   = top
                best_left  = left

    # Scale coordinates back to original image size
    scale_x = img.width  / w
    scale_y = img.height / h

    # Add padding around the detected region
    padding = 0.1
    orig_top    = max(0, int(
        (best_top - h * padding) * scale_y
    ))
    orig_left   = max(0, int(
        (best_left - w * padding) * scale_x
    ))
    orig_bottom = min(img.height, int(
        (best_top + window_h + h * padding) * scale_y
    ))
    orig_right  = min(img.width, int(
        (best_left + window_w + w * padding) * scale_x
    ))

    # Only crop if the region is meaningfully smaller
    # than the full image — otherwise use full image
    crop_area = (
        (orig_right  - orig_left) *
        (orig_bottom - orig_top)
    )
    full_area = img.width * img.height

    if crop_area < full_area * 0.85:
        return img.crop((
            orig_left,
            orig_top,
            orig_right,
            orig_bottom
        ))

    return img


def load_image_from_bytes(
    image_bytes: bytes,
    smart_crop: bool = True
) -> np.ndarray:
    """
    Converts raw image bytes into a numpy array
    ready for the model.

    smart_crop: if True, automatically detects and
    crops to the fossil region before resizing.
    Significantly improves results when the fossil
    occupies a small part of the frame.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')

    if smart_crop:
        img = find_fossil_region(img)

    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(img, dtype=np.float32)


# ============================================================
# IDENTIFICATION LOGIC
# ============================================================

def build_bar(score: float, width: int = 10) -> str:
    """Converts a probability score to a visual bar.
    Example: 0.70 → ███████░░░
    """
    filled = round(score * width)
    empty  = width - filled
    return '█' * filled + '░' * empty


def get_genus_wording(score: float) -> str:
    """Converts a normalised genus score to wording."""
    if score >= GENUS_BEST_MATCH_THRESHOLD:
        return 'best match'
    elif score >= GENUS_POSSIBLE_THRESHOLD:
        return 'possible'
    else:
        return 'less likely'


def identify_single(image_array: np.ndarray) -> dict:
    """
    Runs a single preprocessed image array through
    the model and returns raw and grouped scores.
    """
    # Add batch dimension — model expects (1, 224, 224, 3)
    batch = np.expand_dims(image_array, axis=0)
    raw   = model.predict(batch, verbose=0)[0]

    # Map index to class name
    class_scores = {
        INDEX_TO_CLASS[i]: float(raw[i])
        for i in range(NUM_CLASSES)
    }

    # Separate genus scores from non-ammonite scores
    genus_scores  = {}
    non_am_scores = {}

    for name, score in class_scores.items():
        if name in GENUS_TO_FAMILY:
            genus_scores[name]  = score
        else:
            non_am_scores[name] = score

    # Sum genus scores within each family
    family_scores = {
        family: sum(
            genus_scores.get(g, 0.0)
            for g in genera
        )
        for family, genera in FAMILY_TO_GENERA.items()
    }

    non_am_total = sum(non_am_scores.values())
    top_non_am   = max(
        non_am_scores, key=non_am_scores.get
    )

    return {
        'class_scores':  class_scores,
        'genus_scores':  genus_scores,
        'family_scores': family_scores,
        'non_am_scores': non_am_scores,
        'non_am_total':  non_am_total,
        'top_non_am':    top_non_am,
    }


def combine_results(single_results: list) -> dict:
    """
    Combines results from multiple photos of the
    same specimen by averaging all scores.

    When multiple photos agree the combined confidence
    increases. When they disagree the confidence drops
    which is the honest and correct response.
    """
    if len(single_results) == 1:
        return single_results[0]

    all_classes = single_results[0]['class_scores'].keys()

    # Average scores across all photos
    avg_class = {
        cls: float(np.mean([
            r['class_scores'][cls]
            for r in single_results
        ]))
        for cls in all_classes
    }

    # Regroup averaged scores
    genus_scores  = {}
    non_am_scores = {}

    for name, score in avg_class.items():
        if name in GENUS_TO_FAMILY:
            genus_scores[name]  = score
        else:
            non_am_scores[name] = score

    family_scores = {
        family: sum(
            genus_scores.get(g, 0.0)
            for g in genera
        )
        for family, genera in FAMILY_TO_GENERA.items()
    }

    non_am_total = sum(non_am_scores.values())
    top_non_am   = max(
        non_am_scores, key=non_am_scores.get
    )

    return {
        'class_scores':  avg_class,
        'genus_scores':  genus_scores,
        'family_scores': family_scores,
        'non_am_scores': non_am_scores,
        'non_am_total':  non_am_total,
        'top_non_am':    top_non_am,
    }


def build_result(
    combined: dict,
    num_photos: int
) -> dict:
    """
    Takes combined scores and builds the full
    structured result with scenario, genus breakdown
    and formatted text output.
    """
    family_scores    = combined['family_scores']
    non_am_total     = combined['non_am_total']
    top_non_am       = combined['top_non_am']
    genus_scores     = combined['genus_scores']

    top_family       = max(
        family_scores, key=family_scores.get
    )
    top_family_score = family_scores[top_family] * 100
    top_non_am_score = combined['non_am_scores'][top_non_am]

    # ── Determine scenario ───────────────────────────────────
    if non_am_total > top_family_score:
        scenario = 'non_ammonite'
    elif top_family_score >= FAMILY_LIKELY_THRESHOLD:
        scenario = 'likely'
    elif top_family_score >= FAMILY_POSSIBLE_THRESHOLD:
        scenario = 'possible'
    else:
        scenario = 'uncertain'

    print(f"SCENARIO DEBUG: scenario={scenario}, score={top_family_score}")

    

    # ── Build genus breakdown ────────────────────────────────
    genus_breakdown = []
    if scenario in ('likely', 'possible'):
        family_genera = FAMILY_TO_GENERA[top_family]
        family_total = family_scores[top_family]  # Use raw probability

        for genus in family_genera:
            raw  = genus_scores.get(genus, 0.0)
            norm = (
                raw / family_total
                if family_total > 0 else 0.0
            )
            print(f"GENUS DEBUG: {genus} = {norm}")
            genus_breakdown.append({
                'genus':            genus,
                'normalised_score': norm,
                'bar':              build_bar(norm),
                'wording':          get_genus_wording(norm),
                'percentage':       round(norm * 100),
            })

        genus_breakdown.sort(
            key=lambda x: x['normalised_score'],
            reverse=True
        )

    # ── Identify non-ammonite category ───────────────────────
    non_am_category = NON_AMMONITE_MAP.get(
        top_non_am, 'Other_Fossil'
    )

    result = {
        'scenario':         scenario,
        'num_photos':       num_photos,
        'top_family':       top_family,
        'top_family_score': round(top_family_score),
        'family_scores': {
            k: round(v, 1)
            for k, v in family_scores.items()
        },
        'genus_breakdown':  genus_breakdown,
        'non_am_total':     round(non_am_total * 100),
        'top_non_am':       top_non_am,
        'top_non_am_score': round(top_non_am_score * 100),
        'non_am_category':  non_am_category,
        'non_am_display':   NON_AM_DISPLAY.get(
                               top_non_am,
                               top_non_am
                            ),
    }

    result['formatted_output'] = format_output(result)
    return result


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def format_output(result: dict) -> str:
    """
    Produces the agreed display text for the app.
    Handles all six output scenarios.
    """
    scenario   = result['scenario']
    num_photos = result.get('num_photos', 1)
    lines      = []

    # ── Scenario 1 and 2: Ammonite identified ────────────────
    if scenario in ('likely', 'possible'):
        family    = result['top_family']
        score_pct = result['top_family_score']
        wording   = (
            'Likely' if scenario == 'likely'
            else 'Possible'
        )

        lines.append(
            f"FAMILY:  {family}"
            f"     [{wording} — {score_pct}% confidence]"
        )

        if num_photos > 1:
            lines.append(
                f"         Based on {num_photos} photographs"
            )

        lines.append("")
        lines.append("GENUS:")

        for g in result['genus_breakdown']:
            lines.append(
                f"  {g['genus']:<28}"
                f"  {g['bar']}  {g['wording']}"
            )

        lines.append("")
        lines.append(
            "If a more accurate identification is required,"
        )
        lines.append(
            "it is recommended to consult with an expert."
        )

    # ── Scenario 3: Uncertain ────────────────────────────────
    elif scenario == 'uncertain':
        lines.append(
            "FAMILY:  Uncertain — confidence too low"
            " to suggest a family"
        )
        lines.append("")
        lines.append(
            "GENUS:   Cannot be determined from this image."
        )
        lines.append("")
        lines.append("For best results:")
        lines.append(
            "  — Crop the photo so the fossil fills"
            " most of the frame"
        )
        lines.append(
            "  — Photograph from directly above"
        )
        lines.append(
            "  — Use even lighting with no shadows"
            " across the ribs"
        )
        lines.append(
            "  — Try a second photo from a"
            " different angle"
        )

    # ── Scenarios 4, 5 and 6: Non-ammonite ──────────────────
    elif scenario == 'non_ammonite':
        non_am_cat = result['non_am_category']

        if non_am_cat == 'Not_Fossil':
            lines.append(
                "FAMILY:  No ammonite detected"
            )
            lines.append("")
            lines.append(
                "This appears to be "
                + result['non_am_display'] + "."
            )
            lines.append("")
            lines.append("For best results:")
            lines.append(
                "  — Crop the photo so the fossil fills"
                " most of the frame"
            )
            lines.append(
                "  — Make sure the specimen is well lit"
                " with no strong shadows"
            )
            lines.append(
                "  — Photograph from directly above"
            )

        else:
            lines.append(
                "FAMILY:  Other fossil type detected"
            )
            lines.append(
                "         (not an ammonite)"
            )
            lines.append("")

            if result['top_non_am_score'] > 60:
                lines.append(
                    "This appears to be "
                    + result['non_am_display'] + "."
                )
            else:
                lines.append(
                    "This resembles another fossil type"
                    " but the image is not clear enough"
                    " to determine which."
                )

            lines.append("")
            lines.append(
                "If a more accurate identification"
                " is required, it is recommended"
                " to consult with an expert."
            )

    return '\n'.join(lines)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def identify_from_bytes_list(
    images_bytes: list,
    num_photos: int
) -> dict:
    """
    Main entry point called by the API.

    Takes a list of raw image byte strings,
    runs smart crop and identification on each,
    combines the results, and returns the full
    structured result with formatted output.
    """
    single_results = []

    for img_bytes in images_bytes:
        img_array = load_image_from_bytes(
            img_bytes,
            smart_crop=True
        )
        result = identify_single(img_array)
        single_results.append(result)

    combined = combine_results(single_results)
    return build_result(combined, num_photos)
