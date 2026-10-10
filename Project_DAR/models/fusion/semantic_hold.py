"""
semantic_hold.py
Unified semantic hold module.

Áp dụng cho:
    YOLO
    YOLO + SORT
    YOLO + DeepSORT

Logic:
    - nếu YOLO detect → trả về full label list
    - nếu YOLO mất detection → giữ label cũ trong vài frame
"""

from configs.ablation_config import (
    SEMANTIC_HOLD_SEC,
    SEMANTIC_HOLD_MAX_AGE_OVERRIDE
)


# =========================================================
# INIT MEMORY
# =========================================================

def init_semantic_memory():
    """
    Format:

    {
        track_id: {
            "labels": list[str],
            "last_seen": frame_idx
        }
    }
    """
    return {}


# =========================================================
# SEMANTIC HOLD
# =========================================================

def update_semantic_hold(
    track_id,
    yolo_labels,
    frame_idx,
    fps,
    semantic_memory,
    hold_sec=SEMANTIC_HOLD_SEC,
    max_age_override=SEMANTIC_HOLD_MAX_AGE_OVERRIDE
):
    """
    Semantic hold giữ label khi YOLO mất detection.

    Args
    ----
    track_id : int | None

    yolo_labels : list[str]
        Full labels từ YOLO detection

    frame_idx : int

    fps : float

    semantic_memory : dict

    Returns
    -------
    labels_for_fusion : list[str]
    """

    # =====================================================
    # NO TRACK
    # =====================================================
    if track_id is None:
        return []

    # =====================================================
    # INIT MEMORY
    # =====================================================
    if track_id not in semantic_memory:

        semantic_memory[track_id] = {
            "labels": [],
            "last_seen": frame_idx
        }

    mem = semantic_memory[track_id]

    # =====================================================
    # MAX HOLD FRAMES
    # =====================================================
    if max_age_override is not None:
        max_hold_frames = max_age_override
    else:
        max_hold_frames = int(hold_sec * fps)

    # =====================================================
    # CASE 1: YOLO DETECT
    # =====================================================
    if yolo_labels:

        mem["labels"] = yolo_labels
        mem["last_seen"] = frame_idx

        return yolo_labels

    # =====================================================
    # CASE 2: YOLO LOST DETECTION
    # =====================================================
    age = frame_idx - mem["last_seen"]

    if age <= max_hold_frames:

        # semantic hold
        return mem["labels"]

    # =====================================================
    # EXPIRE HOLD
    # =====================================================
    mem["labels"] = []

    return []
