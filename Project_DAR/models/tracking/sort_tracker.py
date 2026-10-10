# models/tracking/sort_tracker.py

"""
SORT tracking module
Wrapper cho SORT tracker.
"""

import sys
import numpy as np

from configs.ablation_config import (
    SORT_ROOT,
    SORT_MAX_AGE,
    SORT_MIN_HITS,
    SORT_IOU_THRESHOLD
)

# =========================================================
# IMPORT SORT
# =========================================================

if SORT_ROOT not in sys.path:
    sys.path.insert(0, SORT_ROOT)

try:
    from sort import Sort
    SORT_AVAILABLE = True
    print("SORT imported successfully")
except Exception as e:
    SORT_AVAILABLE = False
    print("SORT import failed:", e)


# =========================================================
# SORT TRACKER WRAPPER
# =========================================================

class SortTracker:
    """
    Wrapper class cho SORT tracker.
    """

    def __init__(
        self,
        max_age=SORT_MAX_AGE,
        min_hits=SORT_MIN_HITS,
        iou_threshold=SORT_IOU_THRESHOLD
    ):

        if not SORT_AVAILABLE:
            raise ImportError("SORT library not available")

        self.tracker = Sort(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )

    # =====================================================
    # UPDATE TRACKS
    # =====================================================

    def update(self, detections):
        """
        Input:
            detections: numpy array Nx4
            [[x1,y1,x2,y2], ...]

        Output:
            tracks: list of dict
        """

        if detections is None or len(detections) == 0:
            detections = np.empty((0, 4))

        tracks_raw = self.tracker.update(detections)

        tracks = []

        for tr in tracks_raw:

            x1, y1, x2, y2, track_id = tr

            tracks.append(
                {
                    "track_id": int(track_id),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                }
            )

        return tracks
