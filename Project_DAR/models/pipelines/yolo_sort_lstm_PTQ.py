# Code 2
# models/pipelines/yolo_sort_lstm_PTQ.py
"""
YOLO + SORT + LSTM PTQ pipeline.
"""

import os
import csv
import cv2
import numpy as np
import torch
import time

from collections import deque

from configs.ablation_config import (
    VIDEO_DIR,
    OUTPUT_DIR,
    VIDEO_ID_MAP,
    DEVICE,
    WINDOW_SIZE,
    ACTION_ID_MAP,
    ID2ACTION,
    NUM_CLASSES,
    MIN_SEGMENT_DURATION,
    CLUSTER_GAP_SEC,
    TIME_TOLERANCE,
    FPS
)

from models.detection.yolo_PTQ import YOLODetector
from models.tracking.sort_tracker import SortTracker
from models.temporal.lstm_model import load_lstm_model

from models.fusion.semantic_hold import (
    init_semantic_memory,
    update_semantic_hold
)

from models.fusion.label_fusion import (
    init_fusion_counters,
    fuse_labels
)

from postprocessing.decision import lstm_argmax_decision

from postprocessing.segment_utils import (
    extract_segments_from_labels,
    remove_overlapping_segments
)

from postprocessing.segment_postprocess import (
    simple_merge_same_actions,
    apply_tolerance_to_segments
)

# =========================================================
# CSV
# =========================================================

def save_video_csv(rows, path):

    with open(path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["video_id", "ID", "start", "stop"])

        for r in rows:

            writer.writerow([
                r["video_id"],
                r["label_id"],
                r["start"],
                r["stop"]
            ])

def save_global_csv(rows, path):

    rows = sorted(rows, key=lambda x: (x["video_id"], x["start"]))

    with open(path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["video_id", "ID", "start", "stop"])

        for r in rows:
            writer.writerow([
                r["video_id"],
                r["label_id"],
                r["start"],
                r["stop"]
            ])

def bbox_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)

    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxB_area = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    union = boxA_area + boxB_area - inter_area

    return inter_area / union
def map_tracks_to_labels(tracks, detections, iou_thresh=0.3):

    track_labels = {}

    for track in tracks:

        track_id = track["track_id"]
        track_box = track["bbox"]

        labels = []

        for det in detections:

            det_box = det["bbox"]

            iou = bbox_iou(track_box, det_box)

            if iou > iou_thresh:

                labels.append(det["class_name"])

        track_labels[track_id] = labels

    return track_labels
def bbox_overlap_ratio_over_box1(box1, box2):
    """Return overlap area / area(box1)."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    inter = x_overlap * y_overlap
    area1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
    return inter / area1 if area1 > 0 else 0.0

# =========================================================
# PIPELINE
# =========================================================

def inference_and_save_csv():

    detector = YOLODetector(device=DEVICE)
    tracker = SortTracker()
    lstm_model = load_lstm_model()

    video_files = [
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    all_rows = []
    print("YOLO + SORT + LSTM pipeline started")

    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_id = VIDEO_ID_MAP.get(video_name, -1)
        
        print("Processing:", video_file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_num = 0

        feature_seq = deque(maxlen=WINDOW_SIZE)
        semantic_memory = init_semantic_memory()
        fusion_counters = init_fusion_counters()
        lstm_time_labels = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            time_in_video = frame_num / fps

            # =================================================
            # 1. YOLO DETECTION - CHỈ GỌI 1 LẦN DUY NHẤT
            # =================================================
            raw_detections = detector.get_raw_detections(frame)
            sort_inputs, detections = detector.get_sort_inputs(frame)
            # sort_inputs: numpy array Nx4 cho SORT
            # detections: list [{"bbox": [x1,y1,x2,y2], "class_name": "...", "conf": ...}]
            
            # =================================================
            # 2. SORT TRACKING
            # =================================================
            tracks = tracker.update(sort_inputs)
            # tracks: [{"track_id": 3, "bbox": [x1,y1,x2,y2]}]

            # =================================================
            # 3. GET DRIVER TRACK (LARGEST BBOX)
            # =================================================

            driver_track_id = None
            driver_bbox = None

            if tracks:

                best_track = max(
                    tracks,
                    key=lambda t: (
                        (t["bbox"][2] - t["bbox"][0]) *
                        (t["bbox"][3] - t["bbox"][1])
                    )
                )

                driver_track_id = best_track["track_id"]
                driver_bbox = best_track["bbox"]
                
            # =================================================
            # 4. YOLO LABELS TRONG DRIVER ROI
            # =================================================

            yolo_labels = []
            
            if driver_bbox is None:
                for det in raw_detections:
                    yolo_labels.append(det["class_name"])
            else:
                for det in raw_detections:
                    det_box = det["bbox"]
                    label = det["class_name"]
                    if bbox_overlap_ratio_over_box1(det_box, driver_bbox) >= 0.0001:
                        yolo_labels.append(label)

            # =================================================
            # 5. SEMANTIC HOLD (TRACK-CENTRIC)
            # =================================================

            labels_for_fusion = update_semantic_hold(
                track_id=driver_track_id,
                yolo_labels=yolo_labels,
                frame_idx=frame_num,
                fps=fps,
                semantic_memory=semantic_memory
            )

            # =================================================
            # 6. FUSION RULE
            # =================================================

            merged_label = fuse_labels(
                labels_for_fusion,
                fusion_counters
            )

            # =================================================
            # 7. ONE-HOT ENCODING
            # =================================================

            onehot = np.zeros(NUM_CLASSES, dtype=np.float32)

            label_id = ACTION_ID_MAP.get(merged_label, 0)

            onehot[label_id] = 1.0

            feature_seq.append(onehot)

            # ===============================
            # LSTM inference
            # ===============================

            if len(feature_seq) == WINDOW_SIZE:

                input_tensor = torch.tensor(
                    [list(feature_seq)],
                    dtype=torch.float32
                ).to(DEVICE)

                with torch.no_grad():

                    logits = lstm_model(input_tensor)

                    pred_class = lstm_argmax_decision(logits)

                    pred_label = ID2ACTION[pred_class]

            else:

                pred_label = merged_label

            lstm_time_labels.append({
                "time": time_in_video,
                "label": pred_label
            })

        cap.release()
        # =================================================
        # 9. SEGMENT EXTRACTION
        # =================================================
        segments = extract_segments_from_labels(
            lstm_time_labels,
            min_segment_duration=MIN_SEGMENT_DURATION
        )
        segments = remove_overlapping_segments(segments)
        segments = simple_merge_same_actions(
            segments,
            cluster_gap_sec=CLUSTER_GAP_SEC
        )
        segments = apply_tolerance_to_segments(
            segments,
            tolerance_sec=TIME_TOLERANCE
        )

        # =================================================
        # 9. SAVE CSV (giữ nguyên)
        # =================================================
        rows = []
        for seg in segments:
            label = seg["action"]
            if label == "No action":
                continue
            label_id = ACTION_ID_MAP[label]
            row = {
                "video_id": video_id,
                "label_id": label_id,
                "start": seg["start_time"],
                "stop": seg["end_time"]
            }
            rows.append(row)
            all_rows.append(row)

        # Save video CSV
        out_dir = os.path.join(
            OUTPUT_DIR,
            "YOLO_SORT_LSTM",
            "inference_and_save_csv"
        )
        os.makedirs(out_dir, exist_ok=True)
        video_csv_path = os.path.join(out_dir, f"{video_id}.csv")
        save_video_csv(rows, video_csv_path)
        print("Saved:", video_csv_path)

    # Save global CSV
    global_csv_path = os.path.join(
        OUTPUT_DIR,
        "YOLO_SORT_LSTM",
        "all_predictions.csv"
    )
    save_global_csv(all_rows, global_csv_path)
    print("Saved:", global_csv_path)
