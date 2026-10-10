# models/fusion/label_fusion.py

"""
Rule-based label fusion.

Kết hợp các detection của YOLO thành nhãn hành động cuối cùng.
Giữ nguyên logic từ code gốc của pipeline.
"""

from configs.ablation_config import FUSION_THRESHOLD_FRAMES


# =========================================================
# INIT COUNTERS
# =========================================================

def init_fusion_counters():
    """
    Khởi tạo bộ đếm cho các action cần tích lũy frame.
    """
    return {
        "talk": 0,
        "yawn": 0,
        "smoking": 0
    }


# =========================================================
# RULE-BASED LABEL FUSION
# =========================================================

def fuse_labels(labels, counters):
    """
    Rule-based fusion cho YOLO labels.

    Args
    ----
    labels : list[str]
        Danh sách label YOLO detect trong frame.

    counters : dict
        Bộ đếm frame để xác nhận action.

    Returns
    -------
    merged_label : str
        Nhãn hành động cuối cùng.
    """

    merged_label = "No action"
    talk_cnt = counters.get('talk', 0)
    yawn_cnt = counters.get('yawn', 0)
    smoke_cnt = counters.get('smoking', 0)
    #threshold_frames = max(1, int(fps * 1.0))  # ~1 giây
    threshold_frames = FUSION_THRESHOLD_FRAMES
    CUSTOM_NAMES = {
    0: 'Smoking',
    1: 'Calling', 
    2: 'Texting',
    3: 'Drinking',
    4: 'Yawning',
    5: 'Talking', 
    6: 'Cigarette',
    7: 'Phone',
    8: 'Bottle',
    9: 'Open mouth'
    }

    if not labels:
        counters['talk'] = counters['yawn'] = counters['smoking'] = 0
        return "No action"
    label_set = set(labels)

    # ====== STRICT RULES ======
    if "Smoking" in label_set and "Cigarette" in label_set:
        smoke_cnt += 1;
        #merged_label = "Hut thuoc"
        merged_label = "Smoking" if smoke_cnt >= threshold_frames else "No action"
        yawn_cnt = 0; talk_cnt = 0

    elif "Drinking" in label_set or "Bottle" in label_set:
        merged_label = "Drinking"
        yawn_cnt = 0; talk_cnt = 0; smoke_cnt = 0

    elif "Phone" in label_set:
        if "Texting" in label_set:
            merged_label = "Texting"
        elif "Calling" in label_set:
            merged_label = "Calling"
        else:
            merged_label = "Texting"
        yawn_cnt = 0; talk_cnt = 0; smoke_cnt = 0

    elif "Talking" in label_set:
        talk_cnt += 1; yawn_cnt = 0; smoke_cnt = 0
        merged_label = "Talking" if talk_cnt >= threshold_frames else "No action"

    elif "Yawning" in label_set and "Open mouth" in label_set:
        yawn_cnt += 1; talk_cnt = 0; smoke_cnt = 0
        merged_label = "Yawning" if yawn_cnt >= threshold_frames else "No action"

    else:
        yawn_cnt = 0; talk_cnt = 0; smoke_cnt = 0
        if label_set == {"Texting"} or label_set == {"Cigarette"} or label_set == {"Smoking"} or label_set == {"Open mouth"}:
            merged_label = "No action"
        else:
            merged_label = labels[0] if labels else "No action"

    counters['talk'] = talk_cnt
    counters['yawn'] = yawn_cnt
    counters['smoking'] = smoke_cnt
    return merged_label
