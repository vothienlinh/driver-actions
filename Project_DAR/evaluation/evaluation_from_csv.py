import numpy as np
import os
import pandas as pd
import csv
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# =========================================================
# BIẾN TOÀN CỤC
# =========================================================
GT_FILE = r"/home/linhvt/Documents/GT_new.csv"
PRED_FILE = r"/home/linhvt/Documents/Project_DAR/Outputs/YOLO_SORT_LSTM/out.csv"
OUTPUT_BASE_DIR = r"/home/linhvt/Documents/Project_DAR/Outputs/YOLO_SORT_LSTM/out2S"

THRESHOLD = 2.0
SEGMENT_OS_MODE = "N"  

# tolerance apply sau khi load pred csv
APPLY_PRED_TOLERANCE = True
PRED_TOLERANCE_SEC = 1.0

my_video_durations = {
    111: 319, 112: 319, 121: 325, 122: 325, 131: 273, 132: 273,
    141: 315, 142: 315, 151: 263, 152: 263, 161: 287, 162: 287,
    171: 279, 172: 279, 181: 335, 182: 335
}

CLASS_NAMES = ['No action', 'Smoking', 'Calling', 'Texting', 'Drinking', 'Yawning', 'Talking']


# =========================================================
# I/O & CHUẨN HÓA
# =========================================================
def read_any_csv(path):
    """Đọc CSV tự đoán delimiter."""
    return pd.read_csv(path, sep=None, engine="python")


def standardize_cols(df):
    """Chuẩn hóa header và schema."""
    df.columns = (df.columns
                  .str.replace('\ufeff', '', regex=True)
                  .str.strip()
                  .str.replace(r'\s+', '', regex=True)
                  .str.lower())

    rename_map = {
        'videoid': 'video_id',
        'video_id': 'video_id',
        'label': 'ID',
        'id': 'ID',
        'starttime': 'start',
        'endtime': 'stop',
        'startframe': 'start',
        'endframe': 'stop',
        'start': 'start',
        'stop': 'stop',
    }
    df = df.rename(columns=rename_map)

    required = {'video_id', 'ID', 'start', 'stop'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột {missing}. Columns hiện có: {list(df.columns)}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=required).copy()
    df['video_id'] = df['video_id'].astype(int)
    df['ID'] = df['ID'].astype(int)
    df['start'] = df['start'].astype(float)
    df['stop'] = df['stop'].astype(float)
    return df


def create_output_directory(base_folder):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_folder, f"TOL_eval_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "timelines"), exist_ok=True)
    return out_dir


# =========================================================
# TIME FORMAT
# =========================================================
def seconds_to_time_str(seconds):
    seconds = max(0.0, float(seconds))
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


# =========================================================
# TOLERANCE ON PRED CSV
# =========================================================
def apply_tolerance_to_segments(segments, tolerance_sec=1.0):
    if not segments:
        return segments

    segments = sorted(segments, key=lambda x: x['start_time'])
    expanded = []

    for i, s in enumerate(segments):
        sc = s.copy()

        if sc['action'] != 'No action':
            prev_end = None
            next_start = None

            # tìm hành động trước
            for j in range(i - 1, -1, -1):
                if segments[j]['action'] != 'No action':
                    prev_end = segments[j]['end_time']
                    break

            # tìm hành động sau
            for j in range(i + 1, len(segments)):
                if segments[j]['action'] != 'No action':
                    next_start = segments[j]['start_time']
                    break

            # --- START tolerance ---
            if prev_end is None or (sc['start_time'] - prev_end) >= tolerance_sec:
                sc['start_time'] = max(0.0, sc['start_time'] - tolerance_sec)

            # --- END tolerance ---
            if next_start is None or (next_start - sc['end_time']) >= tolerance_sec:
                sc['end_time'] = sc['end_time'] + tolerance_sec

            sc['duration_sec'] = sc['end_time'] - sc['start_time']
            sc['start_time_str'] = seconds_to_time_str(sc['start_time'])
            sc['end_time_str'] = seconds_to_time_str(sc['end_time'])

        expanded.append(sc)

    # giữ nguyên logic xử lý No action
    final = []
    i = 0
    while i < len(expanded):
        cur = expanded[i]
        if cur['action'] == 'No action':
            prev_end, next_start = None, None
            for j in range(i - 1, -1, -1):
                if expanded[j]['action'] != 'No action':
                    prev_end = expanded[j]['end_time']
                    break
            for j in range(i + 1, len(expanded)):
                if expanded[j]['action'] != 'No action':
                    next_start = expanded[j]['start_time']
                    break

            ns, ne = cur['start_time'], cur['end_time']
            if prev_end is not None and prev_end > ns:
                ns = prev_end
            if next_start is not None and next_start < ne:
                ne = next_start

            if ne - ns >= 0.1:
                cur['start_time'] = ns
                cur['end_time'] = ne
                cur['duration_sec'] = ne - ns
                cur['start_time_str'] = seconds_to_time_str(ns)
                cur['end_time_str'] = seconds_to_time_str(ne)
                final.append(cur)
        else:
            final.append(cur)
        i += 1

    return sorted(final, key=lambda x: x['start_time'])


def dataframe_to_segments(df):
    """
    Convert dataframe prediction -> list segments
    schema df: video_id, ID, start, stop
    """
    segments = []
    for _, row in df.iterrows():
        class_id = int(row["ID"])
        action_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"Class_{class_id}"
        segments.append({
            "video_id": int(row["video_id"]),
            "action": action_name,
            "start_time": float(row["start"]),
            "end_time": float(row["stop"]),
            "duration_sec": float(row["stop"]) - float(row["start"]),
            "start_time_str": seconds_to_time_str(float(row["start"])),
            "end_time_str": seconds_to_time_str(float(row["stop"])),
        })
    return segments


def segments_to_dataframe(segments):
    """
    Convert list segments -> dataframe with schema
    video_id, ID, start, stop
    """
    rows = []
    name_to_id = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    for seg in segments:
        action_name = seg["action"]
        if action_name not in name_to_id:
            continue

        rows.append({
            "video_id": int(seg["video_id"]),
            "ID": int(name_to_id[action_name]),
            "start": float(seg["start_time"]),
            "stop": float(seg["end_time"]),
        })

    if not rows:
        return pd.DataFrame(columns=["video_id", "ID", "start", "stop"])

    return pd.DataFrame(rows).sort_values(
        ["video_id", "start", "stop"]
    ).reset_index(drop=True)


def apply_tolerance_to_prediction_df(pred_df, tolerance_sec=1.0):
    """
    Apply tolerance per video_id.
    Quan trọng: phải xử lý từng video riêng để tránh segment video này
    ảnh hưởng hàng xóm video khác.
    """
    all_out = []

    for video_id, df_vid in pred_df.groupby("video_id"):
        segs = dataframe_to_segments(df_vid)
        segs_tol = apply_tolerance_to_segments(segs, tolerance_sec=tolerance_sec)
        out_vid = segments_to_dataframe(segs_tol)
        if len(out_vid) > 0:
            out_vid["video_id"] = int(video_id)
        all_out.append(out_vid)

    if not all_out:
        return pred_df.copy()

    out_df = pd.concat(all_out, ignore_index=True)
    out_df = out_df.sort_values(["video_id", "start", "stop"]).reset_index(drop=True)
    return out_df


# =========================================================
# HÀM CHUNG
# =========================================================
def temporal_iou(seg1, seg2):
    s1, e1 = seg1
    s2, e2 = seg2
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def normalize_confusion_matrix(cm):
    """Chuẩn hóa ma trận theo hàng."""
    row_sums = cm.sum(axis=1, keepdims=True)
    return cm.astype(float) / np.where(row_sums == 0, 1, row_sums)


def prepare_for_json(data):
    """Chuẩn bị dữ liệu để dump JSON."""
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(k, str) and k.startswith('_'):
                continue
            out[k] = prepare_for_json(v)
        return out
    elif isinstance(data, list):
        return [prepare_for_json(x) for x in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    else:
        return data


# =========================================================
# OVERLAP SCORE
# =========================================================
def calculate_total_iou_MplusN(gt, pred, thres=10.0):
    M, N = len(gt), len(pred)
    if M + N == 0:
        return {
            'threshold': thres,
            'overlap_score': 0.0,
            'total_gt': M,
            'total_pred': N,
            'matched_gt': 0,
            'matched_pred': 0,
            'total_iou_sum': 0.0,
            'num_positive_iou': 0
        }

    gt_sorted = gt.sort_values('start').reset_index(drop=True)
    pred = pred.reset_index(drop=True)

    used = set()
    total_iou = 0.0
    num_pos_iou = 0
    matched_gt = 0

    for _, g in gt_sorted.iterrows():
        vid, cls = g['video_id'], g['ID']
        gs, ge = g['start'], g['stop']

        cands = pred[(pred['video_id'] == vid) & (pred['ID'] == cls)]
        iou_this_gt = 0.0

        for idx, p in cands.iterrows():
            if idx in used:
                continue

            ps, pe = p['start'], p['stop']
            if abs(ps - gs) <= thres and abs(pe - ge) <= thres:
                iou = temporal_iou((gs, ge), (ps, pe))
                if iou > 0:
                    iou_this_gt += iou
                    total_iou += iou
                    num_pos_iou += 1
                    used.add(idx)

        if iou_this_gt > 0:
            matched_gt += 1

    matched_pred = len(used)
    score = total_iou / (M + N) if (M + N) > 0 else 0.0

    return {
        'threshold': thres,
        'overlap_score': score,
        'total_gt': M,
        'total_pred': N,
        'matched_gt': matched_gt,
        'matched_pred': matched_pred,
        'total_iou_sum': total_iou,
        'num_positive_iou': num_pos_iou
    }


def calculate_total_iou_only_N(gt, pred, thres=10.0):
    res = calculate_total_iou_MplusN(gt, pred, thres)
    res['overlap_score'] = res['total_iou_sum'] / res['total_pred'] if res['total_pred'] > 0 else 0.0
    return res


def calculate_segment_metrics_by_mode(gt, pred, thres=10.0, mode="N"):
    mode_raw = str(mode).strip()
    mode_upper = mode_raw.upper()

    if mode_upper == "N":
        res = calculate_total_iou_only_N(gt, pred, thres)
        protocol_name = "Total IoU / N"
    else:
        raise ValueError("Error")

    tp = res['num_positive_iou']
    fp = res['total_pred'] - res['num_positive_iou']
    fn = res['total_gt'] - res['num_positive_iou']

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'protocol': protocol_name,
        'segment_mode': mode_raw,
        'temporal_tolerance': thres,

        'overlap_score': res['overlap_score'],

        'precision': precision,
        'recall': recall,
        'f1_score': f1,

        'true_positive': tp,
        'false_positive': fp,
        'false_negative': fn,

        'total_gt': res['total_gt'],
        'total_pred': res['total_pred'],
        'matched_gt': res['matched_gt'],
        'matched_pred': res['matched_pred'],

        'total_iou_sum': res['total_iou_sum'],
        'num_positive_iou': res['num_positive_iou']
    }


def calculate_class_base_metrics(gt, pred, thres=10.0):
    gt_sorted = gt.sort_values('start').reset_index(drop=True)
    pred = pred.reset_index(drop=True)

    all_classes = sorted(set(gt['ID'].unique()) | set(pred['ID'].unique()))
    class_results = {}

    for class_id in all_classes:
        if class_id == 0:
            continue

        gt_class = gt_sorted[gt_sorted['ID'] == class_id]
        pred_class = pred[pred['ID'] == class_id]

        used = set()
        total_iou_sum = 0.0
        num_positive_iou = 0
        matched_gt = 0
        matched_pred = 0
        gt_scores_sum_iou = []

        for _, g in gt_class.iterrows():
            vid = g['video_id']
            gs, ge = g['start'], g['stop']

            cands = pred_class[pred_class['video_id'] == vid]
            iou_this_gt = 0.0
            matched_this_gt = []

            for idx, p in cands.iterrows():
                if idx in used:
                    continue

                ps, pe = p['start'], p['stop']
                if abs(ps - gs) <= thres and abs(pe - ge) <= thres:
                    iou = temporal_iou((gs, ge), (ps, pe))
                    if iou > 0:
                        iou_this_gt += iou
                        total_iou_sum += iou
                        num_positive_iou += 1
                        matched_this_gt.append(idx)
                        used.add(idx)

            if matched_this_gt:
                matched_gt += 1
                matched_pred += len(matched_this_gt)
                gt_scores_sum_iou.append(iou_this_gt)
            else:
                gt_scores_sum_iou.append(0.0)

        num_unmatched_pred = len(pred_class) - matched_pred

        class_results[class_id] = {
            'class_id': class_id,
            'total_gt': len(gt_class),
            'total_pred': len(pred_class),
            'matched_gt': matched_gt,
            'matched_pred': matched_pred,
            'total_iou_sum': total_iou_sum,
            'num_positive_iou': num_positive_iou,
            'gt_scores_sum_iou': gt_scores_sum_iou,
            'num_unmatched_pred': num_unmatched_pred,
            'all_scores': gt_scores_sum_iou + [0.0] * num_unmatched_pred
        }

    return class_results


def calculate_os_per_class(gt, pred, thres=10.0):
    class_results = calculate_class_base_metrics(gt, pred, thres)
    os_per_class = {}

    for class_id, base in class_results.items():
        M = base['total_gt']
        N = base['total_pred']

        os3 = base['total_iou_sum'] / N if N > 0 else 0.0

        tp = base['matched_gt']
        fp = base['total_pred'] - base['matched_pred']
        fn = base['total_gt'] - base['matched_gt']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        os_per_class[class_id] = {
            'class_id': class_id,
            'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}",
            'total_gt': M,
            'total_pred': N,
            'matched_gt': tp,
            'matched_pred': base['matched_pred'],
            'total_iou_sum': base['total_iou_sum'],
            'num_positive_iou': base['num_positive_iou'],
            'os_N': os3,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    return os_per_class


def calculate_os_per_class_for_thresholds(gt, pred, thresholds=range(1, 11)):
    all_results = {}
    for th in thresholds:
        os_per_class = calculate_os_per_class(gt, pred, thres=th)
        all_results[th] = os_per_class
    return all_results


# =========================================================
# FRAME-WISE METRICS
# =========================================================
def build_frame_labels(df_vid, T_max, num_classes=7):
    arr = np.zeros(T_max + 1, dtype=np.int32)
    for _, r in df_vid.iterrows():
        s = max(0, int(r['start']))
        e = max(0, int(r['stop']))
        if s > e:
            s, e = e, s
        s, e = min(s, T_max), min(e, T_max)
        cls = int(r['ID'])
        if 0 <= cls < num_classes:
            arr[s:e+1] = cls
    return arr


def compute_framewise_metrics(gt_df, pred_df, num_classes=7):
    all_vids = sorted(set(gt_df['video_id'].unique()).union(set(pred_df['video_id'].unique())))

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_frames = 0
    all_gt_labels = []
    all_pred_labels = []

    gt_group = gt_df.groupby('video_id')
    pr_group = pred_df.groupby('video_id')

    for vid in all_vids:
        gt_v = gt_group.get_group(vid) if vid in gt_group.groups else pd.DataFrame(columns=gt_df.columns)
        pr_v = pr_group.get_group(vid) if vid in pr_group.groups else pd.DataFrame(columns=pred_df.columns)

        gt_max = int(gt_v['stop'].max()) if len(gt_v) else 0
        pr_max = int(pr_v['stop'].max()) if len(pr_v) else 0
        T = max(gt_max, pr_max)

        if T <= 0:
            continue

        gt_arr = build_frame_labels(gt_v, T, num_classes)
        pr_arr = build_frame_labels(pr_v, T, num_classes)

        for g, p in zip(gt_arr, pr_arr):
            cm[g, p] += 1

        all_gt_labels.extend(gt_arr.tolist())
        all_pred_labels.extend(pr_arr.tolist())
        total_frames += T + 1

    overall_acc = np.trace(cm) / total_frames if total_frames > 0 else 0.0

    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    diag = np.diag(cm)

    per_class = {}
    for c in range(num_classes):
        precision = diag[c] / col_sum[c] if col_sum[c] > 0 else 0.0
        recall = diag[c] / row_sum[c] if row_sum[c] > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        class_acc = recall

        per_class[c] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(class_acc),
            'support_gt_frames': int(row_sum[c]),
            'predicted_frames': int(col_sum[c]),
            'true_frames': int(diag[c])
        }

    valid = [c for c in range(1, num_classes) if row_sum[c] > 0]
    macro_p = np.mean([per_class[c]['precision'] for c in valid]) if valid else 0.0
    macro_r = np.mean([per_class[c]['recall'] for c in valid]) if valid else 0.0
    macro_f1 = np.mean([per_class[c]['f1_score'] for c in valid]) if valid else 0.0
    macro_acc = np.mean([per_class[c]['accuracy'] for c in valid]) if valid else 0.0

    total_tp = np.trace(cm)
    total_samples = np.sum(cm)
    micro_p = total_tp / total_samples if total_samples > 0 else 0.0
    micro_r = total_tp / total_samples if total_samples > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    return {
        'num_classes': num_classes,
        'total_frames': int(total_frames),

        'overall_accuracy': float(overall_acc),
        'macro_accuracy': float(macro_acc),

        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),

        'micro_precision': float(micro_p),
        'micro_recall': float(micro_r),
        'micro_f1': float(micro_f1),

        'confusion_matrix': cm.tolist(),
        'per_class': per_class,

        'class_distribution_gt': row_sum.tolist(),
        'class_distribution_pred': col_sum.tolist(),
        'true_positives_per_class': diag.tolist(),

        '_confusion_matrix_raw': cm,
        '_all_gt_labels': all_gt_labels,
        '_all_pred_labels': all_pred_labels
    }


# =========================================================
# SEGMENT-WISE CONFUSION MATRIX
# =========================================================
def get_no_action_segments(df, video_id, max_duration):
    v_df = df[df['video_id'] == video_id].sort_values('start')
    no_action = []
    last_end = 0.0

    for _, row in v_df.iterrows():
        if row['start'] > last_end:
            no_action.append({
                'video_id': video_id,
                'ID': 0,
                'start': last_end,
                'stop': row['start']
            })
        last_end = max(last_end, row['stop'])

    if last_end < max_duration:
        no_action.append({
            'video_id': video_id,
            'ID': 0,
            'start': last_end,
            'stop': max_duration
        })
    return no_action


def compute_segment_cm(gt_df, pred_df, video_durations, num_classes=6, thres=10.0):
    all_vids = set(gt_df['video_id']).union(set(pred_df['video_id']))
    gt_no = []
    pred_no = []

    for vid in all_vids:
        gt_stop = gt_df[gt_df['video_id'] == vid]['stop']
        pr_stop = pred_df[pred_df['video_id'] == vid]['stop']

        gt_max = gt_stop.max() if not gt_stop.empty and not gt_stop.isna().all() else 0
        pr_max = pr_stop.max() if not pr_stop.empty and not pr_stop.isna().all() else 0

        duration = video_durations.get(vid, max(gt_max, pr_max))

        gt_no.extend(get_no_action_segments(gt_df, vid, duration))
        pred_no.extend(get_no_action_segments(pred_df, vid, duration))

    gt_full = pd.concat([gt_df, pd.DataFrame(gt_no)], ignore_index=True)
    pred_full = pd.concat([pred_df, pd.DataFrame(pred_no)], ignore_index=True)

    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    matched = set()

    for _, gt_row in gt_full.iterrows():
        cls = int(gt_row['ID'])
        vid = gt_row['video_id']
        gs, ge = gt_row['start'], gt_row['stop']

        best_iou = -1
        best_idx = None
        cands = pred_full[(pred_full['video_id'] == vid) & (pred_full['ID'] == cls)]

        for idx, p in cands.iterrows():
            if idx in matched:
                continue
            ps, pe = p['start'], p['stop']
            if abs(ps - gs) <= thres and abs(pe - ge) <= thres:
                iou = temporal_iou((gs, ge), (ps, pe))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

        if best_idx is not None:
            cm[cls, cls] += 1
            matched.add(best_idx)
        else:
            cm[cls, 0] += 1

    for idx, p in pred_full.iterrows():
        if idx not in matched and int(p['ID']) > 0:
            cm[0, int(p['ID'])] += 1

    return cm


# =========================================================
# SAVE FILES
# =========================================================
def save_frame_prf_metrics(frame_metrics, output_dir):
    path = os.path.join(output_dir, "frame_level_prf_metrics.csv")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Accuracy',
                    'GT Frames', 'Pred Frames', 'TP Frames'])

        for c in sorted(frame_metrics['per_class'].keys()):
            m = frame_metrics['per_class'][c]
            w.writerow([
                c,
                f"{m['precision']:.6f}",
                f"{m['recall']:.6f}",
                f"{m['f1_score']:.6f}",
                f"{m['accuracy']:.6f}",
                m['support_gt_frames'],
                m['predicted_frames'],
                m['true_frames']
            ])

        w.writerow([])
        w.writerow([
            'Macro Average (Classes 1-6)',
            f"{frame_metrics['macro_precision']:.6f}",
            f"{frame_metrics['macro_recall']:.6f}",
            f"{frame_metrics['macro_f1']:.6f}",
            f"{frame_metrics['macro_accuracy']:.6f}",
            '', '', ''
        ])

        w.writerow([
            'Micro Average',
            f"{frame_metrics['micro_precision']:.6f}",
            f"{frame_metrics['micro_recall']:.6f}",
            f"{frame_metrics['micro_f1']:.6f}",
            f"{frame_metrics['overall_accuracy']:.6f}",
            frame_metrics['total_frames'],
            frame_metrics['total_frames'],
            sum(m['true_frames'] for m in frame_metrics['per_class'].values())
        ])
    return path


def save_confusion_matrices(frame_cm, frame_cm_norm, seg_cm, seg_cm_norm, output_dir, class_names):
    frame_cm_path = os.path.join(output_dir, "frame_confusion_matrix.csv")
    pd.DataFrame(frame_cm, index=class_names, columns=class_names).to_csv(frame_cm_path, encoding='utf-8')

    frame_cm_norm_path = os.path.join(output_dir, "frame_confusion_matrix_normalized.csv")
    pd.DataFrame(frame_cm_norm, index=class_names, columns=class_names).to_csv(frame_cm_norm_path, encoding='utf-8')

    seg_cm_path = os.path.join(output_dir, "segment_confusion_matrix_raw.csv")
    pd.DataFrame(seg_cm, index=class_names, columns=class_names).to_csv(seg_cm_path, encoding='utf-8')

    seg_cm_norm_path = os.path.join(output_dir, "segment_confusion_matrix_normalized.csv")
    pd.DataFrame(seg_cm_norm, index=class_names, columns=class_names).to_csv(seg_cm_norm_path, encoding='utf-8')

    return frame_cm_path, frame_cm_norm_path, seg_cm_path, seg_cm_norm_path


def save_os_multi_threshold_csv(res_list, output_dir):
    path = os.path.join(output_dir, "os_multi_threshold.csv")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'Threshold',
            'OS_only_N',
            'Matched_GT',
            'Matched_Pred',
            'Total_GT',
            'Total_Pred',
            'TP',
            'FP',
            'FN',
            'Precision',
            'Recall',
            'F1',
            'Total_IoU_Sum',
            'Num_Positive_IoU'
        ])

        for r1, r3 in res_list:
            tp = r1['matched_gt']
            fp = r1['total_pred'] - r1['matched_pred']
            fn = r1['total_gt'] - r1['matched_gt']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            w.writerow([
                r1['threshold'],
                f"{r3['overlap_score']:.6f}",
                r1['matched_gt'],
                r1['matched_pred'],
                r1['total_gt'],
                r1['total_pred'],
                tp, fp, fn,
                f"{p:.6f}",
                f"{r:.6f}",
                f"{f1:.6f}",
                f"{r1['total_iou_sum']:.6f}",
                r1['num_positive_iou']
            ])
    return path


def save_os_per_class_csv(os_per_class, output_dir, threshold=2.0):
    path = os.path.join(output_dir, f"os_per_class_thres_{threshold}s.csv")

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'Class ID', 'Class Name',
            'GT', 'Pred', 'Matched GT',
            'OS /N',
            'Precision', 'Recall', 'F1',
            'Total IoU Sum', 'Num IoU>0'
        ])

        for class_id in sorted(os_per_class.keys()):
            r = os_per_class[class_id]
            w.writerow([
                class_id,
                r['class_name'],
                r['total_gt'],
                r['total_pred'],
                r['matched_gt'],
                f"{r['os_N']:.6f}",
                f"{r['precision']:.6f}",
                f"{r['recall']:.6f}",
                f"{r['f1_score']:.6f}",
                f"{r['total_iou_sum']:.6f}",
                r['num_positive_iou']
            ])

    return path


def save_os_per_class_multi_threshold_csv(os_per_class_all, output_dir):
    path = os.path.join(output_dir, "os_per_class_multi_threshold.csv")

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'Threshold', 'Class ID', 'Class Name',
            'GT', 'Pred', 'Matched GT',
            'OS /N',
            'Precision', 'Recall', 'F1'
        ])

        for th in sorted(os_per_class_all.keys()):
            for class_id in sorted(os_per_class_all[th].keys()):
                r = os_per_class_all[th][class_id]
                w.writerow([
                    th,
                    class_id,
                    r['class_name'],
                    r['total_gt'],
                    r['total_pred'],
                    r['matched_gt'],
                    f"{r['os_N']:.6f}",
                    f"{r['precision']:.6f}",
                    f"{r['recall']:.6f}",
                    f"{r['f1_score']:.6f}"
                ])

    return path


def save_predictions_csv(df, output_path):
    df = df.sort_values(["video_id", "start", "stop"]).reset_index(drop=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


# =========================================================
# VẼ BIỂU ĐỒ
# =========================================================
def plot_cm_heatmap(cm, class_names, title, path, normalize=True, cmap='GnBu', figsize=(10, 8)):
    plt.figure(figsize=figsize)

    if normalize:
        cm_show = normalize_confusion_matrix(cm) if not np.issubdtype(cm.dtype, np.floating) else cm
        cm_show = np.nan_to_num(cm_show)

        cm_display = np.round(cm_show, decimals=3)
        row_sums_rounded = cm_display.sum(axis=1)
        diff = 1.0 - row_sums_rounded

        for i in range(len(diff)):
            if diff[i] > 1e-2:
                max_col = np.argmax(cm_show[i])
                cm_display[i, max_col] += diff[i]
            elif diff[i] < 0:
                max_col = np.argmax(cm_show[i])
                cm_display[i, max_col] -= diff[i]

        cm_display = np.clip(cm_display, 0.0, 1.0)
        fmt = '.2f'
        vmin, vmax = 0, 1
    else:
        cm_show = cm
        fmt = 'd'
        vmin, vmax = None, None

    sns.heatmap(
        cm_show,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 14}
    )

    plt.title(title, fontsize=15, fontweight='bold', pad=16)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    return path


# =========================================================
# TIMELINE
# =========================================================
def get_timestamp_by_video_id(csv_file, video_id):
    data = standardize_cols(read_any_csv(csv_file))
    data = data[data['video_id'] == video_id]
    p_start = data['start'].tolist()
    p_end = data['stop'].tolist()
    p_label = data['ID'].tolist()
    return p_start, p_end, p_label


def visualize_timeline(gt_file, pred_file, video_id, video_name, output_dir):
    CLASS_COLORS = {
        0: (0.0, 0.0, 0.0, 0.0),
        1: (0.5, 0.447, 0.741, 0.5),
        2: (1.0, 0.5, 0.0, 1.0),
        3: (0.0, 0.6, 0.0, 1.0),
        4: (0.8, 0.0, 0.0, 1.0),
        5: (0.5, 0.0, 0.5, 1.0),
        6: (0.0, 0.0, 0.741, 1.0),
    }

    ACTION_NAMES = {
        0: "No action",
        1: "Smoking",
        2: "Calling",
        3: "Texting",
        4: "Drinking",
        5: "Yawning",
        6: "Talking",
    }

    def get_color(label_id):
        return CLASS_COLORS.get(int(label_id), (0.5, 0.5, 0.5, 1.0))

    gt_start, gt_end, gt_labels = get_timestamp_by_video_id(gt_file, video_id)
    pred_start, pred_end, pred_labels = get_timestamp_by_video_id(pred_file, video_id)

    if not gt_start:
        return None

    fig = plt.figure(figsize=(22, 4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.02)

    timeline_ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])

    max_time = max(max(gt_end) if gt_end else 1, max(pred_end) if pred_end else 1)
    padding = max_time * 0.01

    timeline_ax.set_xlim(0 - padding, max_time + padding)
    timeline_ax.set_ylim(-1.0, 1.0)
    timeline_ax.spines['top'].set_visible(False)
    timeline_ax.spines['right'].set_visible(False)
    timeline_ax.spines['left'].set_visible(False)
    timeline_ax.set_yticks([])
    timeline_ax.grid(False)

    label_offset = max_time * 0.015
    timeline_ax.text(-label_offset, 0.7, 'Ground truth',
                     fontsize=11, fontweight='bold', ha='right', va='center')
    timeline_ax.text(-label_offset, -0.7, 'Prediction',
                     fontsize=11, fontweight='bold', ha='right', va='center')

    for start, end, label in zip(gt_start, gt_end, gt_labels):
        if label > 0:
            width = end - start
            timeline_ax.add_patch(Rectangle((start, 0.2), width, 0.8,
                                            facecolor=get_color(label), alpha=0.7))

    for start, end, label in zip(pred_start, pred_end, pred_labels):
        if label > 0:
            width = end - start
            timeline_ax.add_patch(Rectangle((start, -1.0), width, 0.8,
                                            facecolor=get_color(label), alpha=0.7))

    if max_time > 60:
        tick_step = 30
    elif max_time > 30:
        tick_step = 10
    else:
        tick_step = 5

    ticks = list(range(0, int(max_time) + 1, tick_step))
    timeline_ax.set_xticks(ticks)
    timeline_ax.set_xticklabels([str(t) for t in ticks], fontsize=9)

    timeline_ax.set_xlabel('Time (seconds)', fontsize=11)
    timeline_ax.set_title(f'{video_name} (Video ID: {video_id})',
                          fontsize=13, fontweight='bold', pad=10)

    legend_ax.axis('off')
    y_pos = 0.95
    y_step = 0.11

    for class_id in range(1, 7):
        color = get_color(class_id)
        label = ACTION_NAMES.get(class_id, f'Class {class_id}')

        legend_ax.add_patch(Rectangle((0.05, y_pos - 0.025), 0.15, 0.05,
                                      facecolor=color, edgecolor='black',
                                      linewidth=0.5, transform=legend_ax.transAxes))

        legend_ax.text(0.22, y_pos, label, fontsize=9, fontweight='bold',
                       va='center', ha='left', transform=legend_ax.transAxes)
        y_pos -= y_step

    save_path = os.path.join(output_dir, "timelines", f"{video_name}_vid_{video_id}_timeline.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    return save_path


# =========================================================
# README
# =========================================================
def save_readme(output_dir, gt_file, pred_file, pred_eval_file, segment_metrics, frame_metrics,
                frame_cm, seg_cm, res_list, os_per_class, generated_timeline_count):
    path = os.path.join(output_dir, "README.txt")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE EVALUATION RESULTS\n")
        f.write(f"Ground Truth File: {gt_file}\n")
        f.write(f"Prediction File (raw): {pred_file}\n")
        f.write(f"Prediction File (used for evaluation): {pred_eval_file}\n")
        f.write(f"Saved Path: {output_dir}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"THRESHOLD (temporal tolerance for matching): {THRESHOLD} seconds\n")
        f.write(f"Apply tolerance on prediction CSV before evaluation: {APPLY_PRED_TOLERANCE}\n")
        f.write(f"Prediction tolerance seconds: {PRED_TOLERANCE_SEC}\n")
       
        f.write("KEY METRICS SUMMARY:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Segment Protocol: {segment_metrics['protocol']}\n")
        f.write(f"Selected Overlap Score: {segment_metrics['overlap_score']:.6f}\n")
        f.write(f"Frame Overall Accuracy: {frame_metrics['overall_accuracy']:.6f}\n")
        f.write(f"Frame Macro F1: {frame_metrics['macro_f1']:.6f}\n\n")

        f.write("SEGMENT-LEVEL DETAILS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Protocol: {segment_metrics['protocol']}\n")
        f.write(f"Mode: {segment_metrics['segment_mode']}\n")
        f.write(f"Precision: {segment_metrics['precision']:.6f}\n")
        f.write(f"Recall: {segment_metrics['recall']:.6f}\n")
        f.write(f"F1-Score: {segment_metrics['f1_score']:.6f}\n")
        f.write(f"Overlap Score: {segment_metrics['overlap_score']:.6f}\n")
        f.write(f"GT Segments: {segment_metrics['total_gt']}\n")
        f.write(f"Pred Segments: {segment_metrics['total_pred']}\n")
        f.write(f"Matched GT: {segment_metrics['matched_gt']}\n")
        f.write(f"Matched Pred: {segment_metrics['matched_pred']}\n")
        f.write(f"TP/FP/FN: {segment_metrics['true_positive']}/{segment_metrics['false_positive']}/{segment_metrics['false_negative']}\n")
        f.write(f"Total IoU Sum: {segment_metrics['total_iou_sum']:.6f}\n")
        f.write(f"Num Positive IoU: {segment_metrics['num_positive_iou']}\n\n")

        f.write("FRAME-LEVEL DETAILS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Frames: {frame_metrics['total_frames']:,}\n")
        f.write(f"Overall Accuracy: {frame_metrics['overall_accuracy']:.6f}\n")
        f.write(f"Macro Accuracy (Classes 1-6): {frame_metrics['macro_accuracy']:.6f}\n")
        f.write(f"Macro Precision/Recall/F1: {frame_metrics['macro_precision']:.6f}/{frame_metrics['macro_recall']:.6f}/{frame_metrics['macro_f1']:.6f}\n")
        f.write(f"Micro Precision/Recall/F1: {frame_metrics['micro_precision']:.6f}/{frame_metrics['micro_recall']:.6f}/{frame_metrics['micro_f1']:.6f}\n\n")

        f.write("OVERLAP SCORE PER CLASS (threshold={}s):\n".format(THRESHOLD))
        f.write("-" * 50 + "\n")
        for class_id in sorted(os_per_class.keys()):
            r = os_per_class[class_id]
            f.write(f"{r['class_name']}: OS/N={r['os_N']:.4f}, P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1_score']:.4f}\n")
        f.write("\n")

        f.write("CONFUSION MATRIX INFO:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Frame-level CM: {frame_cm.shape[0]}x{frame_cm.shape[1]}\n")
        f.write(f"Segment-level CM: {seg_cm.shape[0]}x{seg_cm.shape[1]}\n")
        f.write("Classes: 0=No action, 1=Smoking, 2=Calling, 3=Texting, 4=Drinking, 5=Yawning, 6=Talking\n\n")

        f.write("MULTI-THRESHOLD OVERLAP SCORE (1->10s):\n")
        f.write("-" * 50 + "\n")
        f.write("th OS(N)\n")
        for r1, r3 in res_list:
            f.write(f"{r1['threshold']:>2} | {r3['overlap_score']:.6f}\n")
        f.write("\n")

        f.write(f"Generated timeline files: {generated_timeline_count}\n")

    return path


# =========================================================
# OVERLAP SCORE PER CLASS 
# =========================================================
def calculate_os_per_class_all_classes(gt, pred, thres=10.0, num_classes=7, video_durations=None):
    gt_copy = gt.copy()
    pred_copy = pred.copy()

    if video_durations is not None:
        print(f"  Adding No action segments using video durations...")
        gt_copy = add_no_action_to_dataframe(gt_copy, video_durations)
        pred_copy = add_no_action_to_dataframe(pred_copy, video_durations)
        print(f"    GT after adding No action: {len(gt_copy)} segments")
        print(f"    Pred after adding No action: {len(pred_copy)} segments")

    gt_sorted = gt_copy.sort_values('start').reset_index(drop=True)
    pred_sorted = pred_copy.sort_values('start').reset_index(drop=True)

    all_classes = range(num_classes)
    class_results = {}

    for class_id in all_classes:
        gt_class = gt_sorted[gt_sorted['ID'] == class_id].reset_index(drop=True)
        pred_class = pred_sorted[pred_sorted['ID'] == class_id].reset_index(drop=True)

        M_class = len(gt_class)
        N_class = len(pred_class)

        if M_class == 0 and N_class == 0:
            class_results[class_id] = {
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id],
                'total_gt': 0,
                'total_pred': 0,
                'matched_gt': 0,
                'matched_pred': 0,
                'total_iou_sum': 0.0,
                'num_positive_iou': 0,
                'os_N': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'true_positive': 0,
                'false_positive': 0,
                'false_negative': 0
            }
            continue

        used = set()
        total_iou_sum = 0.0
        num_positive_iou = 0
        matched_gt = 0
        gt_scores_sum_iou = []

        for _, g in gt_class.iterrows():
            vid = g['video_id']
            gs, ge = g['start'], g['stop']

            cands = pred_class[pred_class['video_id'] == vid]
            iou_this_gt = 0.0
            matched_this_gt = []

            for idx, p in cands.iterrows():
                if idx in used:
                    continue

                ps, pe = p['start'], p['stop']
                if abs(ps - gs) <= thres and abs(pe - ge) <= thres:
                    iou = temporal_iou((gs, ge), (ps, pe))
                    if iou > 0:
                        iou_this_gt += iou
                        total_iou_sum += iou
                        num_positive_iou += 1
                        matched_this_gt.append(idx)
                        used.add(idx)

            if matched_this_gt:
                matched_gt += 1
                gt_scores_sum_iou.append(iou_this_gt)
            else:
                gt_scores_sum_iou.append(0.0)

        matched_pred = len(used)

        tp = matched_gt
        fp = N_class - matched_pred
        fn = M_class - matched_gt

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        os3 = total_iou_sum / N_class if N_class > 0 else 0.0

        class_results[class_id] = {
            'class_id': class_id,
            'class_name': CLASS_NAMES[class_id],
            'total_gt': M_class,
            'total_pred': N_class,
            'matched_gt': matched_gt,
            'matched_pred': matched_pred,
            'total_iou_sum': total_iou_sum,
            'num_positive_iou': num_positive_iou,
            'os_N': os3,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positive': tp,
            'false_positive': fp,
            'false_negative': fn
        }

    return class_results


def calculate_os_per_class_all_classes_multi_threshold(gt, pred, thresholds=range(1, 11), num_classes=7):
    all_results = {}
    for th in thresholds:
        os_per_class = calculate_os_per_class_all_classes(gt, pred, thres=th, num_classes=num_classes)
        all_results[th] = os_per_class
    return all_results


# =========================================================
# SAVE FILES - for 7 class
# =========================================================
def save_os_per_class_all_classes_csv(os_per_class, output_dir, threshold=10.0):
    path = os.path.join(output_dir, f"os_per_class_all_classes_thres_{threshold}s.csv")

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'Class ID', 'Class Name',
            'GT', 'Pred', 'Matched GT', 'Matched Pred',
            'OS /N',
            'Precision', 'Recall', 'F1',
            'TP', 'FP', 'FN',
            'Total IoU Sum', 'Num IoU>0'
        ])

        for class_id in sorted(os_per_class.keys()):
            r = os_per_class[class_id]
            w.writerow([
                class_id,
                r['class_name'],
                r['total_gt'],
                r['total_pred'],
                r['matched_gt'],
                r['matched_pred'],
                f"{r['os_N']:.6f}",
                f"{r['precision']:.6f}",
                f"{r['recall']:.6f}",
                f"{r['f1_score']:.6f}",
                r['true_positive'],
                r['false_positive'],
                r['false_negative'],
                f"{r['total_iou_sum']:.6f}",
                r['num_positive_iou']
            ])

    return path


def add_no_action_to_dataframe(df, video_durations):
    all_videos = set(df['video_id'].unique())
    no_action_segments = []

    for video_id in all_videos:
        duration = video_durations.get(video_id)
        if duration is None:
            video_df = df[df['video_id'] == video_id]
            duration = video_df['stop'].max() if len(video_df) else 0

        no_action_segments.extend(get_no_action_segments(df, video_id, duration))

    if no_action_segments:
        no_action_df = pd.DataFrame(no_action_segments)
        df_with_no_action = pd.concat([df, no_action_df], ignore_index=True)
    else:
        df_with_no_action = df.copy()

    return df_with_no_action.sort_values(['video_id', 'start']).reset_index(drop=True)


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 80)
    print("FULL EVALUATION (OS MODE VERSION + PER CLASS)")
    print("=" * 80)

    gt_file = GT_FILE
    pred_file = PRED_FILE
    output_dir = create_output_directory(OUTPUT_BASE_DIR)

    # ======================
    # Load data
    # ======================
    gt = standardize_cols(read_any_csv(gt_file))
    pred_raw = standardize_cols(read_any_csv(pred_file))

    if APPLY_PRED_TOLERANCE:
        print(f"Applying tolerance to prediction CSV: {PRED_TOLERANCE_SEC:.2f}s")
        pred = apply_tolerance_to_prediction_df(pred_raw, tolerance_sec=PRED_TOLERANCE_SEC)
    else:
        pred = pred_raw.copy()

    pred_eval_file = os.path.join(output_dir, "pred_used_for_evaluation.csv")
    save_predictions_csv(pred, pred_eval_file)

    print(f"GT segments                 : {len(gt)}")
    print(f"Pred segments before tol    : {len(pred_raw)}")
    print(f"Pred segments after tol     : {len(pred)}")
    print(f"Saved pred eval csv         : {pred_eval_file}")

    segment_metrics = calculate_segment_metrics_by_mode(
        gt, pred, thres=THRESHOLD, mode=SEGMENT_OS_MODE
    )

    print(f"Segment Protocol     : {segment_metrics['protocol']}")
    print(f"Overlap Score        : {segment_metrics['overlap_score']:.6f}")
    print(f"Precision            : {segment_metrics['precision']:.4f}")
    print(f"Recall               : {segment_metrics['recall']:.4f}")
    print(f"F1-Score             : {segment_metrics['f1_score']:.4f}")
    print(f"TP/FP/FN             : {segment_metrics['true_positive']}/{segment_metrics['false_positive']}/{segment_metrics['false_negative']}")

    # ======================
    # Frame-wise
    # ======================
    print("\n" + "=" * 50)
    print("FRAME-LEVEL METRICS")
    print("=" * 50)

    frame_metrics = compute_framewise_metrics(gt, pred, num_classes=7)
    frame_cm = frame_metrics['_confusion_matrix_raw']
    frame_cm_norm = normalize_confusion_matrix(frame_cm)

    print(f"Frame Overall Accuracy : {frame_metrics['overall_accuracy']:.6f}")
    print(f"Frame Macro Accuracy   : {frame_metrics['macro_accuracy']:.6f}")

    # ======================
    # Segment-wise confusion matrix
    # ======================
    seg_cm = compute_segment_cm(gt, pred, my_video_durations, num_classes=6, thres=THRESHOLD)
    seg_cm_norm = normalize_confusion_matrix(seg_cm)

    # ======================
    # OS per class cho threshold hiện tại
    # ======================
    os_per_class = calculate_os_per_class(gt, pred, thres=THRESHOLD)

    print("\nCalculating OS per class for all thresholds (1-10s)...")
    os_per_class_all = calculate_os_per_class_for_thresholds(gt, pred, thresholds=range(1, 11))

    res_list = []
    for th in range(1, 11):
        r3 = calculate_total_iou_only_N(gt, pred, th)
        r1 = calculate_total_iou_MplusN(gt, pred, th)
        res_list.append((r1, r3))

    prf_csv_path = save_frame_prf_metrics(frame_metrics, output_dir)
    frame_cm_path, frame_cm_norm_path, seg_cm_path, seg_cm_norm_path = save_confusion_matrices(
        frame_cm, frame_cm_norm, seg_cm, seg_cm_norm, output_dir, CLASS_NAMES
    )
    os_csv_path = save_os_multi_threshold_csv(res_list, output_dir)
    os_class_path = save_os_per_class_csv(os_per_class, output_dir, threshold=THRESHOLD)
    os_class_multi_path = save_os_per_class_multi_threshold_csv(os_per_class_all, output_dir)

    # ======================
    # Vẽ plots
    # ======================
    plots_dir = os.path.join(output_dir, "plots")
    plot_cm_heatmap(
        frame_cm, CLASS_NAMES,
        "Frame-level Confusion Matrix (Normalized by Row)",
        os.path.join(plots_dir, "frame_confusion_matrix_normalized.png"),
        normalize=True, cmap='Blues'
    )
    plot_cm_heatmap(
        frame_cm, CLASS_NAMES,
        "Frame-level Confusion Matrix (Raw Counts)",
        os.path.join(plots_dir, "frame_confusion_matrix_raw.png"),
        normalize=False, cmap='Blues'
    )
    plot_cm_heatmap(
        seg_cm, CLASS_NAMES,
        "Segment-level Confusion Matrix (Raw Counts)",
        os.path.join(plots_dir, "segment_confusion_matrix_raw.png"),
        normalize=False, cmap='Oranges'
    )
    plot_cm_heatmap(
        seg_cm_norm, CLASS_NAMES,
        "Segment-level Confusion Matrix (Normalized by Row)",
        os.path.join(plots_dir, "segment_confusion_matrix_normalized.png"),
        normalize=True, cmap='Oranges'
    )

    # ======================
    # Timeline
    # dùng pred_eval_file để timeline phản ánh đúng prediction sau tolerance
    # ======================
    video_ids = gt['video_id'].unique()
    sample_videos = video_ids[:min(16, len(video_ids))]
    timeline_paths = []

    for video_id in sample_videos:
        video_name = f"video_{int(video_id)}"
        p = visualize_timeline(gt_file, pred_eval_file, video_id, video_name, output_dir)
        if p:
            timeline_paths.append(p)

    # ======================
    # README
    # ======================
    readme_path = save_readme(
        output_dir=output_dir,
        gt_file=gt_file,
        pred_file=pred_file,
        pred_eval_file=pred_eval_file,
        segment_metrics=segment_metrics,
        frame_metrics=frame_metrics,
        frame_cm=frame_cm,
        seg_cm=seg_cm,
        res_list=res_list,
        os_per_class=os_per_class,
        generated_timeline_count=len(timeline_paths)
    )

    # ======================
    # 7-class metrics with No action
    # ======================
    os_per_class_all_7 = calculate_os_per_class_all_classes(
        gt, pred,
        thres=THRESHOLD,
        num_classes=7,
        video_durations=my_video_durations
    )

    os_per_class_all_multi = calculate_os_per_class_all_classes_multi_threshold(
        gt, pred, thresholds=range(1, 11), num_classes=7
    )

    os_class_all_path = save_os_per_class_all_classes_csv(
        os_per_class_all_7, output_dir, threshold=THRESHOLD
    )

    # ======================
    # Done
    # ======================
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    print(f"\nKEY RESULTS:")
    print(f"  Segment Protocol    : {segment_metrics['protocol']}")
    print(f"  Overlap Score       : {segment_metrics['overlap_score']:.6f}")
    print(f"  Frame Overall Acc   : {frame_metrics['overall_accuracy']:.6f}")
    print(f"  Frame Macro F1      : {frame_metrics['macro_f1']:.6f}")

    print(f"\nCONFUSION MATRICES:")
    print(f"  Frame-level   : {frame_cm.shape[0]}x{frame_cm.shape[1]}")
    print(f"  Segment-level : {seg_cm.shape[0]}x{seg_cm.shape[1]}")

    print(f"\nFILES:")
    print(f"  Prediction used for evaluation : {pred_eval_file}")
    print(f"  Frame PRF CSV                  : {prf_csv_path}")
    print(f"  Frame CM CSV                   : {frame_cm_path}")
    print(f"  Frame CM Norm CSV              : {frame_cm_norm_path}")
    print(f"  Segment CM CSV                 : {seg_cm_path}")
    print(f"  Segment CM Norm CSV            : {seg_cm_norm_path}")
    print(f"  OS Multi Threshold CSV         : {os_csv_path}")
    print(f"  OS Per Class CSV               : {os_class_path}")
    print(f"  OS Per Class Multi CSV         : {os_class_multi_path}")
    print(f"  OS Per Class 7 Classes CSV     : {os_class_all_path}")

    print(f"\nOutput directory : {output_dir}")
    print(f"Generated plots  : {plots_dir}")
    print(f"Generated timeline files: {len(timeline_paths)}")
    print(f"README           : {readme_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
