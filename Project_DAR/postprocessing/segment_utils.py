# postprocessing/segment_utils.py

"""
Utilities for temporal segment processing.

Bao gồm:
1. Frame labels → segments
2. Remove overlapping segments
3. Convert seconds → time string
"""

# =========================================================
# TIME FORMAT
# =========================================================

def seconds_to_time_str(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.xxx format.
    """

    h = int(seconds // 3600)

    m = int((seconds % 3600) // 60)

    s = seconds % 60

    return f"{h:02d}:{m:02d}:{s:06.3f}"


# =========================================================
# FRAME LABELS → SEGMENTS
# =========================================================
def extract_segments_from_labels1(
    time_labels,
    min_segment_duration=0.5,
    use_action_specific_duration=True,  # THÊM: bật/tắt adaptive duration
    action_min_duration=None  # THÊM: dict chứa min duration theo action
):
   
    # Action-specific min duration (nếu không được cung cấp)
    if action_min_duration is None:
        action_min_duration = {
            "Smoking": 0.5,    # Hút thuốc: động tác hít/thở tối thiểu 0.5s
            "Calling": 1.0,     # Gọi điện: cần thời gian áp tai
            "Texting": 2.5,     # Nhắn tin: gõ chữ cần thời gian
            "Drinking": 0.5,    # Uống nước: động tác nhanh
            "Yawning": 0.5,     # Ngáp: cử động hàm
            "Talking": 0.5,     # Nói chuyện: phát âm
            "No action": 0.0    # Không áp dụng
        }
    
    segments = []
    
    if not time_labels:
        return segments
    
    time_labels = sorted(time_labels, key=lambda x: x["time"])
    
    current_label = time_labels[0]["label"]
    start_time = time_labels[0]["time"]
    last_time = start_time
    
    for i in range(1, len(time_labels)):
        t = time_labels[i]["time"]
        label = time_labels[i]["label"]
        
        if label != current_label:
            end_time = last_time
            duration = end_time - start_time
            
            # Xác định min_duration cho action hiện tại
            if use_action_specific_duration:
                min_dur = action_min_duration.get(current_label, min_segment_duration)
            else:
                min_dur = min_segment_duration
            
            # Chỉ giữ segment nếu duration >= min_dur
            if duration >= min_dur:
                segments.append({
                    "action": current_label,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_sec": duration,
                    "min_duration_used": min_dur,  # THÊM: để debug
                    "start_time_str": seconds_to_time_str(start_time),
                    "end_time_str": seconds_to_time_str(end_time)
                })
            else:
                # Debug: có thể log các segment bị loại
                # print(f"Removed {current_label}: {duration:.2f}s < {min_dur:.2f}s")
                pass
            
            start_time = t
            current_label = label
        
        last_time = t
    
    # Xử lý segment cuối cùng
    end_time = last_time
    duration = end_time - start_time
    
    if use_action_specific_duration:
        min_dur = action_min_duration.get(current_label, min_segment_duration)
    else:
        min_dur = min_segment_duration
    
    if duration >= min_dur:
        segments.append({
            "action": current_label,
            "start_time": start_time,
            "end_time": end_time,
            "duration_sec": duration,
            "min_duration_used": min_dur,
            "start_time_str": seconds_to_time_str(start_time),
            "end_time_str": seconds_to_time_str(end_time)
        })
    
    return segments
    
def extract_segments_from_labels(
    time_labels,
    min_segment_duration=0.5,
    use_action_specific_duration=False, # THÊM: bật/tắt adaptive duration
    action_min_duration=None, # THÊM: dict chứa min duration theo action
):
    skip_initial_seconds=11.0 # THÊM: loại bỏ tất cả labels trong khoảng 0s đến X giây đầu video (set = 0 để tắt)
    # Action-specific min duration (nếu không được cung cấp)
    if action_min_duration is None:
        action_min_duration = {
            "Smoking": 0.5,
            "Calling": 1.0,
            "Texting": 2.5,
            "Drinking": 0.5,
            "Yawning": 0.5,
            "Talking": 0.5,
            "No action": 0.0
        }
   
    # ==================== THÊM: Loại bỏ labels trong 11 giây đầu video TRƯỚC khi extract ====================
    if skip_initial_seconds > 0 and time_labels:
        time_labels = [entry for entry in time_labels if entry["time"] >= skip_initial_seconds]
    # ==============================================================================================
   
    segments = []
   
    if not time_labels:           # Sau khi lọc có thể rỗng
        return segments
   
    time_labels = sorted(time_labels, key=lambda x: x["time"])  # vẫn sort lại để chắc chắn
   
    current_label = time_labels[0]["label"]
    start_time = time_labels[0]["time"]
    last_time = start_time
   
    for i in range(1, len(time_labels)):
        t = time_labels[i]["time"]
        label = time_labels[i]["label"]
       
        if label != current_label:
            end_time = last_time
            duration = end_time - start_time
           
            # Xác định min_duration cho action hiện tại
            if use_action_specific_duration:
                min_dur = action_min_duration.get(current_label, min_segment_duration)
            else:
                min_dur = min_segment_duration
           
            if duration >= min_dur:
                segments.append({
                    "action": current_label,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_sec": duration,
                    "min_duration_used": min_dur,
                    "start_time_str": seconds_to_time_str(start_time),
                    "end_time_str": seconds_to_time_str(end_time)
                })
           
            start_time = t
            current_label = label
       
        last_time = t
   
    # Xử lý segment cuối cùng
    end_time = last_time
    duration = end_time - start_time
   
    if use_action_specific_duration:
        min_dur = action_min_duration.get(current_label, min_segment_duration)
    else:
        min_dur = min_segment_duration
   
    if duration >= min_dur:
        segments.append({
            "action": current_label,
            "start_time": start_time,
            "end_time": end_time,
            "duration_sec": duration,
            "min_duration_used": min_dur,
            "start_time_str": seconds_to_time_str(start_time),
            "end_time_str": seconds_to_time_str(end_time)
        })
   
    return segments


# =========================================================
# REMOVE OVERLAPPING SEGMENTS
# =========================================================

def remove_overlapping_segments(segments):
    """
    Remove temporal overlaps giữa các segments.
    """

    if len(segments) <= 1:
        return segments

    segments = sorted(segments, key=lambda x: x["start_time"])

    cleaned = []

    current = segments[0].copy()

    for i in range(1, len(segments)):

        nxt = segments[i]

        if nxt["start_time"] < current["end_time"]:

            current["end_time"] = nxt["start_time"]

            current["duration_sec"] = current["end_time"] - current["start_time"]

            current["end_time_str"] = seconds_to_time_str(current["end_time"])

            if current["duration_sec"] >= 0.1:
                cleaned.append(current)

            current = nxt.copy()

        else:

            if current["duration_sec"] >= 0.1:
                cleaned.append(current)

            current = nxt.copy()

    if current["duration_sec"] >= 0.1:
        cleaned.append(current)

    return cleaned
