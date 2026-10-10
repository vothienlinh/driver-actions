# postprocessing/segment_postprocess.py

"""
Temporal segment post-processing.

Bao gồm:
1. Merge các segment cùng nhãn gần nhau
2. Temporal tolerance expansion
3. Final overlap cleanup
"""

from postprocessing.segment_utils import (
    seconds_to_time_str,
    remove_overlapping_segments
)

model_names =['No action', 'Smoking', 'Calling', 'Texting', 'Drinking', 'Yawning', 'Talking']
# =========================================================
# MERGE SAME ACTIONS
# =========================================================
def simple_merge_same_actions(segments, cluster_gap_sec=2.0):
    window_size = 5# 6
    min_main_count = 4 #  5
    debug = True
    min_action_duration = 0.8
    
    # Tạo mapping từ model_names
    if model_names:
        action_to_id = {name: idx for idx, name in enumerate(model_names)}
    else:
        action_to_id = {
            'No action': 0, 'Smoking': 1, 'Calling': 2, 
            'Texting': 3, 'Drinking': 4, 'Yawning': 5, 'Talking': 6
        }
    
    # ==================== ĐẢM BẢO TẤT CẢ SEGMENTS CÓ ACTION_ID ====================
    for s in segments:
        if 'action_id' not in s:
            s['action_id'] = action_to_id.get(s['action'], 0)
    
    # ==================== BƯỚC 1: GỘP THEO GAP ====================
    # Tách No action và Actions
    no_action = []
    actions = []
    
    for s in segments:
        if s['action'] == "No action":
            no_action.append(s)
        else:
            actions.append(s)
    
    # Gộp các action cùng loại theo gap
    clusters = {}
    for s in actions:
        lst = clusters.setdefault(s['action'], [])
        t0, t1 = s['start_time'], s['end_time']
        if not lst or t0 - lst[-1][1] > cluster_gap_sec:
            lst.append([t0, t1, s['action_id']])
        else:
            lst[-1][1] = max(lst[-1][1], t1)
    
    # Tạo merged actions
    merged_actions = []
    for act, cl in clusters.items():
        for st, en, aid in cl:
            merged_actions.append({
                'action': act,
                'action_id': aid,
                'start_time': st,
                'end_time': en,
                'duration_sec': en - st,
                'start_time_str': seconds_to_time_str(st),
                'end_time_str': seconds_to_time_str(en)
            })
    
    # Ghép No action và Actions đã gộp
    final_result = merged_actions + no_action
    final_result.sort(key=lambda x: x['start_time'])
  
   
# ==================== BƯỚC 2: XỬ LÝ CỬA SỔ TRƯỢT ====================

    working_segments = [s.copy() for s in final_result]
    action_indices = [i for i, s in enumerate(working_segments) if s['action'] != "No action"]
    to_replace = set()

    if len(action_indices) >= window_size:
        for i in range(len(action_indices) - window_size + 1):
            window_real_indices = action_indices[i:i + window_size]
            window_segments = [working_segments[idx] for idx in window_real_indices]
            
            first_action = window_segments[0]['action']
            last_action = window_segments[-1]['action']
            
            
            # ==================== ĐIỀU KIỆN 1: Talking ở đầu là noise (ưu tiên cao nhất) ====================
            if first_action == 'Talking':

                
                # Đánh dấu TALKING ở ĐẦU là noise
                to_replace.add(window_real_indices[0])

            
            # ==================== ĐIỀU KIỆN 2: Smoking là chính ====================
            elif first_action == 'Smoking' and last_action == 'Smoking':
                main_action = 'Smoking'
                main_count = sum(1 for seg in window_segments if seg['action'] == main_action)
                
                
                # Đánh dấu tất cả action khác Smoking ở giữa là noise
                for j, seg in enumerate(window_segments):
                    if j == 0 or j == len(window_segments) - 1:
                        continue  # Bỏ qua đầu và cuối (đã là Smoking)
                    
                    if seg['action'] != main_action:
                        to_replace.add(window_real_indices[j])
            
            #============= ĐIỀU KIỆN 3: Calling bị kẹp giữa 2 Texting ====================
            elif len(window_segments) >= 3:
                # Kiểm tra pattern: Texting, Calling, Texting ở 3 vị trí liên tiếp
                for j in range(len(window_segments) - 2):
                    if (window_segments[j]['action'] == 'Texting' and 
                        window_segments[j+1]['action'] == 'Calling' and 
                        window_segments[j+2]['action'] == 'Texting'):
                        
                        # Đánh dấu 2 Texting ở 2 bên là noise
                        to_replace.add(window_real_indices[j])      # Texting bên trái
                        to_replace.add(window_real_indices[j+2])    # Texting bên phải
                        
            
            # ==================== ĐIỀU KIỆN 4: Các trường hợp khác (giữ logic cũ) ====================
            elif first_action == last_action and first_action != "No action":
                main_action = first_action
                main_count = sum(1 for seg in window_segments if seg['action'] == main_action)
                
                
                if main_count >= min_main_count:
                    for j, seg in enumerate(window_segments):
                        if j == 0 or j == len(window_segments) - 1:
                            continue
                        
                        if seg['action'] != main_action:
                            to_replace.add(window_real_indices[j])

    # Thực hiện thay thế noise bằng No action
    for idx in to_replace:
        old_action = working_segments[idx]['action']
        working_segments[idx]['action'] = 'No action'
        working_segments[idx]['action_id'] = action_to_id['No action']
       
    # ==================== BƯỚC 3: GỘP NO ACTION LIÊN TIẾP ====================
    
    merged_no_action = []
    i = 0
    while i < len(working_segments):
        if working_segments[i]['action'] == 'No action':
            start = working_segments[i]['start_time']
            end = working_segments[i]['end_time']
            count = 1
            j = i + 1
            while j < len(working_segments) and working_segments[j]['action'] == 'No action':
                end = max(end, working_segments[j]['end_time'])
                count += 1
                j += 1
            
            merged_no_action.append({
                'action': 'No action',
                'action_id': action_to_id['No action'],
                'start_time': start,
                'end_time': end,
                'duration_sec': end - start,
                'start_time_str': seconds_to_time_str(start),
                'end_time_str': seconds_to_time_str(end)
            })
            i = j
        else:
            merged_no_action.append(working_segments[i])
            i += 1

    # ==================== BƯỚC 4: LỌC SEGMENTS NGẮN ====================
    
    filtered_result = []
    removed_count = 0
    
    for seg in merged_no_action:
        if seg['action'] != 'No action' and seg['duration_sec'] < min_action_duration:
            removed_count += 1

        else:
            filtered_result.append(seg)
    
    return filtered_result

# =========================================================
# TEMPORAL TOLERANCE
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
            sc['end_time_str']   = seconds_to_time_str(sc['end_time'])

        expanded.append(sc)

    # giữ nguyên logic xử lý No action như bạn đang làm
    final = []
    i = 0
    while i < len(expanded):
        cur = expanded[i]
        if cur['action'] == 'No action':
            prev_end, next_start = None, None
            for j in range(i-1, -1, -1):
                if expanded[j]['action'] != 'No action':
                    prev_end = expanded[j]['end_time']
                    break
            for j in range(i+1, len(expanded)):
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
                cur['end_time']   = ne
                cur['duration_sec'] = ne - ns
                cur['start_time_str'] = seconds_to_time_str(ns)
                cur['end_time_str']   = seconds_to_time_str(ne)
                final.append(cur)
        else:
            final.append(cur)
        i += 1

    return sorted(final, key=lambda x: x['start_time'])

class TrackWrap:
    def __init__(self,bbox,tid,cls_id):
        self._bbox=bbox
        self.track_id=tid
        self.class_id = cls_id
    def is_confirmed(self): return True
    def to_tlbr(self): return self._bbox
def update_and_get_semantic_label(
    yolo_labels,          # list[str] từ YOLO (đã ROI)
    driver_track_id,      # track_id của driver
    frame_idx,
    fps
):
    """
    Trả về labels để đưa vào fuse_driver_labels_strict
    """
    if driver_track_id is None:
        return []

    # === CASE 1: YOLO có label ===
    if yolo_labels:
        semantic_memory[driver_track_id] = {
            "label": yolo_labels[0],     # label chính
            "last_seen": frame_idx
        }
        return yolo_labels

    # === CASE 2: YOLO mất detection ===
    mem = semantic_memory.get(driver_track_id)
    if mem:
        age = frame_idx - mem["last_seen"]
        max_age = int(SEMANTIC_HOLD_SEC * fps)
        if age <= 4: # max_age=30
            return [mem["label"]]        # semantic hold

    return []  # No action

