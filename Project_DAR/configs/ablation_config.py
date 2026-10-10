#Common
# configs/ablation_config.py
"""
Configuration for ablation study

"""

import os
import torch

# =========================================================
# ROOT PATHS
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# =========================================================
# WEIGHT PATHS
# =========================================================
YOLO_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "weights",
    "yolo",
    "bestv11n_250725.pt"
)

LSTM_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "weights",
    "lstm",
    "best_lstm32_200825.pth"
)

# =========================================================
# DATA PATHS
# =========================================================
VIDEO_DIR = r"/media/linhvt/febd2ef4-e940-4722-b949-df092e2ee637/Du lieu sao luu 01032026/LSTM/dataset_200825/testclips"
#VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "testclips")


# SORT root 
SORT_ROOT = r"/home/linhvt/Documents/Yolo_sort_LSTM"

# =========================================================
# OUTPUT ROOT
# =========================================================
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "Outputs")
OUTPUT_DIR = OUTPUTS_ROOT
# =========================================================
# DEVICE
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# YOLO SETTINGS
# =========================================================
YOLO_IMGSZ = 640
YOLO_BASE_CONF = 0.7

# Custom class names 
CUSTOM_NAMES = {
    0: "Smoking",
    1: "Calling",
    2: "Texting",
    3: "Drinking",
    4: "Yawning",
    5: "Talking",
    6: "Cigarette",
    7: "Phone",
    8: "Bottle",
    9: "Open mouth",
}


CLASS_CONF_THRESH_2 = {
    "Smoking": 0.5,
    "Cigarette": 0.7,
    "Calling": 0.7,
    "Texting": 0.7,
    "Drinking": 0.7,
    "Yawning": 0.5,
    "Talking": 0.7,
    "Phone": 0.7,
    "Bottle": 0.7,
    "Open mouth": 0.7,
}

CLASS_CONF_THRESH = {
    "Smoking": 0.5,
    "Cigarette": 0.5,
    "Calling": 0.7,
    "Texting": 0.7,
    "Drinking": 0.5,
    "Yawning": 0.7,
    "Talking": 0.6,
    "Phone": 0.7,
    "Bottle": 0.5,
    "Open mouth": 0.8,
}

SKIP_NHANTIN_IN_TRACKING = False

# =========================================================
# ACTION CLASSES
# =========================================================
MODEL_NAMES = [
    "No action",
    "Smoking",
    "Calling",
    "Texting",
    "Drinking",
    "Yawning",
    "Talking",
]

ACTION_ID_MAP = {name: idx for idx, name in enumerate(MODEL_NAMES)}
ID2ACTION = {idx: name for name, idx in ACTION_ID_MAP.items()}
NUM_CLASSES = len(MODEL_NAMES)

# =========================================================
# LSTM SETTINGS
# =========================================================
WINDOW_SIZE = 32
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2

# =========================================================
# TEMPORAL SETTINGS
# =========================================================
TIME_TOLERANCE = 1.0           # giây
MIN_SEGMENT_DURATION = 0.5     # giây được giữ lại
CLUSTER_GAP_SEC = 3.0            # giây tối đa cho phép gộp


DECISION_MODE = "argmax"

# Chỉ dùng khi DECISION_MODE = mean_softmax / majority_vote
DECISION_WINDOW = 3

# =========================================================
# SEMANTIC HOLD
# =========================================================
SEMANTIC_HOLD_SEC = 0.5
SEMANTIC_HOLD_MAX_AGE_OVERRIDE = 10 #5
FPS = 30
# =========================================================
# RULE FUSION
# =========================================================

FUSION_THRESHOLD_FRAMES = 25

# =========================================================
# TRACKER SETTINGS - SORT
# =========================================================
SORT_MAX_AGE = 20
SORT_MIN_HITS = 3
SORT_IOU_THRESHOLD = 0.3

# =========================================================
DRIVER_OVERLAP_THRESH = 0.1
# =========================================================
# VIDEO ID MAP
# =========================================================
VIDEO_ID_MAP = {
    "01_011_01_0_front": 111,
    "01_011_01_0_rear": 112,
    "01_012_01_0_front": 121,
    "01_012_01_0_rear": 122,
    "01_013_01_0_front": 131,
    "01_013_01_0_rear": 132,
    "01_014_01_0_front": 141,
    "01_014_01_0_rear": 142,
    "01_015_01_0_front": 151,
    "01_015_01_0_rear": 152,
    "01_016_01_0_front": 161,
    "01_016_01_0_rear": 162,
    "01_017_01_0_front": 171,
    "01_017_01_0_rear": 172,
    "01_018_01_0_front": 181,
    "01_018_01_0_rear": 182,
}

# =========================================================
# EVALUATION FILES
# =========================================================
GT_FILE = r"/home/vtlinh/Documents/GT_new.csv"
#GT_FILE = os.path.join(PROJECT_ROOT, "data", "gt.csv")
EVAL_OUTPUT_BASE_DIR = r"/home/vtlinh/Documents/3s"

EVAL_THRESHOLD = 3.0 #defaut
SEGMENT_OS_MODE = "N"   # 

MY_VIDEO_DURATIONS = {
    111: 319, 112: 319,
    121: 325, 122: 325,
    131: 273, 132: 273,
    141: 315, 142: 315,
    151: 263, 152: 263,
    161: 287, 162: 287,
    171: 279, 172: 279,
    181: 335, 182: 335,
}

CLASS_NAMES_EVAL = [
    "No action",
    "Smoking",
    "Calling",
    "Texting",
    "Drinking",
    "Yawning",
    "Talking",
]

# =========================================================
# ABLATION RUN SETTINGS
# =========================================================
# Options:
# - "YOLO_ONLY"
# - "YOLO_LSTM"
# - "YOLO_SORT_LSTM"
# - "YOLO_DEEPSORT_LSTM"
# - "all"
DEFAULT_PIPELINE = "all"

# Chế độ chạy
RUN_INFERENCE = True
RUN_VISUALIZATION = True

# =========================================================
# OUTPUT FOLDER NAMES
# =========================================================
PIPELINE_OUTPUT_NAMES = {
    "YOLO_ONLY": "YOLO_ONLY",
    "YOLO_LSTM": "YOLO_LSTM",
    "YOLO_SORT_LSTM": "YOLO_SORT_LSTM",
    "YOLO_DEEPSORT_LSTM": "YOLO_DEEPSORT_LSTM",
}

INFERENCE_OUTPUT_SUBDIR = "inference_and_save_csv"
VISUAL_OUTPUT_SUBDIR = "visualize_videos_from_csv"

# =========================================================
# HELPERS
# =========================================================
def get_pipeline_output_dir(pipeline_name: str) -> str:
    return os.path.join(
        OUTPUTS_ROOT,
        PIPELINE_OUTPUT_NAMES[pipeline_name]
    )


def get_inference_output_dir(pipeline_name: str) -> str:
    return os.path.join(
        get_pipeline_output_dir(pipeline_name),
        INFERENCE_OUTPUT_SUBDIR
    )


def get_visual_output_dir(pipeline_name: str) -> str:
    return os.path.join(
        get_pipeline_output_dir(pipeline_name),
        VISUAL_OUTPUT_SUBDIR
    )
    
##
