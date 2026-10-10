#README.md

# Ablation Study: YOLO + LSTM with tracking methods
---

# Driver Action Recognition – Ablation Study

This repository implements an ablation study for **driver action recognition** using different combinations of detection, tracking, and temporal modeling.

The goal is to analyze the impact of:

- YOLO detection
- temporal modeling with LSTM
- tracking (SORT / DeepSORT)
- post-processing strategies

The experiments follow the **AI City Challenge Track 3: Naturalistic Driving Action Recognition** evaluation protocol.

---
# 1. Project Structure
ablation_study/
├── configs/
│   └── ablation_config.py
│
├── weights/
│ ├── yolo/
│ │ └── bestv11n.pt
│ └── lstm/
│ └── best_lstm64.pth
│
├── models/
│   ├── detection/
│   │   └── yolov11.py
│   │
│   ├── tracking/
│   │   ├── sort_tracker.py
│   │   
│   │
│   ├── temporal/
│   │   └── lstm_model.py
│   │
│   ├── fusion/
│   │   ├── semantic_hold.py
│   │   └── label_fusion.py
│   │
│   └── pipelines/
│       ├── yolo_sort_lstm.py
│      
│
├── postprocessing/
│   ├── decision.py
│   ├── segment_utils.py
│   └── segment_postprocess.py
│
├── evaluation/
│   └── convert_gt_to_frame.py
│   └── evaluate_continuous.py
│
├── run_ablation.py
├── visualize_videos_from_csv.py
│
├── Outputs/
│   ├── YOLO_SORT_LSTM/
│       ├── inference_and_save_csv/
│       └── visualize_videos_from_csv/
└── README.md

---

# 2. Pipeline
Pipeline is implemented for ablation analysis.

| Pipeline               | Description                                   |
|------------------------|-----------------------------------------------|
| YOLO + SORT + LSTM     |Detection + tracking + temporal modeling       |


Pipeline flow example:

### YOLO + SORT + LSTM
```
YOLO
→ SORT tracking
→ semantic hold (track-based)
→ rule-based fusion
→ one-hot encoding
→ LSTM
→ decision
→ segment extraction
→ segment post-processing
→ CSV output
````
---
# 3. Post-processing

Three main post-processing steps are applied after LSTM inference.

### 3.1. Linking

Convert frame-wise predictions into continuous action segments.
```
extract_segments_from_labels()

```
### 3.2. Merging

Merge fragmented segments with the same label.

```
simple_merge_same_actions()

```

### 3.3. Temporal tolerance

Expand predicted segments to compensate timing errors.

```
apply_tolerance_to_segments()
```
---
# 4. Installation
## Python environment
```
python >= 3.12
```
Install dependencies:
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install numpy pandas
```
```
# 5. Model Weights
Place model weights in the **weights folder**:
```
weights/
    bestv11n_250725.pt
    best_lstm32_200825.pth
```
Paths are configured in:
```
configs/ablation_config.py
```
---
# 6. Running Inference

Run pipeline on the full dataset.

### YOLO + SORT + LSTM

python run_ablation.py --pipeline yolo_sort_lstm

### YOLO + SORT + LSTM (PTQ)

python run_ablation.py --pipeline yolo_sort_lstm_PTQ

### Run all pipelines

python run_ablation.py --pipeline all

---

# 7. Output Structure

Results are stored in:

```
Outputs/
```
Example:
```
Outputs/
YOLO_ONLY/
    inference_and_save_csv/
        111.csv
        112.csv
        ...
    all_predictions.csv
YOLO_LSTM/
    inference_and_save_csv/
    all_predictions.csv

YOLO_SORT_LSTM/
    inference_and_save_csv/
    all_predictions.csv
YOLO_DEEPSORT_LSTM/
    inference_and_save_csv/
    all_predictions.csv
```
CSV format (continuous level):
```
video_id ID start stop
111      1  10.2 10.2
111      6  20.4 22.1
```
---
# 9. Evaluation
Metrics include:
* Overlap Score (OS)
* Frame-wise Accuracy
* Precision / Recall
* F1-score
Run:
python evaluation/evaluation_from_csv.py
---
# 10. Dataset Format

Videos follow the UTC naming convention:

```
01_011_01_0_front.mp4
01_011_01_0_rear.mp4
```
Video ID mapping is defined in:
```
configs/ablation_config.py
```
---
# 11. Actions

The system recognizes 7 driver actions:

| ID | Action    |
| -- | --------- |
| 0  | No action |
| 1  | Smoking   |
| 2  | Calling   |
| 3  | Texting   |
| 4  | Drinking  |
| 5  | Yawning   |
| 6  | Talking   |



