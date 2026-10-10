# models/detection/yolov.py

"""
YOLO detection module
Giữ logic từ code gốc nhưng tách thành module riêng.
"""

import numpy as np
from ultralytics import YOLO

from configs.ablation_config import (
    YOLO_PTQ_MODEL_PATH,
    YOLO_IMGSZ,
    YOLO_BASE_CONF,
    CLASS_CONF_THRESH,
    CUSTOM_NAMES,
    DEVICE
)


class YOLODetector:
    """
    Wrapper cho YOLO model.
    """

    def __init__(
        self,
        model_path=YOLO_MODEL_PATH,
        device=DEVICE,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_BASE_CONF,
    ):

        self.model = YOLO(model_path)

        self.device = device
        self.imgsz = imgsz
        self.base_conf = conf

        #print("YOLO original classes:", self.model.names)

        # Override class names giống code gốc
        self.model.model.names = CUSTOM_NAMES

        print("Using custom classes:", CUSTOM_NAMES)

    # =========================================================
    # RAW YOLO PREDICTION
    # =========================================================

    def predict(self, frame):

        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.base_conf,
            device=self.device,
            verbose=False,
        )

        return results

    # =========================================================
    # FILTERED DETECTIONS
    # =========================================================
    def get_raw_detections(self, frame):
        results = self.predict(frame)
        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.int().cpu().numpy()

        for b, c, k in zip(boxes, confs, clss):
            class_name = self.model.model.names[int(k)].strip()
            detections.append({
                "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "conf": float(c),
                "class_id": int(k),
                "class_name": class_name,
            })
        return detections
    def get_filtered_detections(self, frame, skip_class=None):
        """
        Trả về danh sách detection sau khi lọc threshold theo class.
        """

        results = self.predict(frame)

        detections = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.int().cpu().numpy()

        for b, c, k in zip(boxes, confs, clss):

            class_name = self.model.model.names[int(k)].strip()

            # threshold riêng từng class
            conf_thresh = CLASS_CONF_THRESH.get(class_name, self.base_conf)

            if c < conf_thresh:
                continue

            if skip_class and class_name == skip_class:
                continue

            detections.append(
                {
                    "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "conf": float(c),
                    "class_id": int(k),
                    "class_name": class_name,
                }
            )

        return detections

    # =========================================================
    # FORMAT FOR DEEPSORT
    # =========================================================

    def get_deepsort_inputs(self, frame, skip_class=None):
        """
        Format detection cho DeepSORT.
        Output:
        ([x, y, w, h], conf, class_id)
        """

        detections = self.get_filtered_detections(frame, skip_class)

        outputs = []

        for d in detections:

            x1, y1, x2, y2 = d["bbox"]

            w = x2 - x1
            h = y2 - y1

            outputs.append(([x1, y1, w, h], d["conf"], d["class_id"]))

        return outputs, detections

    # =========================================================
    # FORMAT FOR SORT
    # =========================================================

    def get_sort_inputs(self, frame, skip_class=None):
        """
        Format detection cho SORT.
        Output: numpy array Nx4
        """

        detections = self.get_filtered_detections(frame, skip_class)

        boxes = [d["bbox"] for d in detections]

        if len(boxes) == 0:
            return np.empty((0, 4)),[]

        return np.array(boxes),detections

    # =========================================================
    # SORT + CLASS IDS
    # =========================================================

    def get_sort_inputs_with_classes(self, frame, skip_class=None):
        """
        SORT + class_id mapping.
        """

        detections = self.get_filtered_detections(frame, skip_class)

        boxes = []
        class_ids = []

        for d in detections:
            boxes.append(d["bbox"])
            class_ids.append(d["class_id"])

        if len(boxes) == 0:
            dets = np.empty((0, 4))
        else:
            dets = np.array(boxes)

        return dets, class_ids
