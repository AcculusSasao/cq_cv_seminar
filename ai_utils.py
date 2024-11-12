# MIT License : Copyright (c) 2024 Yukiyoshi Sasao
import numpy as np
import cv2
import time
from typing import Tuple

POSE17_KEYPOINTS_STR = (
    "nose",         # 0
    "leftEye",      # 1
    "rightEye",     # 2
    "leftEar",      # 3
    "rightEar",     # 4
    "leftShoulder", # 5
    "rightShoulder",# 6
    "leftElbow",    # 7
    "rightElbow",   # 8
    "leftWrist",    # 9
    "rightWrist",   # 10
    "leftHip",      # 11
    "rightHip",     # 12
    "leftKnee",     # 13
    "rightKnee",    # 14
    "leftAnkle",    # 15
    "rightAnkle"    # 16
)
POSE17_JOINTS = (
    (0, 5),
    (0, 6),
    (1, 2),
    (3, 1),
    (4, 2),
    (11, 5),
    (7, 5),
    (7, 9),
    (11, 13),
    (13, 15),
    (12, 6),
    (8, 6),
    (8, 10),
    (12, 14),
    (14, 16),
    (5, 6),
    (11, 12),
)

COCO_CATEGORY = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

def normalize_image(bgr_img: np.array, target_width: int, target_height: int, dtype: int,
                    mean: float = 0, std : float = 1, quantization = None, 
                    b_swap_rb: bool = True, b_show_resized_image: bool = False) -> np.array:
    if target_width is None:
        img = bgr_img
    else:
        if target_width != target_height:
            raise ValueError('target_width must be same with target_height.')
        h, w = bgr_img.shape[:2]
        if h <= w:
            srcimg = np.zeros((w, w, 3), dtype=np.uint8)
            e = (w - h) // 2
            srcimg[e:e+h] = bgr_img
            offset = [e, 0]
        else:
            srcimg = np.zeros((h, h, 3), dtype=np.uint8)
            e = (h - w) // 2
            srcimg[:, e:e+w] = bgr_img
            offset = [0, e]
        img = cv2.resize(srcimg, dsize=(target_width, target_height))
    if b_swap_rb:
        data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        data = img
    if b_show_resized_image:
        cv2.imshow('resized image', data)
    data = ((data - mean) / std)
    if quantization is not None:
        sc, zp = quantization
        if sc != 0:
            data = data / sc + zp
    if dtype == np.int8:
        data = data.clip(-128, 127)
    elif dtype == np.uint8:
        data = data.clip(0, 255)
    data = data.astype(dtype)
    if target_width is None:
        scale = 1
        offset = np.zeros((2))
    else:
        scale = img.shape[0] / srcimg.shape[0]
        offset = np.array(offset)
    return data[np.newaxis, :], scale, offset

class TimeManager:
    def __init__(self):
        self.pre = None
        self.mean = 0
        self.num = 0
        self.last_period = None
        pass
    def measure(self) -> Tuple[float, float]:
        cur = time.perf_counter()
        period = None
        if self.pre is not None:
            period = cur - self.pre
            # update mean
            self.mean = (self.mean * self.num + period) / (self.num + 1)
            self.num += 1
        self.last_period = period
        self.pre = cur
        return cur, period
    def draw(self, img: np.array, pos: Tuple[int, int] = (0, 20)) -> None:
        if self.last_period is None:
            return
        fps = 1 / self.mean
        string = '{:.2f} msec, FPS {:.2f}'.format(self.last_period * 1000, fps)
        cv2.putText(img, string, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
    def get_info(self) -> str:
        if self.last_period is None:
            return ''
        fps = 1 / self.mean
        string = '{:.2f} msec, FPS {:.2f}'.format(self.last_period * 1000, fps)
        return string
