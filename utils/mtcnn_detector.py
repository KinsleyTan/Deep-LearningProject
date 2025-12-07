# ~/3DDFA_V2/utils/mtcnn_detector.py
from facenet_pytorch import MTCNN
import numpy as np
import cv2

_mtcnn = None

def get_mtcnn(device='cpu'):
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(keep_all=True, device=device)
    return _mtcnn

def detect(img):
    """
    img: OpenCV BGR image (H,W,3)
    returns: list of [x1,y1,x2,y2,score]
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mtcnn = get_mtcnn(device='cpu')
    boxes, probs = mtcnn.detect(img_rgb)
    if boxes is None:
        return []
    boxes_out = []
    for b, p in zip(boxes, probs):
        if p is None:
            continue
        x1, y1, x2, y2 = map(float, b[:4])
        boxes_out.append([x1, y1, x2, y2, float(p)])
    return boxes_out

class MTCNNDetector:
    def __init__(self):
        pass

    def __call__(self, img):
        return detect(img)
