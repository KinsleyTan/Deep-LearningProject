import cv2, numpy as np
from insightface import model_zoo

# Lazy-load model
_MODEL = None

def get_model(model_name='scrfd_2.5g_kps'):
    global _MODEL
    if _MODEL is None:
        _MODEL = model_zoo.get_model(model_name)  # model is downloaded on first call
        _MODEL.prepare(ctx_id=-1, nms=0.4)       # ctx_id=-1 -> CPU; set >=0 for GPU
    return _MODEL

def detect(img, model_name='scrfd_2.5g_kps', threshold=0.4, max_det=10):
    """
    img: BGR image (numpy)
    returns: list of boxes [x1,y1,x2,y2,score]
    """
    model = get_model(model_name)
    # model expects BGR
    dets = model.detect(img, threshold=threshold, max_num=max_det)[0]
    # dets: Nx(5 or 15...) -> first 5 are [x1,y1,x2,y2,score], optionally kps
    if dets is None or len(dets)==0:
        return []
    boxes = []
    for d in dets:
        x1,y1,x2,y2,score = float(d[0]),float(d[1]),float(d[2]),float(d[3]),float(d[4])
        boxes.append([x1,y1,x2,y2,score])
    return boxes

# convenience: keep API similar to FaceBoxes.FaceBoxes __call__
class SCRFDDetector:
    def __init__(self, model_name='scrfd_2.5g_kps', threshold=0.4):
        self.model_name = model_name
        self.threshold = threshold

    def __call__(self, img):
        return detect(img, model_name=self.model_name, threshold=self.threshold)
