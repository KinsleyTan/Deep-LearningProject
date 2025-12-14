import mediapipe as mp
import numpy as np
import cv2

mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

def extract_landmarks(img):
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)

    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[p.x*w, p.y*h, p.z] for p in lm]).reshape(-1)
    return pts
