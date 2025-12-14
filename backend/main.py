from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp

# =========================
# App
# =========================
app = FastAPI()

# Allow frontend (Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MediaPipe FaceMesh
# =========================
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# =========================
# Health Check
# =========================
@app.get("/")
def root():
    return {"status": "MediaPipe FaceMesh backend running"}

# =========================
# Inference Endpoint
# =========================
@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # Read image
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return {"landmarks": []}

    # BGR -> RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # FaceMesh
    result = mp_face.process(rgb)

    if not result.multi_face_landmarks:
        return {"landmarks": []}

    face_landmarks = result.multi_face_landmarks[0]

    # Extract landmarks
    landmarks = [
        {
            "x": float(p.x),
            "y": float(p.y),
            "z": float(p.z)
        }
        for p in face_landmarks.landmark
    ]

    return {
        "num_landmarks": len(landmarks),
        "landmarks": landmarks
    }
