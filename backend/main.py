import torch
from models.expression_head import ExpressionHead
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
    allow_origins=["*"],
    allow_credentials=False,
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
# Expression Head
# =========================
DEVICE = "cpu"  # or "cuda" if deployed with GPU

expr_model = None
@app.on_event("startup")
def load_model():
    global expr_model
    print("Loading ExpressionHead model...")
    expr_model = ExpressionHead().to(DEVICE)
    expr_model.load_state_dict(
        torch.load("weights/expression_head.pth", map_location=DEVICE)
    )
    expr_model.eval()
    print("Model loaded")

# =========================
# Health Check
# =========================
@app.get("/")
def root():
    return {"status": "MediaPipe FaceMesh backend running"}

def landmarks_to_tensor(landmarks):
    lm = []
    for p in landmarks:
        lm.extend([p.x, p.y, p.z])
    return torch.tensor(lm, dtype=torch.float32).unsqueeze(0)

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    if expr_model is None:
        return {"error": "Model not loaded yet"}
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_face.process(rgb)

    if not result.multi_face_landmarks:
        return {"landmarks": [], "expression": []}

    face_landmarks = result.multi_face_landmarks[0].landmark

    # -------- Landmarks --------
    landmarks = [
        {"x": float(p.x), "y": float(p.y), "z": float(p.z)}
        for p in face_landmarks
    ]

    # -------- Expression inference --------
    lm_tensor = landmarks_to_tensor(face_landmarks).to(DEVICE)

    with torch.no_grad():
        expr = expr_model(lm_tensor).cpu().numpy()[0]

    return {
        # "num_landmarks": len(landmarks),
        "landmarks": landmarks,
        "expression": expr.tolist()  # 10D
    }

@app.get("/debug")
def debug():
    return {
        "landmarks": [
            {"x": 0.1, "y": 0.1, "z": 0.0},
            {"x": 0.9, "y": 0.1, "z": 0.0},
            {"x": 0.5, "y": 0.9, "z": 0.0},
        ]
    }