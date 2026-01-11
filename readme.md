# 3DDFA_v2 Expression Modeling with MediaPipe FaceMesh

This repository is a **research and application project** built on top of  
**3DDFA_v2 (TDDFA)**, extending it with **facial expression modeling** and a **real-time web-based visualization system**.

> **Original 3DDFA_v2 repository:**  
> https://github.com/cleardusk/3DDFA_v2  
>
> All core 3D face alignment, rendering, and Basel Face Model (BFM) components are credited to the original authors.

---

## 📌 Research Overview

The objective of this project is to extend **3DDFA_v2**, which primarily focuses on
3D face pose and shape estimation, by introducing **learned facial expression parameters**
that enable **interactive facial animation in a web environment**.

This work combines **classical 3D face alignment**, **deep learning–based expression modeling**,
and **modern frontend visualization**.

---

## 🔬 Methodology and Contributions

### 1. Landmark Extraction
- Uses **MediaPipe FaceMesh** to obtain dense and stable 2D facial landmarks.
- These landmarks replace or complement traditional face detectors for improved robustness.

### 2. Expression Parameter Learning
- A **Multi-Layer Perceptron (MLP)** is trained to predict expression parameters.
- Inputs:
  - 3DDFA pose and shape parameters
  - MediaPipe FaceMesh landmark features
- Output:
  - Low-dimensional **expression parameters** suitable for animation.

### 3. Interactive Frontend
- Real-time **open / close mouth** interaction.
- **Three.js**-based 3D visualization.
- Landmark-to-mesh mapping for expressive facial animation.

### 4. Full Stack Implementation
- **Backend:** FastAPI for inference and model serving.
- **Frontend:** Next.js + Three.js for real-time rendering.

---

## 🧠 Training and Evaluation

All **training and evaluation code** is located in:

```text
notebooks/testing.ipynb
```
This notebook includes:

- Data preprocessing
- MLP training
- Evaluation and error analysis

Expression parameter visualization
---

Pretrained models and related artifacts are stored in:

```text
notebooks/checkpoints/
notebooks/expr_head_final.pth
```

---

###  Project Structure (Simplified)
``` directory
.
├── TDDFA.py                  # Core 3DDFA inference
├── TDDFA_ONNX.py             # ONNX inference (optional)
├── models/                   # Backbone models + expression head
├── utils/                    # Geometry, rendering, helpers
├── bfm/                      # Basel Face Model (BFM)
├── configs/                  # Model configs and statistics
├── weights/                  # Pretrained weights
├── Sim3DR/                   # 3D rendering backend
├── FaceBoxes/                # Face detector (optional)
├── backend/                  # FastAPI backend
├── frontend/                 # Next.js + Three.js frontend
├── notebooks/                # Training & evaluation
└── dataset/aflw2000-3d       # Evaluation dataset

```


---

## 🚀 Running the Project Locally

1️⃣ Backend (FastAPI)
```bash
  cd backend
  pip install -r requirements.txt
  uvicorn main:app --port 8000 --reload
```
2️⃣ Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```
---

## 🌐 Hosted Demo

The project is also deployed and accessible online:

Frontend (Vercel):
https://my-deep-learning-project.vercel.app/

Backend API (Azure Container Apps):
https://dl-backend-app.bravedune-d9b3fa99.southeastasia.azurecontainerapps.io

---

## 📊 Dataset Credit

This project uses the AFLW2000-3D dataset for evaluation.

Dataset link:[mohamedadlyi/aflw2000-3d](https://www.kaggle.com/datasets/mohamedadlyi/aflw2000-3d)

All dataset rights belong to the original authors.

---

## 📚 Original 3DDFA_v2 Credit

If you use this repository, please also cite the original 3DDFA_v2 work:

Guo, J., Zhu, X., Yang, Y., Yang, F., Lei, Z., & Li, S. Z.
Towards Fast, Accurate and Stable 3D Dense Face Alignment
ECCV 2020

Original implementation:
https://github.com/cleardusk/3DDFA_v2

---

## ⚠️ Disclaimer

This project is intended for research, educational, and demonstration purposes only.

It is not an official extension of the original 3DDFA_v2 repository.