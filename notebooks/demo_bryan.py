# coding: utf-8
import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
import cv2
import numpy as np
from collections import deque
import yaml
import torch
import mediapipe as mp
from models.expression_head import ExpressionHead
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.functions import cv_draw_landmark

from models.expression_head import ExpressionHead
model = ExpressionHead()
print(model)


def extract_mp_landmarks(img, mp_face):
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)

    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[p.x * w, p.y * h, p.z] for p in lm])
    return pts.reshape(-1)

def draw_expression_text(img, expr, title, x, y):
    if expr is None:
        return

    cv2.putText(img, title, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    for i, v in enumerate(expr):
        cv2.putText(img, f"{i}: {v:.2f}", (x, y + 15*(i+1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # init
    gpu_mode = args.mode == 'gpu'
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    # -------- Expression Head --------
    expr_model = ExpressionHead()
    expr_model.load_state_dict(
        torch.load("./notebooks/checkpoints/expr_head_final.pth", map_location="cpu")
    )
    expr_model.eval()

    # -------- MediaPipe --------
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    )

    face_boxes = FaceBoxes()

    cap = cv2.VideoCapture("examples/inputs/videos/214.avi")

    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1

    queue_ver = deque()
    queue_frame = deque()

    dense_flag = args.opt in ('2d_dense', '3d')
    pre_ver = None
    first_frame = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = frame.copy()
        lm = extract_mp_landmarks(img, mp_face)
        expr_ours = None

        if lm is not None:
            with torch.no_grad():
                lm_t = torch.from_numpy(lm).float().unsqueeze(0)
                expr_ours = expr_model(lm_t).numpy()[0]

        if first_frame:
            boxes = face_boxes(img)
            if len(boxes) == 0:
                draw_expression_text(out, expr_3ddfa, "3DDFA Expr", 10, 20)
                draw_expression_text(out, expr_ours, "Ours (2D LM)", 200, 20)
                cv2.imshow("image", img)
                if cv2.waitKey(1) == ord('q'):
                    break
                continue

            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(img, boxes)
            expr_3ddfa = param_lst[0][12:22]  # 10-dim expression
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # padding queue
            for _ in range(n_pre):
                queue_ver.append(ver.copy())
                queue_frame.append(img.copy())

            queue_ver.append(ver.copy())
            queue_frame.append(img.copy())

            pre_ver = ver.copy()
            first_frame = False

        else:
            param_lst, roi_box_lst = tddfa(img, [pre_ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            queue_ver.append(ver.copy())
            queue_frame.append(img.copy())

            pre_ver = ver.copy()

        # ---- smoothing & rendering ----
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)
            base_frame = queue_frame[n_pre]  # FIX: always exists

            if args.opt == '2d_sparse':
                out = cv_draw_landmark(base_frame, ver_ave)
            elif args.opt == '2d_dense':
                out = cv_draw_landmark(base_frame, ver_ave, size=1)
            elif args.opt == '3d':
                out = render(base_frame, [ver_ave], tddfa.tri, alpha=0.7)

            cv2.imshow("image", out)
            if cv2.waitKey(1) == ord('q'):
                break

            queue_ver.popleft()
            queue_frame.popleft()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/mb1_120x120.yml')
    parser.add_argument('-m', '--mode', default='cpu')
    parser.add_argument('-o', '--opt', default='3d',
                        choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-n_pre', default=1, type=int)
    parser.add_argument('-n_next', default=1, type=int)

    args = parser.parse_args()
    main(args)
