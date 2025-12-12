# coding: utf-8
import argparse
import cv2
import numpy as np
from collections import deque
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.functions import cv_draw_landmark


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # init
    gpu_mode = args.mode == 'gpu'
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    face_boxes = FaceBoxes()

    cap = cv2.VideoCapture(0)

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

        if first_frame:
            boxes = face_boxes(img)
            if len(boxes) == 0:
                cv2.imshow("image", img)
                if cv2.waitKey(1) == ord('q'):
                    break
                continue

            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(img, boxes)
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
