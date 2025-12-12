import torch
import torchvision


def nms(dets, thresh):
    """
    Pure PyTorch NMS replacement for cpu_nms.
    dets: numpy array [N, 5] -> [x1, y1, x2, y2, score]
    thresh: float - IoU threshold
    """
    if dets.shape[0] == 0:
        return []

    boxes = torch.tensor(dets[:, :4], dtype=torch.float32)
    scores = torch.tensor(dets[:, 4], dtype=torch.float32)

    keep = torchvision.ops.nms(boxes, scores, thresh)

    return keep.numpy()


# Soft-NMS fallback (not used, but required to avoid import errors)
def cpu_soft_nms(dets, thresh, method=1, sigma=0.5):
    return nms(dets, thresh)
