import numpy as np

# ---------- NME (2D landmarks) ----------
def normalized_mean_error(pred_2d, gt_2d, norm='inter_ocular'):
    """
    pred_2d, gt_2d: (L,2) numpy arrays
    norm: 'inter_ocular' or 'bbox' or 'inter_pupil'
    """
    assert pred_2d.shape == gt_2d.shape
    L = pred_2d.shape[0]
    diff = np.linalg.norm(pred_2d - gt_2d, axis=1)

    # choose normalization
    if norm == 'inter_ocular':
        # Example for 68-landmarks: use outer eye corners (indices may differ in your dataset)
        # Replace indices if using a different landmark scheme.
        left_outer = gt_2d[36]
        right_outer = gt_2d[45]
        d = np.linalg.norm(left_outer - right_outer)
    elif norm == 'bbox':
        x_min, y_min = np.min(gt_2d, axis=0)
        x_max, y_max = np.max(gt_2d, axis=0)
        d = np.sqrt((x_max-x_min)*(y_max-y_min))
    else:
        raise ValueError("Unsupported normalization mode.")

    return float(np.mean(diff) / d)
