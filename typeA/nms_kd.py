import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions import clip_black_norm, canny_edge

def _imread(path):
    return cv2.imdecode(np.fromfile(os.path.abspath(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def _imwrite(path, img):
    cv2.imencode('.jpg', img)[1].tofile(os.path.abspath(path))

CLAHE_OBJ = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32, 32))
WR_THRESH = 0.6
A_THRESHOLD = 110
A_CANNY_LOW = 20
A_CANNY_HIGH = 100

K_MIN, K_MAX, K_STEP = -50.0, 50.0, 0.05
D_STEP = 2.0
KD_THRESHOLD = 80
K_MERGE = 0.3    # NMS: 기울기 차이 임계값
D_MERGE = 60.0   # NMS: 절편 차이 임계값

OUT_DIR = os.path.join(SCRIPT_DIR, "ch_nms_kd_output")
os.makedirs(OUT_DIR, exist_ok=True)

def classify(img_gray):
    blurred = cv2.medianBlur(CLAHE_OBJ.apply(img_gray), 3)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    H, W = otsu.shape
    patch = otsu[int(H*0.45):int(H*0.55), int(W*0.45):int(W*0.55)]
    if np.sum(patch == 0) / patch.size >= 0.9:
        otsu = cv2.bitwise_not(otsu)
    return "type_a" if np.sum(otsu == 255) / otsu.size >= WR_THRESH else "type_b"


def hough_lines_kd(edge):
    H, W = edge.shape
    k_vals = np.arange(K_MIN, K_MAX + K_STEP * 0.5, K_STEP, dtype=np.float32)
    num_k  = len(k_vals)
    d_min  = float(-H)
    d_max  = float(W * abs(K_MAX) + H)
    num_d  = int((d_max - d_min) / D_STEP) + 1

    ys, xs = np.where(edge > 0)
    if len(xs) == 0:
        return None

    d_mat  = ys[:, None].astype(np.float32) - k_vals[None, :] * xs[:, None].astype(np.float32)
    d_idx  = np.round((d_mat - d_min) / D_STEP).astype(np.int32)
    k_idx  = np.broadcast_to(np.arange(num_k, dtype=np.int32), d_idx.shape)

    valid    = (d_idx >= 0) & (d_idx < num_d)
    flat_idx = k_idx[valid] * num_d + d_idx[valid]

    counts = np.bincount(flat_idx.ravel(), minlength=num_k * num_d)
    accum  = counts.reshape(num_k, num_d).astype(np.int32)

    ki_arr, di_arr = np.where(accum >= KD_THRESHOLD)
    if len(ki_arr) == 0:
        return None

    order  = np.argsort(-accum[ki_arr, di_arr])
    ki_arr, di_arr = ki_arr[order], di_arr[order]

    k_out = k_vals[ki_arr]
    d_out = d_min + di_arr * D_STEP
    return np.stack([k_out, d_out], axis=1)


def nms_lines_kd(lines):
    """k, d 공간에서 중복 직선 제거"""
    if lines is None:
        return None
    kept = []
    for k, d in lines:
        if not any(abs(k - kk) < K_MERGE and abs(d - kd) < D_MERGE for kk, kd in kept):
            kept.append((k, d))
    return np.array(kept) if kept else None


def draw_kd_lines(bgr, lines_kd, color=(0, 255, 0)):
    vis = bgr.copy()
    if lines_kd is None:
        return vis
    H, W = vis.shape[:2]
    for k, d in lines_kd:
        x0, y0 = 0,   int(round(d))
        x1, y1 = W-1, int(round(k * (W-1) + d))
        cv2.line(vis, (x0, y0), (x1, y1), color, 2)
    return vis


type_a_ids = []
results = {}
for i in range(1, 22):
    path = os.path.join(SCRIPT_DIR, "..", "businesscard (1)", f"BC{i}.jpg")
    img  = _imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if classify(gray) != "type_a":
        continue
    type_a_ids.append(i)

    stretch   = clip_black_norm(gray, A_THRESHOLD)
    edges     = canny_edge(stretch, A_CANNY_LOW, A_CANNY_HIGH)
    lines_raw = hough_lines_kd(edges)
    lines_nms = nms_lines_kd(lines_raw)
    vis       = draw_kd_lines(img, lines_nms)
    nb = 0 if lines_raw is None else len(lines_raw)
    na = 0 if lines_nms is None else len(lines_nms)
    results[i] = (vis, nb, na)
    _imwrite(os.path.join(OUT_DIR, f"BC{i}_nms_kd.jpg"), vis)
    print(f"BC{i}: {nb} → {na} lines (k,d NMS)")

n = len(type_a_ids)
cols = 7
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
axes = np.array(axes).reshape(-1)
for j, i in enumerate(type_a_ids):
    vis, nb, na = results[i]
    axes[j].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[j].set_title(f"BC{i}  {nb}→{na}", fontsize=7)
    axes[j].axis('off')
for j in range(len(type_a_ids), len(axes)):
    axes[j].axis('off')
plt.suptitle(f"TypeA: NMS (k,d)  k_merge={K_MERGE}  d_merge={D_MERGE}  |  수직선 검출 불가", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "_grid.png"), dpi=150)
plt.show()
