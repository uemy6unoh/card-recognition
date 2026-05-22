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

# k, d Hough 파라미터
K_MIN, K_MAX, K_STEP = -6.0, 6.0, 0.05   # 기울기 범위 (수직선은 커버 불가)
D_STEP = 2.0                               # d 해상도 (픽셀 단위)
KD_THRESHOLD = 80                          # 최소 투표수

OUT_DIR = os.path.join(SCRIPT_DIR, "ch_hough_kd_output")
os.makedirs(OUT_DIR, exist_ok=True)

def classify(img_gray):
    blurred = cv2.medianBlur(CLAHE_OBJ.apply(img_gray), 3)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    H, W = otsu.shape
    patch = otsu[int(H*0.45):int(H*0.55), int(W*0.45):int(W*0.55)]
    if np.sum(patch == 0) / patch.size >= 0.9:
        otsu = cv2.bitwise_not(otsu)
    return "type_a" if np.sum(otsu == 255) / otsu.size >= WR_THRESH else "type_b"


def hough_lines_kd(edge, k_min=K_MIN, k_max=K_MAX, k_step=K_STEP,
                   d_step=D_STEP, threshold=KD_THRESHOLD):
    """y = kx + d 파라미터 공간에서 Hough 투표
    수직선(|k|→∞)은 표현 불가 — 이것이 rho/theta 방식과의 핵심 차이"""
    H, W = edge.shape
    k_vals = np.arange(k_min, k_max + k_step * 0.5, k_step, dtype=np.float32)
    num_k  = len(k_vals)
    d_min  = float(-H)
    d_max  = float(W * abs(k_max) + H)
    num_d  = int((d_max - d_min) / d_step) + 1

    ys, xs = np.where(edge > 0)
    if len(xs) == 0:
        return None, k_vals, d_min

    # d = y - k*x  →  shape (n_pts, num_k)
    d_mat  = ys[:, None].astype(np.float32) - k_vals[None, :] * xs[:, None].astype(np.float32)
    d_idx  = np.round((d_mat - d_min) / d_step).astype(np.int32)
    k_idx  = np.broadcast_to(np.arange(num_k, dtype=np.int32), d_idx.shape)

    valid    = (d_idx >= 0) & (d_idx < num_d)
    flat_idx = k_idx[valid] * num_d + d_idx[valid]

    counts = np.bincount(flat_idx.ravel(), minlength=num_k * num_d)
    accum  = counts.reshape(num_k, num_d).astype(np.int32)

    ki_arr, di_arr = np.where(accum >= threshold)
    if len(ki_arr) == 0:
        return None, k_vals, d_min

    order  = np.argsort(-accum[ki_arr, di_arr])
    ki_arr, di_arr = ki_arr[order], di_arr[order]

    k_out = k_vals[ki_arr]
    d_out = d_min + di_arr * d_step
    return np.stack([k_out, d_out], axis=1), k_vals, d_min  # (N, 2)


def draw_kd_lines(bgr, lines_kd, color=(0, 0, 255)):
    vis = bgr.copy()
    if lines_kd is None:
        return vis
    H, W = vis.shape[:2]
    for k, d in lines_kd:
        # y = kx + d  →  두 끝점 계산
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

    stretch  = clip_black_norm(gray, A_THRESHOLD)
    edges    = canny_edge(stretch, A_CANNY_LOW, A_CANNY_HIGH)
    lines_kd, _, _ = hough_lines_kd(edges)
    vis      = draw_kd_lines(img, lines_kd)
    n_lines  = 0 if lines_kd is None else len(lines_kd)
    results[i] = (vis, n_lines)
    _imwrite(os.path.join(OUT_DIR, f"BC{i}_hough_kd.jpg"), vis)
    print(f"BC{i}: {n_lines} lines (k,d)")

n = len(type_a_ids)
cols = 7
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.5))
axes = np.array(axes).reshape(-1)
for j, i in enumerate(type_a_ids):
    vis, n_lines = results[i]
    axes[j].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[j].set_title(f"BC{i}  ({n_lines}lines)", fontsize=7)
    axes[j].axis('off')
for j in range(len(type_a_ids), len(axes)):
    axes[j].axis('off')
plt.suptitle(f"TypeA: Hough (k,d)  k∈[{K_MIN},{K_MAX}] step={K_STEP}  |  수직선 검출 불가", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "_grid.png"), dpi=150)
plt.show()
