import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions import clip_black_norm, canny_edge, hough_lines

def _imread(path):
    return cv2.imdecode(np.fromfile(os.path.abspath(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def _imwrite(path, img):
    cv2.imencode('.jpg', img)[1].tofile(os.path.abspath(path))

CLAHE_OBJ = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32, 32))
WR_THRESH = 0.6
A_THRESHOLD = 110
A_CANNY_LOW = 20
A_CANNY_HIGH = 100
A_HOUGH_THR = 120

OUT_DIR = os.path.join(SCRIPT_DIR, "ch_hough_output")
os.makedirs(OUT_DIR, exist_ok=True)

def classify(img_gray):
    blurred = cv2.medianBlur(CLAHE_OBJ.apply(img_gray), 3)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    H, W = otsu.shape
    patch = otsu[int(H*0.45):int(H*0.55), int(W*0.45):int(W*0.55)]
    if np.sum(patch == 0) / patch.size >= 0.9:
        otsu = cv2.bitwise_not(otsu)
    return "type_a" if np.sum(otsu == 255) / otsu.size >= WR_THRESH else "type_b"

def draw_lines(bgr, lines_n12):
    """lines shape (N,1,2) — rho,theta per entry"""
    vis = bgr.copy()
    if lines_n12 is None:
        return vis
    for rho, theta in lines_n12[:, 0]:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x0, y0 = cos_t * rho, sin_t * rho
        pt1 = (int(x0 + 1500 * (-sin_t)), int(y0 + 1500 * cos_t))
        pt2 = (int(x0 - 1500 * (-sin_t)), int(y0 - 1500 * cos_t))
        cv2.line(vis, pt1, pt2, (0, 0, 255), 2)
    return vis

type_a_ids = []
results = {}
for i in range(1, 22):
    path = os.path.join(SCRIPT_DIR, "..", "businesscard (1)", f"BC{i}.jpg")
    img = _imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if classify(gray) != "type_a":
        continue
    type_a_ids.append(i)
    stretch = clip_black_norm(gray, A_THRESHOLD)
    edges = canny_edge(stretch, A_CANNY_LOW, A_CANNY_HIGH)
    lines_raw = hough_lines(edges, 1, np.pi/180, A_HOUGH_THR)
    vis = draw_lines(img, lines_raw)
    n_lines = 0 if lines_raw is None else len(lines_raw)
    results[i] = (vis, n_lines)
    _imwrite(os.path.join(OUT_DIR, f"BC{i}_hough.jpg"), vis)
    print(f"BC{i} saved  ({n_lines} lines)")

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
plt.suptitle("TypeA Checkpoint: Hough Lines (threshold=120)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "_grid.png"), dpi=150)
plt.show()
