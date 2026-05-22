import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions import fill_holes, region_filter, merge_center_regions, find_corners

def _imread(path):
    return cv2.imdecode(np.fromfile(os.path.abspath(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def _imwrite(path, img):
    cv2.imencode('.jpg', img)[1].tofile(os.path.abspath(path))

CLAHE_OBJ = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32, 32))
WR_THRESH = 0.6

OUT_DIR = os.path.join(SCRIPT_DIR, "ch_findcorner_output")
os.makedirs(OUT_DIR, exist_ok=True)

CORNER_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0)]  # TL, TR, BR, BL
CORNER_LABELS = ['TL', 'TR', 'BR', 'BL']

def get_otsu(img_gray):
    blurred = cv2.medianBlur(CLAHE_OBJ.apply(img_gray), 3)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    H, W = otsu.shape
    patch = otsu[int(H*0.45):int(H*0.55), int(W*0.45):int(W*0.55)]
    if np.sum(patch == 0) / patch.size >= 0.9:
        otsu = cv2.bitwise_not(otsu)
    return otsu

def is_type_b(otsu):
    return np.sum(otsu == 255) / otsu.size < WR_THRESH

type_b_ids = []
results = {}
for i in range(1, 22):
    path = os.path.join(SCRIPT_DIR, "..", "businesscard (1)", f"BC{i}.jpg")
    gray = cv2.cvtColor(_imread(path), cv2.COLOR_BGR2GRAY)
    otsu = get_otsu(gray)
    if not is_type_b(otsu):
        continue
    type_b_ids.append(i)

    filled_otsu = fill_holes(otsu)
    selected = region_filter(filled_otsu)
    merged = merge_center_regions(filled_otsu, selected, gray.size)
    corners = find_corners(merged)  # binary에서 코너 추출

    vis = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)
    pts_poly = corners[[0, 1, 2, 3]].astype(np.int32)  # TL TR BR BL
    cv2.polylines(vis, [pts_poly], isClosed=True, color=(0, 255, 255), thickness=2)
    for k, (x, y) in enumerate(corners.astype(int)):
        cv2.circle(vis, (x, y), 8, CORNER_COLORS[k], -1)
        cv2.putText(vis, CORNER_LABELS[k], (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CORNER_COLORS[k], 2)

    results[i] = vis
    _imwrite(os.path.join(OUT_DIR, f"BC{i}_findcorner.jpg"), vis)
    print(f"BC{i} saved  corners={corners.astype(int).tolist()}")

n = len(type_b_ids)
cols = min(n, 6)
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 4))
axes = np.array(axes).reshape(-1)
for j, i in enumerate(type_b_ids):
    axes[j].imshow(cv2.cvtColor(results[i], cv2.COLOR_BGR2RGB))
    axes[j].set_title(f"BC{i}", fontsize=8)
    axes[j].axis('off')
for j in range(len(type_b_ids), len(axes)):
    axes[j].axis('off')
plt.suptitle("TypeB Checkpoint: find_corners (PCA)  on binary\n"
             "red=TL  orange=TR  green=BR  blue=BL", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "_grid.png"), dpi=150)
plt.show()
