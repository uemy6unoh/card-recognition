import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from functions import clip_black_norm, canny_edge, hough_lines, nms_lines, line_intersect

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
A_RHO_MERGE = 100
A_ANGLE_MERGE = 18.5
A_CENTER_PCT = 0.10
A_MIN_RHO_GAP = 100
A_MIN_ANGLE_DIFF = 50

OUT_DIR = os.path.join(SCRIPT_DIR, "ch_top3_output")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0)]  # 1st: red, 2nd: orange, 3rd: green

def classify(img_gray):
    blurred = cv2.medianBlur(CLAHE_OBJ.apply(img_gray), 3)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    H, W = otsu.shape
    patch = otsu[int(H*0.45):int(H*0.55), int(W*0.45):int(W*0.55)]
    if np.sum(patch == 0) / patch.size >= 0.9:
        otsu = cv2.bitwise_not(otsu)
    return "type_a" if np.sum(otsu == 255) / otsu.size >= WR_THRESH else "type_b"

def find_top3_quads(lines_nms, H, W,
                    center_pct=0.10, min_rho_gap=100, min_angle_diff_deg=50):
    """find_best_quad의 변형 — 점수순 상위 3개 반환"""
    if lines_nms is None or len(lines_nms) < 4:
        return []

    def is_inside(x, y):
        return 0 <= x <= W and 0 <= y <= H

    top3     = []  # (score, clamped) 리스트, 최대 3개 유지
    min_ang  = np.deg2rad(min_angle_diff_deg)
    img_diag = np.hypot(H, W)
    cx_half  = W * center_pct
    cy_half  = H * center_pct
    lines    = list(lines_nms)

    for idx in combinations(range(len(lines)), 4):
        l = [lines[k] for k in idx]
        for a1i, a2i, b1i, b2i in [(0,1,2,3),(0,2,1,3),(0,3,1,2)]:
            la1, la2 = l[a1i], l[a2i]
            lb1, lb2 = l[b1i], l[b2i]

            if abs((la1[1]+la2[1])/2 - (lb1[1]+lb2[1])/2) < min_ang:
                continue
            if abs(la1[0]-la2[0]) < min_rho_gap or abs(lb1[0]-lb2[0]) < min_rho_gap:
                continue

            pt_aa = line_intersect(la1[0], la1[1], la2[0], la2[1])
            if pt_aa is not None and is_inside(pt_aa[0], pt_aa[1]):
                continue
            pt_bb = line_intersect(lb1[0], lb1[1], lb2[0], lb2[1])
            if pt_bb is not None and is_inside(pt_bb[0], pt_bb[1]):
                continue

            corners_raw = [line_intersect(la[0], la[1], lb[0], lb[1])
                           for la in [la1, la2] for lb in [lb1, lb2]]
            if any(p is None for p in corners_raw):
                continue
            if sum(is_inside(x, y) for x, y in corners_raw) < 3:
                continue

            clamped = [(np.clip(x, 0, W-1), np.clip(y, 0, H-1)) for x, y in corners_raw]

            quad_cx = sum(x for x, _ in corners_raw) / 4
            quad_cy = sum(y for _, y in corners_raw) / 4
            if abs(quad_cx - W/2) > cx_half or abs(quad_cy - H/2) > cy_half:
                continue

            qx = [clamped[0][0], clamped[1][0], clamped[3][0], clamped[2][0]]
            qy = [clamped[0][1], clamped[1][1], clamped[3][1], clamped[2][1]]
            area   = 0.5 * abs(sum(qx[k]*qy[(k+1)%4] - qx[(k+1)%4]*qy[k] for k in range(4)))
            s_area = area / (H * W)

            center_dist = np.hypot(quad_cx - W/2, quad_cy - H/2)
            s_center = np.clip(
                1.0 - np.log1p(center_dist) / np.log1p(np.hypot(cx_half, cy_half)),
                0.0, 1.0)

            c = corners_raw
            la1_ = np.hypot(c[0][0]-c[1][0], c[0][1]-c[1][1])
            la2_ = np.hypot(c[2][0]-c[3][0], c[2][1]-c[3][1])
            lb1_ = np.hypot(c[0][0]-c[2][0], c[0][1]-c[2][1])
            lb2_ = np.hypot(c[1][0]-c[3][0], c[1][1]-c[3][1])
            len_a = (la1_ + la2_) / 2
            len_b = (lb1_ + lb2_) / 2
            ratio       = max(len_a, len_b) / (min(len_a, len_b) + 1e-8)
            asp_penalty = abs(ratio - 1.8)
            side_penalty = (abs(la1_ - la2_) + abs(lb1_ - lb2_)) / img_diag

            score = s_area * s_center - asp_penalty - side_penalty

            top3.append((score, clamped))
            top3.sort(key=lambda x: x[0], reverse=True)
            if len(top3) > 3:
                top3.pop()

    return top3  # list of (score, clamped), 길이 최대 3


def draw_quad(bgr, clamped, color, rank, score):
    vis = bgr
    pts = np.array([[int(x), int(y)] for x, y in
                    [clamped[0], clamped[1], clamped[3], clamped[2]]], dtype=np.int32)
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=3)
    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
    cv2.putText(vis, f"#{rank} {score:.3f}", (cx-40, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


type_a_ids = []
results = {}  # i -> (img, top3_list)
for i in range(1, 22):
    path = os.path.join(SCRIPT_DIR, "..", "businesscard (1)", f"BC{i}.jpg")
    img = _imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if classify(gray) != "type_a":
        continue
    type_a_ids.append(i)

    H_img, W_img = gray.shape
    stretch = clip_black_norm(gray, A_THRESHOLD)
    edges = canny_edge(stretch, A_CANNY_LOW, A_CANNY_HIGH)
    lines_raw = hough_lines(edges, 1, np.pi/180, A_HOUGH_THR)
    lines_nms = nms_lines(lines_raw, A_RHO_MERGE, A_ANGLE_MERGE) if lines_raw is not None else None

    top3 = find_top3_quads(lines_nms, H_img, W_img,
                           A_CENTER_PCT, A_MIN_RHO_GAP, A_MIN_ANGLE_DIFF)
    results[i] = (img, top3)

    # 개별 저장: rank별로 파일 분리
    for rank, (score, clamped) in enumerate(top3, 1):
        vis = draw_quad(img.copy(), clamped, COLORS[rank-1], rank, score)
        _imwrite(os.path.join(OUT_DIR, f"BC{i}_rank{rank}.jpg"), vis)
    print(f"BC{i}: top{len(top3)} quads found")

# 시각화: 행=rank(1~3), 열=각 typeA 이미지
n = len(type_a_ids)
fig, axes = plt.subplots(3, n, figsize=(n * 3, 10))
axes = np.array(axes).reshape(3, n)

for j, i in enumerate(type_a_ids):
    img, top3 = results[i]
    for rank in range(1, 4):
        ax = axes[rank-1][j]
        if rank <= len(top3):
            score, clamped = top3[rank-1]
            vis = draw_quad(img.copy(), clamped, COLORS[rank-1], rank, score)
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.set_title(f"BC{i}\n{score:.3f}", fontsize=7)
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"BC{i}\n(없음)", fontsize=7, color='gray')
        ax.axis('off')

for rank in range(1, 4):
    axes[rank-1][0].set_ylabel(f"#{rank}", fontsize=11, rotation=0, labelpad=30, va='center')

plt.suptitle("TypeA Checkpoint: Top-3 Quads\n"
             "행1=1위(red)  행2=2위(orange)  행3=3위(green)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "_grid.png"), dpi=150)
plt.show()
