import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from functions import (
    # TYPE A
    clip_black_norm, canny_edge, hough_lines, nms_lines, find_best_quad, sort_corners,
    # TYPE B
    fill_holes, region_filter, merge_center_regions, find_corners,
    # 공통
    compute_homography, warp_perspective,
)

IMG_DIR = "businesscard (1)"

bc_imgs  = {}
bc_grays = {}
for i in range(1, 22):
    path = os.path.join(IMG_DIR, f"BC{i}.jpg")
    img  = cv2.imread(path)
    bc_imgs[i]  = img
    bc_grays[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Loaded {len(bc_imgs)} images")

# ── 분류 파라미터 ──────────────────────────────────────────────────────────────
CLAHE_OBJ = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32, 32))
WR_THRESH = 0.6
# ──────────────────────────────────────────────────────────────────────────────

# ── TYPE A 파라미터 ────────────────────────────────────────────────────────────
A_THRESHOLD      = 110    # 이 값 미만 픽셀 → 0, 이상만 [0,255] 스트레치
A_CANNY_LOW      = 20     # Canny 하위 임계값: 이하 픽셀은 엣지 후보 제외
A_CANNY_HIGH     = 100    # Canny 상위 임계값: 초과 픽셀은 확실한 엣지로 확정
A_HOUGH_THR      = 120    # HoughLines 임계값: 이 수 이상 투표된 (ρ,θ)만 직선 검출
A_RHO_MERGE      = 100    # NMS 병합 기준: ρ 차이가 이 값 미만->동일 직선으로 간주
A_ANGLE_MERGE    = 18.5   # NMS 병합 기준: 각도 차이(도)가 이 값 미만-> 동일 직선으로 간주
A_CENTER_PCT     = 0.10   # 중앙 필터: 사각형 무게중심이 이미지 중앙 10% 내부여야 통과
A_MIN_RHO_GAP    = 100    # 평행 검증: 같은 쌍 두 직선의 ρ 차이 최소값
A_MIN_ANGLE_DIFF = 50     # 수직 검증: 두 직선 쌍의 평균 각도 차이 최소갑
# ──────────────────────────────────────────────────────────────────────────────


def center_black_ratio(otsu): #otsu 이후 중심부가 과도하게 검정일 경우 반전된 이미지일 것을 가정
    H, W = otsu.shape
    patch = otsu[int(H*0.45):int(H*0.55), int(W*0.45):int(W*0.55)] # 이미지 중심 10% 영역 확인
    return np.sum(patch == 0) / patch.size


def classify(img_gray):
    blurred = cv2.medianBlur(CLAHE_OBJ.apply(img_gray), 3)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if center_black_ratio(otsu) >= 0.9: # 중심부 90%가 검정일 시, 반전
        otsu = cv2.bitwise_not(otsu) # 전처리 이므로 cv2 함수 사용
    mode = "type_a" if np.sum(otsu == 255) / otsu.size >= WR_THRESH else "type_b"
    return otsu, mode


def process(i):
    img_color = bc_imgs[i]
    img_gray  = bc_grays[i]
    otsu, mode = classify(img_gray)

    if mode == "type_a": # 1차 painpoint 흰 배경과 흰 명함 -> 밝은 색의 표현을 극대화
        H_img, W_img = img_gray.shape
        stretch   = clip_black_norm(img_gray, A_THRESHOLD) # 픽셀값 110 이하는 전체 0으로, 이후는 선형으로 늘림
        canny     = canny_edge(stretch, A_CANNY_LOW, A_CANNY_HIGH) # 캐니 실행
        lines_raw = hough_lines(canny, 1, np.pi/180, A_HOUGH_THR) # line detecter
        lines_nms = nms_lines(lines_raw, A_RHO_MERGE, A_ANGLE_MERGE) # 중복 line 단일화
        clamped   = find_best_quad(lines_nms, H_img, W_img, # 명함 domain에 맞게 네 선분 -> 교점: 네 코너
                                   A_CENTER_PCT, A_MIN_RHO_GAP, A_MIN_ANGLE_DIFF)
        if clamped is None:
            return np.zeros((250, 450, 3), dtype=np.uint8), mode
        corners = sort_corners(clamped) # homography 행렬을 위한 정렬
    else:
        filled_otsu = fill_holes(otsu) # 1차시와 동일: otsu 이후 검정 구멍 메우기 BFS
        selected    = region_filter(filled_otsu) # 8연결 BFS로 가장 큰 흰 component 남기기
        filled      = merge_center_regions(filled_otsu, selected, img_gray.size) # 명함 내부로 인해 분리될 수도 있어 중앙 영역 결합
        corners     = find_corners(filled) # PCA로 주축 방향 탐색 -> 회전 -> 4 코너 추출

    w = np.linalg.norm(corners[1] - corners[0])
    h = np.linalg.norm(corners[3] - corners[0])
    out_w, out_h = (450, 250) if w >= h else (250, 450)
    dst    = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
    H_mat  = compute_homography(corners, dst) # 변환 행렬 만들기
    warped = warp_perspective(img_color, H_mat, (out_w, out_h)) # 와핑
    return warped, mode


OUT_DIR = "output"
# 메인루프 및 시각화
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    success, fail = [], []

    for i in range(1, 22):
        print(f"BC{i}...", end=" ")
        try:
            warped, mode = process(i)
            cv2.imwrite(os.path.join(OUT_DIR, f"BC{i}.jpg"), warped)
            success.append(i)
            print(f"OK ({mode})")
        except Exception as e:
            fail.append(i)
            print(f"FAIL ({e})")

    print(f"\n{len(success)}/21 성공, 실패: {fail}")

    fig, axes = plt.subplots(3, 7, figsize=(24, 12))
    for i in range(1, 22):
        ax = axes[(i-1)//7][(i-1)%7]
        try:
            warped, mode = process(i)
            ax.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            ax.set_title(f"BC{i} ({mode})", fontsize=8,
                         color='navy' if mode == 'type_a' else 'black')
        except Exception as e:
            ax.set_title(f"BC{i} FAIL", fontsize=8, color='red')
        ax.axis('off')

    plt.suptitle("Business Card Recognition — BC1~BC21  (TYPE A: navy / TYPE B: black)", fontsize=13)
    plt.tight_layout()
    plt.show()
