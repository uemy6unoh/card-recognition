import numpy as np                                 
from collections import deque                        # BFS용 양방향 큐
from itertools import combinations                   # nCk 조합 생성용



# TYPE A  (흰 배경 카드)
# ClipBlackNorm -> Canny -> HoughLines -> NMS -> FindBestQuad -> SortCorners -> Warp

def clip_black_norm(img, threshold=110):
    out  = img.astype(np.float32)                   
    mask = img >= threshold                          # threshold 이상인 픽셀만 스트레치 대상
    lo, hi = float(img[mask].min()), float(img[mask].max())  # 대상 픽셀의 최소,최대값
    out[mask]  = (img[mask].astype(np.float32) - lo) / (hi - lo) * 255  # [lo,hi] -> [0,255] 선형 매핑
    out[~mask] = 0                                   # threshold 미만 픽셀 완전 검정(픽셀값 0)
    return np.clip(out, 0, 255).astype(np.uint8)    # 범위 초과 방지 후 uint8 반환


def _sobel_with_angle(img): # canny용
    f = img.astype(np.int16)                         # uint8 -> int16 (음수 연산, overflow 방지)
    p = np.pad(f, 1, mode='reflect')                # 테두리 픽셀 반사
    Gx = (-p[0:-2,0:-2] + p[0:-2,2:]               # Sobel X 커널
          -2*p[1:-1,0:-2] + 2*p[1:-1,2:]           # 수평 방향 밝기 변화율
          -p[2:,  0:-2] + p[2:,  2:])
    Gy = (-p[0:-2,0:-2] - 2*p[0:-2,1:-1] - p[0:-2,2:]  # Sobel Y 커널
          +p[2:,  0:-2] + 2*p[2:,  1:-1] + p[2:,  2:])  # 수직 방향 밝기 변화율
    mag = (np.abs(Gx) + np.abs(Gy)).astype(np.int32)    # L1 norm: |Gx|+|Gy| (단순 크기 합)
    return mag, np.arctan2(Gy.astype(np.float32), Gx.astype(np.float32))  # gradient 크기와 방향각 반환(np arctan2 사용)


def _nms_edge(mag, angle): # canny용
    a = np.degrees(angle) % 180                      # 0~180도로 정규화
    p = np.pad(mag, 1, mode='constant', constant_values=0)  # 경계 비교용 0-패딩
    n1_0,   n2_0   = p[1:-1, 0:-2], p[1:-1, 2:]    # 0도 방향 이웃: 좌우 이웃과 비교
    n1_45,  n2_45  = p[2:,   2:],   p[0:-2, 0:-2]  # 45도 방향 이웃: 우하-좌상 이웃과 비교
    n1_90,  n2_90  = p[0:-2, 1:-1], p[2:,   1:-1]  # 90도 방향 이웃: 상하 이웃과 비교
    n1_135, n2_135 = p[2:,   0:-2], p[0:-2, 2:]    # 135도 방향 이웃: 좌하-우상 이웃과 비교
    m0   = (a < 22.5)   | (a >= 157.5)              # gradient가 수평인 픽셀 마스크
    m45  = (a >= 22.5)  & (a < 67.5)               # gradient가 \ 방향인 픽셀 마스크
    m90  = (a >= 67.5)  & (a < 112.5)              # gradient가 수직인 픽셀 마스크
    m135 = (a >= 112.5) & (a < 157.5)              # gradient가 / 방향인 픽셀 마스크
    keep = ((m0   & (mag > n1_0)   & (mag > n2_0))   |  # 각 방향 이웃 두개보다 크면 살리고
            (m45  & (mag > n1_45)  & (mag > n2_45))  |  # 극대가 아니면 0으로 억제
            (m90  & (mag > n1_90)  & (mag > n2_90))  |
            (m135 & (mag > n1_135) & (mag > n2_135)))
    return np.where(keep, mag, 0)                    # 극대값 픽셀은 mag, 나머지는 0


def _hysteresis(suppressed, low=20, high=100): # canny용
    strong  = suppressed > high                      # 강한 엣지: 상위 임계값 초과 픽셀
    weak    = (suppressed > low) & ~strong           # 약한 엣지 후보: 하위 초과 & 강한 엣지 아님
    edges   = strong.copy()                          # 최종 엣지 맵 초기화 (강한 엣지는 기준)
    visited = strong.copy()                          # 방문 여부 추적 배열
    H, W    = suppressed.shape                       
    q       = deque(zip(*np.where(strong)))          # strong 픽셀 전부 BFS 큐에 삽입
    while q:                                         # 큐가 빌 때까지 BFS 반복
        y, x = q.popleft()                          # 현재 픽셀 꺼내기
        for dy in (-1, 0, 1):                       # 8방향 이웃 탐색 (dy, dx)
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx             # 이웃 픽셀 좌표
                if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and weak[ny, nx]:
                    visited[ny, nx] = True           # 방문 표시
                    edges[ny, nx]   = True           # 강한 엣지와 연결-> 엣지로 확정
                    q.append((ny, nx))               # 이 픽셀의 이웃도 탐색하게 큐에 추가
    result = np.zeros_like(suppressed, dtype=np.uint8)  # 출력 이진 엣지 맵 초기화
    result[edges] = 255                              # 확정된 엣지 픽셀을 흰색(255)으로 설정
    return result                                    # 최종 맵 반환

# canny edge
def canny_edge(img, low=20, high=100):
    mag, angle = _sobel_with_angle(img)              # Sobel로 gradient 크기, 방향 계산
    suppressed = _nms_edge(mag, angle)               # NMS로 엣지 방향 극대값만 남김
    return _hysteresis(suppressed, low, high)        # 이중 임계값으로 최종 엣지 확정

# hough line detection: k,d의 한계 -> 각도와 거리 Hessian normal form 참고.(이후 명함 특성 활용한 각도비교 등에 유리)
def hough_lines(edge, rho_step=1.0, theta_step=np.pi/180, threshold=120):
    H, W = edge.shape                                # 이미지 좌상단을 원점으로해서 (0,0)
    D = int(np.ceil(np.hypot(H, W)))                # 최대 ρ값 = 이미지 대각선 길이
    num_rho   = 2 * D + 1                           # ρ bin 총 개수 (-D ~ +D)
    num_theta = round(np.pi / theta_step)           # θ bin 총 개수 (0° ~ 179°)

    thetas = np.arange(num_theta, dtype=np.float64) * theta_step  # θ 배열 생성 [0, Δθ, 2Δθ, ...]
    cos_t  = np.cos(thetas)                         # 모든 θ에 대해 cosθ 미리 계산
    sin_t  = np.sin(thetas)                         # 모든 θ에 대해 sinθ 미리 계산

    ys, xs = np.where(edge > 0)                     # 엣지 픽셀 좌표 추출 (y배열, x배열)
    if len(xs) == 0:                                # 엣지 픽셀이 없으면
        return None                                  # 결과 없음

    # ρ = x * cosθ + y * sinθ 를 모든 (엣지픽셀, θ) 조합에 대해 한 번에 계산
    rhos    = xs[:, None] * cos_t + ys[:, None] * sin_t  # shape: (엣지픽셀 수, num_theta)
    rho_idx = np.round(rhos / rho_step).astype(np.int32) + D  # ρ -> bin 인덱스 (D 더해 양수화)

    # θ 인덱스를 rho_idx와 같은 shape으로 브로드캐스트
    theta_idx = np.broadcast_to(np.arange(num_theta, dtype=np.int32), rho_idx.shape)

    rho_flat   = rho_idx.ravel()                    # 2D -> 1D로 펼치기
    theta_flat = theta_idx.ravel()                  # 2D -> 1D로 펼치기
    valid      = (rho_flat >= 0) & (rho_flat < num_rho)  # 배열 범위 벗어난 bin 제거
    flat_idx   = theta_flat[valid] * num_rho + rho_flat[valid]  # (θ_idx, ρ_idx) -> 1D 통합 인덱스

    counts = np.bincount(flat_idx, minlength=num_theta * num_rho)  # 각 bin의 투표수 집계
    accum  = counts.reshape(num_theta, num_rho).astype(np.int32)   # 1D -> 2D 누적기 복원

    ti_arr, ri_arr = np.where(accum >= threshold)   # 투표수 threshold 이상인 bin 위치
    if len(ti_arr) == 0:                            # 기준 넘는 직선이 없으면
        return None

    order  = np.argsort(-accum[ti_arr, ri_arr])    # 투표수 내림차순 정렬
    ti_arr = ti_arr[order]                          # 정렬된 θ bin 인덱스
    ri_arr = ri_arr[order]                          # 정렬된 ρ bin 인덱스

    rho_vals   = ((ri_arr - D) * rho_step).astype(np.float32)  # bin 인덱스 -> 실제 ρ 값 복원
    theta_vals = (ti_arr * theta_step).astype(np.float32)       # bin 인덱스 -> 실제 θ 값 복원

    return np.stack([rho_vals, theta_vals], axis=1)[:, None, :]  # (N,1,2) 반환


def nms_lines(lines, rho_thresh=100, angle_thresh_deg=18.5):
    thr  = np.deg2rad(angle_thresh_deg)             # 각도 임계값을 라디안으로 변환
    kept = []                                        # 직선 목록
    for rho, theta in lines[:, 0]:                  # Hough 결과 (N,1,2)에서 (rho,theta) 순회
        if not any(abs(theta - kt) < thr and abs(rho - kr) < rho_thresh
                   for kr, kt in kept):              # 기존 직선과 각도와 거리 둘 다 가까우면 중복
            kept.append((rho, theta))               # 중복이 아니면 보존
    return np.array(kept) if kept else None         # 리스트 -> 배열로 변환 후 반환


def line_intersect(r1, t1, r2, t2):
    A = np.array([[np.cos(t1), np.sin(t1)],         # 직선1 계수
                  [np.cos(t2), np.sin(t2)]])         # 직선2 계수
    b = np.array([r1, r2])                           # 우변 벡터 [ρ₁, ρ₂]
    if abs(np.linalg.det(A)) < 1e-6:                # 행렬식 ≈ 0이면 두 직선이 평행
        return None                                  # 평행이면 교점 없음
    return np.linalg.solve(A, b)                    # 연립방정식 풀어 교점 (x, y) 반환 -> solve method 사용


def find_best_quad(lines_nms, H, W,
                   center_pct=0.10, min_rho_gap=100, min_angle_diff_deg=50):
    if lines_nms is None or len(lines_nms) < 4:     # 직선이 4개 미만이면 사각형 구성 불가
        return None

    def is_inside(x, y):                             # (x,y)가 이미지 영역 안인지 확인
        return 0 <= x <= W and 0 <= y <= H

    best     = None                                  # (최고점수, 코너리스트) 저장용
    min_ang  = np.deg2rad(min_angle_diff_deg)       # 두 직선 쌍 간 최소 각도 차 (라디안)
    img_diag = np.hypot(H, W)                       # 이미지 대각선 길이 (페널티 정규화 기준)
    cx_half  = W * center_pct                       # 중심 허용 범위: H랑 W 중심 10% 지점 직사각형 내부
    cy_half  = H * center_pct                     
    lines    = list(lines_nms)                      # 배열 -> 리스트 (인덱싱 편의)

    for idx in combinations(range(len(lines)), 4):  # NMS 직선 전체에서 4개 조합 선택
        l = [lines[k] for k in idx]                 # 선택된 4개 직선
        for a1i, a2i, b1i, b2i in [(0,1,2,3),(0,2,1,3),(0,3,1,2)]:  # 4개->2+2 분할 3가지 경우
            la1, la2 = l[a1i], l[a2i]              # 그룹A
            lb1, lb2 = l[b1i], l[b2i]              # 그룹B

            if abs((la1[1]+la2[1])/2 - (lb1[1]+lb2[1])/2) < min_ang:
                continue                             # A,B 그룹 평균각 차가 50도 미만이면 스킵 (수직 아님)
            if abs(la1[0]-la2[0]) < min_rho_gap or abs(lb1[0]-lb2[0]) < min_rho_gap:
                continue                             # 같은 그룹 내 ρ 차가 너무 작으면 스킵 (직선 겹침)

            pt_aa = line_intersect(la1[0], la1[1], la2[0], la2[1])
            if pt_aa is not None and is_inside(pt_aa[0], pt_aa[1]):
                continue                             # A 그룹 두 직선이 이미지 내에서 만나면 평행 아님
            pt_bb = line_intersect(lb1[0], lb1[1], lb2[0], lb2[1])
            if pt_bb is not None and is_inside(pt_bb[0], pt_bb[1]):
                continue                             # B 그룹 두 직선이 이미지 내에서 만나면 평행 아님

            # A×B 교점 4개 계산: [la1∩lb1, la1∩lb2, la2∩lb1, la2∩lb2]
            corners_raw = [line_intersect(la[0], la[1], lb[0], lb[1])
                           for la in [la1, la2] for lb in [lb1, lb2]]
            if any(p is None for p in corners_raw): # 어떤 교점이라도 계산 실패 시 스킵
                continue
            if sum(is_inside(x, y) for x, y in corners_raw) < 3:
                continue                             # 4 교점 중 3개 이상이 이미지 내부여야 함

            clamped = [(np.clip(x, 0, W-1), np.clip(y, 0, H-1)) for x, y in corners_raw]  # 이미지 밖 교점을 테두리로 클램프

            quad_cx = sum(x for x, _ in corners_raw) / 4  # 4 교점의 무게중심 x
            quad_cy = sum(y for _, y in corners_raw) / 4  # 4 교점의 무게중심 y
            if abs(quad_cx - W/2) > cx_half or abs(quad_cy - H/2) > cy_half:
                continue                             # 무게중심이 이미지 중앙 10퍼 범위 벗어나면 스킵

            # 신발끈 공식으로 볼록 사각형 넓이 계산
            qx = [clamped[0][0], clamped[1][0], clamped[3][0], clamped[2][0]]  # 꼭짓점 x좌표 (순서 정렬)
            qy = [clamped[0][1], clamped[1][1], clamped[3][1], clamped[2][1]]  # 꼭짓점 y좌표
            area   = 0.5 * abs(sum(qx[i]*qy[(i+1)%4] - qx[(i+1)%4]*qy[i] for i in range(4)))  # 신발끈 공식
            s_area = area / (H * W)                 # 넓이를 이미지 크기로 정규화 -> [0,1]

            center_dist = np.hypot(quad_cx - W/2, quad_cy - H/2)  # 사각형 중심 <-> 이미지 중심 거리
            s_center    = np.clip(
                1.0 - np.log1p(center_dist) / np.log1p(np.hypot(cx_half, cy_half)),
                0.0, 1.0)                           # 중심에 가까울수록 높은 점수 (log 스케일, 중심에 가까울수록 점수 튀게)

            c    = corners_raw
            la1_ = np.hypot(c[0][0]-c[1][0], c[0][1]-c[1][1])  # A방향 변1 길이
            la2_ = np.hypot(c[2][0]-c[3][0], c[2][1]-c[3][1])  # A방향 변2 길이
            lb1_ = np.hypot(c[0][0]-c[2][0], c[0][1]-c[2][1])  # B방향 변1 길이
            lb2_ = np.hypot(c[1][0]-c[3][0], c[1][1]-c[3][1])  # B방향 변2 길이
            len_a = (la1_ + la2_) / 2               # A방향 평균 변 길이
            len_b = (lb1_ + lb2_) / 2               # B방향 평균 변 길이
            ratio        = max(len_a, len_b) / (min(len_a, len_b) + 1e-8)  # 종횡비
            asp_penalty  = abs(ratio - 1.8)          # 명함 표준 비율(1.8) 기준 편차
            side_penalty = (abs(la1_ - la2_) + abs(lb1_ - lb2_)) / img_diag  # 마주보는 변 길이 차 페널티

            score = s_area * s_center - asp_penalty - side_penalty  # 종합 점수
            if best is None or score > best[0]:     # 최고 점수 갱신을 best로
                best = (score, clamped)            

    return best[1] if best is not None else None    # 최고 점수의 코너 4개 반환 (없으면 None)


def sort_corners(pts):
    pts = sorted(pts, key=lambda p: p[1])           # y값 기준 정렬 -> 상위 2개(위), 하위 2개(아래)
    top = sorted(pts[:2], key=lambda p: p[0])       # 위 2개를 x 기준 정렬 -> TL(왼위), TR(오위)
    bot = sorted(pts[2:], key=lambda p: p[0])       # 아래 2개를 x 기준 정렬 -> BL(왼아래), BR(오아래)
    return np.float32([top[0], top[1], bot[1], bot[0]])  # [TL, TR, BR, BL] 순서로 반환

#------------------------------------------------

# TYPE B  (어두운/일반 배경 카드)
# Otsu -> FillHoles -> RegionFilter -> MergeCenterRegions -> FindCorners -> Warp

def fill_holes(binary):
    H, W = binary.shape                              # 이미지 크기
    bg   = np.zeros((H, W), dtype=bool)             # 배경 픽셀 여부 추적 배열 (False로 초기화)
    q    = deque()                                   # BFS 큐

    for u in range(W):                              # 상단 행(v=0)과 하단 행(v=H-1) 순회
        for v in [0, H - 1]:
            if binary[v, u] == 0 and not bg[v, u]: # 테두리 검정 픽셀이면
                bg[v, u] = True                     # 배경으로 표시
                q.append((v, u))                    # BFS 시작점으로 추가
    for v in range(H):                              # 좌측 열(u=0)과 우측 열(u=W-1) 순회
        for u in [0, W - 1]:
            if binary[v, u] == 0 and not bg[v, u]:
                bg[v, u] = True
                q.append((v, u))

    while q:                                         # BFS: 배경 픽셀에서 4방향으로 전파
        v, u = q.popleft()
        for dv, du in [(-1,0),(1,0),(0,-1),(0,1)]: # 상하좌우 4방향 이웃
            nv, nu = v + dv, u + du
            if 0 <= nv < H and 0 <= nu < W and not bg[nv, nu] and binary[nv, nu] == 0:
                bg[nv, nu] = True                   # 연결된 검정 픽셀을 배경으로 확장
                q.append((nv, nu))

    result = binary.copy()                           # 원본 복사
    result[~bg & (binary == 0)] = 255               # 배경이 아닌 검정 픽셀(내부 구멍)을 흰색으로 채움
    return result


def region_filter(binary):
    H, W    = binary.shape                           
    fg      = (binary == 255)                        # 흰 픽셀 마스크
    visited = np.zeros((H, W), dtype=bool)           # 방문 여부
    best    = []                                     # 가장 큰 컴포넌트 저장

    ys, xs = np.where(fg)                           # 흰 픽셀 좌표 목록
    for sy, sx in zip(ys, xs):                      # 모든 흰 픽셀
        if visited[sy, sx]:                          # 이미 방문한 픽셀은 스킵
            continue
        comp = []                                    # 현재 컴포넌트 픽셀 목록
        q = deque([(sy, sx)])                        # BFS 큐 초기화
        visited[sy, sx] = True
        while q:                                     # 8-연결 BFS로 컴포넌트 탐색
            v, u = q.popleft()
            comp.append((v, u))
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    nv, nu = v + dv, u + du
                    if 0 <= nv < H and 0 <= nu < W and not visited[nv, nu] and fg[nv, nu]:
                        visited[nv, nu] = True
                        q.append((nv, nu))
        if len(comp) > len(best):                    # 현재까지 가장 크면
            best = comp                              # 최대 컴포넌트 갱신

    result = np.zeros((H, W), dtype=np.uint8)        # 출력 이미지 초기화
    for v, u in best:                               # 가장 큰 컴포넌트만
        result[v, u] = 255
    return result


def merge_center_regions(binary, base_region, img_area):
    H, W     = binary.shape                         
    cy1, cy2 = H * 3 // 8, H * 5 // 8             # 중앙 세로 (이미지 3/8 ~ 5/8)
    cx1, cx2 = W * 3 // 8, W * 5 // 8             # 중앙 가로 (이미지 3/8 ~ 5/8)
    min_size = int(img_area * 0.01)                 # 무시할 노이즈 컴포넌트 최소 크기 (1%)

    fg      = (binary == 255)                        
    visited = np.zeros((H, W), dtype=bool)           # 방문 여부 추적
    comps   = []                                     # 모든 컴포넌트 목록

    ys, xs = np.where(fg)
    for sy, sx in zip(ys, xs):                      # 모든 흰색 픽셀에서 BFS로 컴포넌트 탐색
        if visited[sy, sx]:
            continue
        comp = []
        q = deque([(sy, sx)])
        visited[sy, sx] = True
        while q:
            v, u = q.popleft()
            comp.append((v, u))
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    nv, nu = v + dv, u + du
                    if 0 <= nv < H and 0 <= nu < W and not visited[nv, nu] and fg[nv, nu]:
                        visited[nv, nu] = True
                        q.append((nv, nu))
        comps.append(comp)                           # 탐색된 컴포넌트 저장

    result = base_region.copy()                      # 기존 최대 영역에서 시작
    for comp in comps:
        if len(comp) < min_size:                     # 너무 작은 노이즈 무시
            continue
        for v, u in comp:
            if cy1 <= v <= cy2 and cx1 <= u <= cx2: # 컴포넌트가 이미지 중앙과 겹치면
                for bv, bu in comp:
                    result[bv, bu] = 255             # 해당 컴포넌트 전체를 결과에 합산
                break
    return result


def find_corners(filled):
    fg       = (filled == 255)                       
    ys, xs   = np.where(fg)                        
    pts      = np.column_stack([xs, ys]).astype(np.float32)  # (x,y) 포인트 배열

    centered = pts - pts.mean(axis=0)               # 평균 빼서 중심화
    Sxx = np.sum(centered[:,0]**2)                  # x 분산
    Syy = np.sum(centered[:,1]**2)                  # y 분산
    Sxy = np.sum(centered[:,0] * centered[:,1])     # x-y 공분산
    mid  = (Sxx + Syy) / 2                          # 고유값 계산용 중간값
    diff = np.sqrt(((Sxx - Syy) / 2)**2 + Sxy**2)  # 고유값 계산용 편차
    lam  = mid + diff                               # 가장 큰 고유값 (주축 방향)

    v = np.array([Sxy, lam - Sxx])                  # 주축 고유벡터 (정규화 전)
    v = v / np.linalg.norm(v)                       # 단위벡터로 정규화

    angle        = np.arctan2(v[1], v[0])           # 주축 각도
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)   # 주축 정렬을 위한 역회전 계수

    rx = centered[:,0] * cos_a - centered[:,1] * sin_a  # 회전 후 x
    ry = centered[:,0] * sin_a + centered[:,1] * cos_a  # 회전 후 y

    tl = pts[np.argmin(rx + ry)]                    # rx+ry 최소 -> 좌상단(TL)
    br = pts[np.argmax(rx + ry)]                    # rx+ry 최대 -> 우하단(BR)
    tr = pts[np.argmax(rx - ry)]                    # rx-ry 최대 -> 우상단(TR)
    bl = pts[np.argmin(rx - ry)]                    # rx-ry 최소 -> 좌하단(BL)

    return np.float32([tl, tr, br, bl])             # [TL, TR, BR, BL] 순서로 반환


# -------------------------------------------------------

# 공통  (TYPE A / TYPE B 모두 사용)
# ComputeHomography -> WarpPerspective

def compute_homography(src_pts, dst_pts):
    A = []                                           # 연립방정식 행렬 A 구성용
    for (x, y), (xp, yp) in zip(src_pts, dst_pts): # 대응점 쌍마다 2개의 방정식 생성
        A.append([-x, -y, -1,  0,  0,  0, xp*x, xp*y, xp])  # x' 방정식 계수
        A.append([ 0,  0,  0, -x, -y, -1, yp*x, yp*y, yp])  # y' 방정식 계수
    A = np.array(A)                                  # 리스트 -> (8, 9) 행렬
    _, _, Vt = np.linalg.svd(A)                     # SVD 분해: A = U * Σ Vt
    H = Vt[-1].reshape(3, 3)                        # 최소 특이값에 해당하는 벡터 -> 3×3 행렬
    return H / H[2, 2]                              # H[2,2]=1이 되도록 정규화


def warp_perspective(img, H, out_size):
    out_w, out_h = out_size                          # 출력 이미지 가로, 세로 크기
    result = np.zeros((out_h, out_w, 3), dtype=np.uint8)  # 출력 이미지 초기화 (검정)
    H_inv  = np.linalg.inv(H)                       # 역 Homography

    us, vs = np.meshgrid(np.arange(out_w), np.arange(out_h))  # 출력 이미지 픽셀 격자
    ones   = np.ones_like(us)                        # 동차좌표용 1 배열
    coords = np.stack([us, vs, ones], axis=-1).reshape(-1, 3).T  # (3, out_w*out_h) 동차좌표

    src  = H_inv @ coords                            # 출력 픽셀 -> 입력 픽셀 좌표 변환
    src /= src[2]                                    # 동차좌표 정규화 (w로 나누기)

    sx    = np.round(src[0]).astype(int)             # 입력 이미지 x좌표 (반올림)
    sy    = np.round(src[1]).astype(int)             # 입력 이미지 y좌표 (반올림)
    valid = (sx >= 0) & (sx < img.shape[1]) & (sy >= 0) & (sy < img.shape[0])  # 유효 범위 마스크
    result[vs.flatten()[valid], us.flatten()[valid]] = img[sy[valid], sx[valid]]  # 유효 픽셀만 복사
    return result                                 
