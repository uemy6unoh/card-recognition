import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _imread(path):
    return cv2.imdecode(np.fromfile(os.path.abspath(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def _imwrite(path, img):
    cv2.imencode('.jpg', img)[1].tofile(os.path.abspath(path))

IMG_DIR = os.path.join(SCRIPT_DIR, "businesscard (1)")
OUT_DIR = os.path.join(SCRIPT_DIR, "ch_Otsu_output")
os.makedirs(OUT_DIR, exist_ok=True)

CLAHE_OBJ = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32, 32))
WR_THRESH = 0.6

def center_black_ratio(otsu):
    H, W = otsu.shape
    patch = otsu[int(H*0.45):int(H*0.55), int(W*0.45):int(W*0.55)]
    return np.sum(patch == 0) / patch.size

results, modes, flipped = {}, {}, {}
for i in range(1, 22):
    gray = cv2.cvtColor(_imread(os.path.join(IMG_DIR, f"BC{i}.jpg")), cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(CLAHE_OBJ.apply(gray), 3)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    was_flipped = False
    if center_black_ratio(otsu) >= 0.9:
        otsu = cv2.bitwise_not(otsu)
        was_flipped = True
    mode = "type_a" if np.sum(otsu == 255) / otsu.size >= WR_THRESH else "type_b"
    results[i] = otsu
    modes[i] = mode
    flipped[i] = was_flipped
    _imwrite(os.path.join(OUT_DIR, f"BC{i}_otsu.jpg"), otsu)
    print(f"BC{i} → {mode}{'  [반전]' if was_flipped else ''}")

fig, axes = plt.subplots(3, 7, figsize=(24, 10))
for i in range(1, 22):
    ax = axes[(i-1)//7][(i-1)%7]
    ax.imshow(results[i], cmap='gray')
    title = f"BC{i} ({modes[i]})"
    if flipped[i]:
        title += "\n[반전]"
    ax.set_title(title, fontsize=7,
                 color='navy' if modes[i] == 'type_a' else 'darkgreen')
    ax.axis('off')
plt.suptitle("Checkpoint: Otsu (+ 반전 판단)  |  navy=typeA  green=typeB", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "_grid.png"), dpi=150)
plt.show()
