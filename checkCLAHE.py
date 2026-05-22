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
OUT_DIR = os.path.join(SCRIPT_DIR, "ch_CLAHE_output")
os.makedirs(OUT_DIR, exist_ok=True)

CLAHE_OBJ = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32, 32))

clahe_results = {}
for i in range(1, 22):
    gray = cv2.cvtColor(_imread(os.path.join(IMG_DIR, f"BC{i}.jpg")), cv2.COLOR_BGR2GRAY)
    result = CLAHE_OBJ.apply(gray)
    clahe_results[i] = result
    _imwrite(os.path.join(OUT_DIR, f"BC{i}_clahe.jpg"), result)
    print(f"BC{i} saved")

fig, axes = plt.subplots(3, 7, figsize=(24, 10))
for i in range(1, 22):
    ax = axes[(i-1)//7][(i-1)%7]
    ax.imshow(clahe_results[i], cmap='gray')
    ax.set_title(f"BC{i}", fontsize=8)
    ax.axis('off')
plt.suptitle("Checkpoint: CLAHE (clipLimit=1.0, tile=32x32)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "_grid.png"), dpi=150)
plt.show()
