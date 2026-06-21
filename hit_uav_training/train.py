"""
Step 3 — Fine-tune YOLOv8 on the HIT-UAV dataset.

Starts from yolov8n.pt (COCO pre-trained) and fine-tunes on the converted
HIT-UAV YOLO dataset produced by convert.py.

Saved weights
-------------
    hit_uav_training/runs/hit_uav/weights/best.pt   (training output)
    models/yolov8_hit_uav_best.pt                   (copied here for easy use)

Run (from project root):
    python hit_uav_training/train.py

Run (from hit_uav_training/ directory):
    python train.py
"""

import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

# ── Config ───────────────────────────────────────────────────────────────────

# Resolves correctly whether run from project root or from hit_uav_training/
_HERE = Path(__file__).parent
DATA_YAML  = _HERE / "data" / "yolo" / "data.yaml"
RUN_DIR    = _HERE / "runs"
RUN_NAME   = "hit_uav"
FINAL_DEST = _HERE.parent / "models" / "yolov8_hit_uav_best.pt"

BASE_WEIGHTS = "yolov8n.pt"

TRAIN_CFG = dict(
    epochs  = 100,
    imgsz   = 640,
    batch   = 16,
    device  = "cuda" if __import__("torch").cuda.is_available() else "cpu",
    workers = 2,
    exist_ok= True,
    save    = True,
    hsv_h   = 0.0,
    hsv_s   = 0.0,
    hsv_v   = 0.3,
    fliplr  = 0.5,
    flipud  = 0.0,
    mosaic  = 0.5,
    weight_decay = 0.0005,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def patch_data_yaml(yaml_path: Path) -> None:
    """
    Rewrite data.yaml so its 'path:' field is the absolute path of the
    directory that contains images/ and labels/.  This makes the file
    portable across machines — ultralytics resolves 'path' relative to CWD,
    not relative to the yaml file, so a relative value breaks when the script
    is called from a different directory.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    correct_path = str(yaml_path.parent.resolve())
    if data.get("path") != correct_path:
        data["path"] = correct_path
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"Patched data.yaml path -> {correct_path}")


def check_prerequisites() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {DATA_YAML}\n"
            "Run  python hit_uav_training/convert.py  first."
        )
    patch_data_yaml(DATA_YAML)
    FINAL_DEST.parent.mkdir(parents=True, exist_ok=True)


# ── Training ─────────────────────────────────────────────────────────────────

def train() -> Path:
    check_prerequisites()

    print(f"Base weights : {BASE_WEIGHTS}")
    print(f"Dataset      : {DATA_YAML}")
    print(f"Epochs       : {TRAIN_CFG['epochs']}")
    print(f"Device       : {TRAIN_CFG['device']}")
    print(f"Output dir   : {RUN_DIR / RUN_NAME}\n")

    model = YOLO(BASE_WEIGHTS)
    model.train(
        data=str(DATA_YAML),
        project=str(RUN_DIR),
        name=RUN_NAME,
        **TRAIN_CFG,
    )

    best_pt = RUN_DIR / RUN_NAME / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError(f"Training finished but best.pt not found at {best_pt}")

    return best_pt


def export_weights(best_pt: Path) -> None:
    shutil.copy2(best_pt, FINAL_DEST)
    print(f"\nWeights copied -> {FINAL_DEST}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    best_pt = train()
    export_weights(best_pt)
