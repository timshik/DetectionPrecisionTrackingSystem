"""
Step 2 — Convert HIT-UAV XML annotations -> YOLO format.

Uses the official train/val/test splits already defined in ImageSets/Main/.

Input layout (after download.py):
    hit_uav_training/data/raw/HIT-UAV/normal_xml/
        JPEGImages/      *.jpg   (640 x 512)
        Annotations/     *.xml   (Pascal-VOC, pixel coords xmin/ymin/xmax/ymax)
        ImageSets/Main/
            train.txt    val.txt    test.txt   (stem names, one per line)

Output layout:
    hit_uav_training/data/yolo/
        images/  train/  val/  test/
        labels/  train/  val/  test/
        data.yaml

Classes
-------
    0  Person
    1  Bicycle
    2  Car
    3  OtherVehicle

Run:
    python hit_uav_training/convert.py
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────

NORMAL_XML  = Path("hit_uav_training/data/raw/HIT-UAV/normal_xml")
IMG_DIR     = NORMAL_XML / "JPEGImages"
ANN_DIR     = NORMAL_XML / "Annotations"
SPLITS_DIR  = NORMAL_XML / "ImageSets" / "Main"
YOLO_DIR    = Path("hit_uav_training/data/yolo")

# ── Classes ───────────────────────────────────────────────────────────────────

CLASSES: Dict[str, int] = {
    "Person":       0,
    "Bicycle":      1,
    "Car":          2,
    "OtherVehicle": 3,
}
SKIP_CLASSES = {"DontCare"}


# ── XML -> YOLO conversion ────────────────────────────────────────────────────

def parse_xml(xml_path: Path) -> Optional[List[Tuple[int, float, float, float, float]]]:
    """
    Parse one Pascal-VOC XML file and return normalised YOLO boxes.

    Returns
    -------
    List of (class_id, cx, cy, bw, bh) with values in [0, 1],
    or None if the file contains no usable annotations.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    W = int(size.findtext("width"))
    H = int(size.findtext("height"))

    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name in SKIP_CLASSES or name not in CLASSES:
            continue

        bnd  = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        cx = max(0.0, min(1.0, (xmin + xmax) / 2 / W))
        cy = max(0.0, min(1.0, (ymin + ymax) / 2 / H))
        bw = max(0.0, min(1.0, (xmax - xmin) / W))
        bh = max(0.0, min(1.0, (ymax - ymin) / H))

        boxes.append((CLASSES[name], cx, cy, bw, bh))

    return boxes if boxes else None


def to_label_string(boxes: List[Tuple[int, float, float, float, float]]) -> str:
    return "\n".join(
        f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        for cls, cx, cy, bw, bh in boxes
    )


# ── Split processing ──────────────────────────────────────────────────────────

def load_split(name: str) -> List[str]:
    """Load stem names from ImageSets/Main/<name>.txt."""
    path = SPLITS_DIR / f"{name}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def process_split(split: str, stems: List[str]) -> None:
    img_out = YOLO_DIR / "images" / split
    lbl_out = YOLO_DIR / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for stem in stems:
        img_src = IMG_DIR / f"{stem}.jpg"
        xml_src = ANN_DIR / f"{stem}.xml"

        if not img_src.exists() or not xml_src.exists():
            skipped += 1
            continue

        boxes = parse_xml(xml_src)
        if boxes is None:
            skipped += 1
            continue

        shutil.copy2(img_src, img_out / img_src.name)
        (lbl_out / f"{stem}.txt").write_text(to_label_string(boxes))

    kept = len(stems) - skipped
    print(f"  {split:5s}: {kept} images written  ({skipped} skipped)")


# ── data.yaml ─────────────────────────────────────────────────────────────────

def write_yaml() -> Path:
    data = {
        "path":  ".",  # relative to data.yaml — works on any machine
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASSES),
        "names": {v: k for k, v in CLASSES.items()},
    }
    yaml_path = YOLO_DIR / "data.yaml"
    with open(yaml_path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
    return yaml_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not ANN_DIR.exists():
        raise FileNotFoundError(
            f"Annotations not found at {ANN_DIR}\n"
            "Run  python hit_uav_training/download.py  first."
        )

    print(f"Source : {NORMAL_XML}")
    print(f"Output : {YOLO_DIR}\n")

    for split in ("train", "val", "test"):
        stems = load_split(split)
        print(f"  {split:5s}: {len(stems)} entries in ImageSets")
        process_split(split, stems)

    yaml_path = write_yaml()
    print(f"\ndata.yaml written -> {yaml_path}")
    print("\nNext step:  python hit_uav_training/train.py")
