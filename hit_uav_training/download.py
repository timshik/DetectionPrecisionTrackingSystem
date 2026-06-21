"""
Step 1 — Download & extract the HIT-UAV dataset.

Source : GitHub release v1.2.1
URL    : https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset
Size   : ~2,898 infrared thermal images, 4 classes
Auth   : none — fully public

Run:
    python hit_uav_training/download.py
"""

import zipfile
from pathlib import Path

import requests

DOWNLOAD_URL = (
    "https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset"
    "/releases/download/v1.2.1/HIT-UAV.zip"
)

# Paths (all relative to project root)
RAW_DIR  = Path("hit_uav_training/data/raw")
ZIP_PATH = RAW_DIR / "HIT-UAV.zip"


def download(url: str, dest: Path) -> None:
    """Stream-download url → dest, printing a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk = 1024 * 64  # 64 KB chunks

    with open(dest, "wb") as fh:
        for data in response.iter_content(chunk_size=chunk):
            fh.write(data)
            downloaded += len(data)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:5.1f}%  {downloaded // 1_000_000} / {total // 1_000_000} MB",
                      end="", flush=True)
    print()


def extract(zip_path: Path, dest: Path) -> None:
    """Extract zip archive, skipping files already present."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for i, member in enumerate(members):
            target = dest / member.filename
            if not target.exists():
                zf.extract(member, dest)
            pct = (i + 1) / len(members) * 100
            print(f"\r  extracting {pct:5.1f}%  ({i + 1}/{len(members)})",
                  end="", flush=True)
    print()


def print_structure(root: Path, max_files: int = 6) -> None:
    """Print a short directory tree so we can verify the layout."""
    print(f"\nExtracted layout under {root}:")
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        depth = len(rel.parts)
        if depth > 3:
            continue
        indent = "  " * (depth - 1)
        if p.is_dir():
            print(f"{indent}📁 {p.name}/")
        else:
            print(f"{indent}   {p.name}")


if __name__ == "__main__":
    # ── Download ──────────────────────────────────────────────────────────────
    if ZIP_PATH.exists():
        print(f"Zip already present: {ZIP_PATH}  (skipping download)")
    else:
        print(f"Downloading HIT-UAV dataset…")
        download(DOWNLOAD_URL, ZIP_PATH)
        print(f"Saved -> {ZIP_PATH}")

    # ── Extract ───────────────────────────────────────────────────────────────
    if any(RAW_DIR.iterdir()) and len(list(RAW_DIR.iterdir())) > 1:
        print(f"Already extracted in {RAW_DIR}  (skipping extraction)")
    else:
        print("Extracting…")
        extract(ZIP_PATH, RAW_DIR)
        print(f"Extracted -> {RAW_DIR}")

    print_structure(RAW_DIR)
    print("\nNext step:  python hit_uav_training/convert.py")
