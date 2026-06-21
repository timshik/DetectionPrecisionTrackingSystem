"""
Entry point — works both ways:
    python -m motion_detection          (from project root)
    python motion_detection/__main__.py (direct file run)
"""

import sys
from pathlib import Path

# When run directly the package is not on sys.path yet; add the project root.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from motion_detection.pipeline import run
else:
    from .pipeline import run

run()
