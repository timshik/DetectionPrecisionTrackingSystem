"""
motion_detection — thermal drone tracking pipeline.

Public API:
    run()  : execute the full pipeline on the configured video.

Example:
    from motion_detection import run
    run()
"""

from .pipeline import run

__all__ = ["run"]
