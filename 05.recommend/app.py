"""
Compatibility entrypoint for Streamlit Cloud.

Streamlit is currently configured to run `05.recommend/app.py`.
The real app moved to `06.recommend/app.py`, so this launcher forwards
execution without requiring immediate cloud-side setting changes.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
TARGET_DIR = CURRENT_DIR.parent / "06.recommend"
TARGET_APP = TARGET_DIR / "app.py"

if not TARGET_APP.exists():
    raise FileNotFoundError(f"Target Streamlit app not found: {TARGET_APP}")

# Ensure sibling imports inside 06.recommend/app.py keep working.
if str(TARGET_DIR) not in sys.path:
    sys.path.insert(0, str(TARGET_DIR))

runpy.run_path(str(TARGET_APP), run_name="__main__")
