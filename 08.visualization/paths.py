"""数据路径（相对项目根目录）。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_WITH_DIMS = ROOT / "03.final_labels/poems_structured_with_dims.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
