from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "03.final_labels/poems_structured_shangxi_wip.csv"
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
LEXICON_JSON = HERE / "emotion_lexicon.json"
DIM_KEYWORDS_JSON = HERE / "dim_keywords.json"
SCENE_PRESETS_JSON = HERE / "scene_presets.json"
