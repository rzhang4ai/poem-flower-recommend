"""
step2b_bert/poem_token_compare.py
---------------------------------
对诗词「正文」用 BERT-CCPoem 提取 token 重要性（注意力聚合 top5 / top10），
并输出 CSV 与每首 [CLS] 向量。

默认数据：
  00.poems_dataset/poems_dataset_merged_done.csv

运行：
  cd /Users/rzhang/Documents/poem-flower-recommend
  source flower_env/bin/activate
  python 02.sample_label_phase2/step2b_bert/poem_token_compare.py

  python 02.sample_label_phase2/step2b_bert/poem_token_compare.py --input path/to.csv

输出（output/）：
  ccpoem_token_importance.csv
  ccpoem_cls_embeddings.npy
  poem_cls_index.csv
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HF_CACHE = MODELS_DIR / ".hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "hub")

DEFAULT_POEMS_CSV = PROJECT_ROOT / "00.poems_dataset" / "poems_dataset_merged_done.csv"
BERT_CCPOEM_PATH = MODELS_DIR / "bert_ccpoem"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

SKIP_TOKENS = {
    "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]",
    "，", "。", "！", "？", "、", "；", "：",
    "（", "）", "(", ")", "「", "」", "『", "』",
    "·", "—", "…", "《", "》", " ", "\n",
}


def load_model(model_path: Path, name: str):
    if not model_path.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_path}")
    print(f"[加载] {name}: {model_path}")
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    mdl = AutoModel.from_pretrained(
        str(model_path), trust_remote_code=True, output_attentions=True
    ).to(DEVICE)
    mdl.eval()
    return tok, mdl


def extract_poem_features(text: str, tok, mdl, top_k: int = 10):
    """
    单次前向同时提取：
      1) CLS 句向量（hidden_size 维）
      2) token 重要性 top-k（基于最后一层注意力均值）
    """
    text = str(text).strip()
    if not text:
        hidden_size = getattr(mdl.config, "hidden_size", 768)
        return np.zeros(hidden_size, dtype=np.float32), []

    inputs = tok(text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0].cpu().tolist())

    with torch.no_grad():
        out = mdl(**inputs, output_attentions=True, return_dict=True)

    cls_vec = out.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().float().numpy()

    attn = out.attentions[-1].squeeze().mean(0).mean(0).cpu().float().numpy()
    merged = defaultdict(float)

    for tok_str, score in zip(tokens, attn):
        if tok_str in SKIP_TOKENS:
            continue
        if tok_str.startswith("##"):
            tok_str = tok_str[2:]
        if not tok_str.strip():
            continue
        merged[tok_str] = max(merged[tok_str], float(score))

    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return cls_vec, ranked[:top_k]


def main():
    ap = argparse.ArgumentParser(description="BERT-CCPoem 诗词正文 token 重要性")
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_POEMS_CSV,
        help=f"诗歌 CSV（需含「正文」列），默认: {DEFAULT_POEMS_CSV.name}",
    )
    args = ap.parse_args()

    t0 = time.time()
    print(f"device={DEVICE}")
    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(f"找不到输入 CSV: {src}")

    df = pd.read_csv(src)
    if "正文" not in df.columns:
        raise ValueError("CSV 必须包含「正文」列")
    n = len(df)
    print(f"输入: {src}")
    print(f"样本数: {n}")

    tok_cc, mdl_cc = load_model(BERT_CCPOEM_PATH, "BERT-CCPoem(正文)")

    rows = []
    cls_cc_all = []
    every = max(1, n // 10)

    for i, row in df.iterrows():
        poem = str(row.get("正文", ""))

        cls_cc, cc_top10 = extract_poem_features(poem, tok_cc, mdl_cc, top_k=10)
        cls_cc_all.append(cls_cc)

        cc_tokens = [t for t, _ in cc_top10]

        rows.append({
            "ID": row.get("ID", i),
            "诗名": row.get("诗名", ""),
            "花名": row.get("花名", ""),
            "朝代": row.get("朝代", ""),
            "正文_preview": poem[:40].replace("\n", " "),
            "ccpoem_top5": "、".join(cc_tokens[:5]),
            "ccpoem_top10": " ".join(cc_tokens),
            "ccpoem_top10_with_score": json.dumps(cc_top10, ensure_ascii=False),
            "ccpoem_cls_l2": round(float(np.linalg.norm(cls_cc)), 6),
        })

        if (i + 1) % every == 0 or (i + 1) == n:
            print(f"进度 {i+1}/{n}")

    out_csv = OUTPUT_DIR / "ccpoem_token_importance.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 弃用旧文件名，避免混淆
    legacy = OUTPUT_DIR / "new_token_importance.csv"
    if legacy.exists():
        legacy.unlink()
        print(f"（已删除旧文件 {legacy.name}）")

    cls_cc_np = np.asarray(cls_cc_all, dtype=np.float32)
    out_cc_npy = OUTPUT_DIR / "ccpoem_cls_embeddings.npy"
    np.save(out_cc_npy, cls_cc_np)
    print(f"\n✅ 已输出: {out_csv}")
    print(f"✅ 已输出: {out_cc_npy}  shape={cls_cc_np.shape}")

    cls_index_df = pd.DataFrame({
        "row_idx": list(range(n)),
        "ID": df["ID"].values if "ID" in df.columns else list(range(n)),
        "诗名": df["诗名"].values if "诗名" in df.columns else [""] * n,
        "花名": df["花名"].values if "花名" in df.columns else [""] * n,
        "朝代": df["朝代"].values if "朝代" in df.columns else [""] * n,
    })
    out_index_csv = OUTPUT_DIR / "poem_cls_index.csv"
    cls_index_df.to_csv(out_index_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 已输出: {out_index_csv}")
    print(f"总耗时: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
