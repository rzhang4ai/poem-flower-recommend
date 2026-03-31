"""
step2e_sentiment / 05_inference_ccpoem.py
==========================================
使用微调后的 BERT-CCPoem 模型对项目 1075 首诗词做情感推断，
并将结果追加到 sentiment_per_poem.csv。

前置条件
--------
  04_finetune_ccpoem.py 已成功运行，models/ccpoem_sentiment_ft/ 目录存在。

输入
----
  models/ccpoem_sentiment_ft/                  : 微调后的模型
  00.poems_dataset/poems_dataset_merged_done.csv : 1075 首原始诗词
  output/results/sentiment_per_poem.csv          : 结果总表（将追加新列）

新增列（追加到 sentiment_per_poem.csv）
--------------------------------------
  ccpoem_label         : 5 类标签 id（0=负面 … 4=正面）
  ccpoem_label_zh      : 中文标签（负面/隐性负面/中性/隐性正面/正面）
  ccpoem_conf          : 最大 softmax 概率（置信度）
  ccpoem_polarity      : 3 类极性（positive / neutral / negative）
  ccpoem_prob_0 … _4   : 各类 softmax 概率（5列）

输出
----
  output/results/sentiment_per_poem.csv      : 追加 ccpoem_* 列
  output/finetune_ccpoem/inference_report.txt: 推断统计报告
"""

import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent.parent

MODEL_PATH = ROOT_DIR / "models" / "ccpoem_sentiment_ft"
BASE_TOKENIZER_PATH = ROOT_DIR / "models" / "bert_ccpoem"
POEMS_CSV = ROOT_DIR / "00.poems_dataset" / "poems_dataset_merged_done.csv"
RESULT_CSV = BASE_DIR / "output" / "results" / "sentiment_per_poem.csv"
REPORT_OUT = BASE_DIR / "output" / "finetune_ccpoem" / "inference_report.txt"

# ── 标签定义 ──────────────────────────────────────────────────────────────
LABEL_ZH = ["负面", "隐性负面", "中性", "隐性正面", "正面"]
POLARITY_MAP = {0: "negative", 1: "negative", 2: "neutral", 3: "positive", 4: "positive"}

MAX_LEN = 128
BATCH_SIZE = 32


def clean_poem_text(text: str) -> str:
    """清洗诗词正文：去除标点和空白，只保留汉字。"""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^\u4e00-\u9fff]", "", text)


def resolve_model_dir(model_root: Path) -> Path:
    """
    优先使用 model_root 本身；若未完成最终 save_model，则回退到最新 checkpoint。
    """
    root_cfg = model_root / "config.json"
    if root_cfg.exists():
        return model_root

    ckpt_dir = model_root / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"未找到可用模型目录：{model_root}\n"
            "请先运行 04_finetune_ccpoem.py 并确保模型已保存。"
        )

    candidates = []
    for p in ckpt_dir.glob("checkpoint-*"):
        cfg = p / "config.json"
        if cfg.exists():
            try:
                step = int(p.name.split("-")[-1])
            except ValueError:
                step = -1
            candidates.append((step, p))

    if not candidates:
        raise FileNotFoundError(
            f"{ckpt_dir} 下未找到可用 checkpoint（缺少 config.json）。"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def resolve_tokenizer_dir(model_dir: Path, model_root: Path, base_tokenizer: Path) -> Path:
    """
    checkpoint 里可能只有权重，不含 tokenizer；按优先级选择可用 tokenizer 目录。
    """
    for p in [model_dir, model_root, base_tokenizer]:
        if (p / "tokenizer_config.json").exists() or (p / "vocab.txt").exists():
            return p
    raise FileNotFoundError(
        "未找到可用 tokenizer 目录。请确认以下路径之一存在 tokenizer 文件：\n"
        f"- {model_dir}\n- {model_root}\n- {base_tokenizer}"
    )


def main():
    # ── 检测硬件 ──────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS (Apple Silicon GPU) 已启用")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA GPU 已启用")
    else:
        device = torch.device("cpu")
        print("使用 CPU 推断")

    # ── 检查文件 ──────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"微调模型不存在：{MODEL_PATH}\n请先运行 04_finetune_ccpoem.py 完成训练。"
        )
    if not POEMS_CSV.exists():
        raise FileNotFoundError(f"原始诗词数据不存在：{POEMS_CSV}")
    if not RESULT_CSV.exists():
        raise FileNotFoundError(f"结果表不存在：{RESULT_CSV}")

    # ── 加载模型 ──────────────────────────────────────────────────────────
    model_dir = resolve_model_dir(MODEL_PATH)
    tokenizer_dir = resolve_tokenizer_dir(model_dir, MODEL_PATH, BASE_TOKENIZER_PATH)
    print(f"加载微调模型：{model_dir}")
    print(f"加载 tokenizer：{tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), trust_remote_code=True)
    model.to(device)
    model.eval()
    print(f"模型标签数：{model.config.num_labels}")

    # ── 读取数据 ──────────────────────────────────────────────────────────
    poems_df = pd.read_csv(POEMS_CSV)
    print(f"原始诗词数据：{len(poems_df)} 行")
    poems_df["_text"] = poems_df["正文"].apply(clean_poem_text)
    valid_cnt = (poems_df["_text"].str.len() > 0).sum()
    print(f"有效诗文：{valid_cnt} / {len(poems_df)}")

    # ── 批量推断 ──────────────────────────────────────────────────────────
    texts = poems_df["_text"].tolist()
    all_probs = []
    print(f"开始推断（batch_size={BATCH_SIZE}）...")
    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        all_probs.append(probs)
        if (start // BATCH_SIZE + 1) % 5 == 0:
            print(f"  已处理 {start + len(batch_texts)}/{len(texts)}")

    all_probs = np.vstack(all_probs)  # (N, 5)
    pred_ids = np.argmax(all_probs, axis=-1)
    pred_conf = all_probs.max(axis=-1)

    poems_df["ccpoem_label"] = pred_ids
    poems_df["ccpoem_label_zh"] = [LABEL_ZH[i] for i in pred_ids]
    poems_df["ccpoem_conf"] = pred_conf.round(4)
    poems_df["ccpoem_polarity"] = [POLARITY_MAP[i] for i in pred_ids]
    for i in range(5):
        poems_df[f"ccpoem_prob_{i}"] = all_probs[:, i].round(4)

    # ── 追加到 sentiment_per_poem.csv ─────────────────────────────────────
    result_df = pd.read_csv(RESULT_CSV)
    print(f"sentiment_per_poem.csv 当前行数：{len(result_df)}")

    cc_cols = [
        "ccpoem_label",
        "ccpoem_label_zh",
        "ccpoem_conf",
        "ccpoem_polarity",
        "ccpoem_prob_0",
        "ccpoem_prob_1",
        "ccpoem_prob_2",
        "ccpoem_prob_3",
        "ccpoem_prob_4",
    ]

    for col in cc_cols:
        if col in result_df.columns:
            result_df.drop(columns=[col], inplace=True)
        result_df[col] = poems_df[col].values

    result_df.to_csv(RESULT_CSV, index=False)
    print(f"已更新 sentiment_per_poem.csv，追加列：{cc_cols}")

    # ── 统计报告 ──────────────────────────────────────────────────────────
    total = len(result_df)
    lbl_cnt = Counter(result_df["ccpoem_label_zh"])
    pol_cnt = Counter(result_df["ccpoem_polarity"])
    avg_conf = float(result_df["ccpoem_conf"].mean())

    lines = [
        "=" * 60,
        "BERT-CCPoem 推断报告（1075 首项目诗词）",
        "=" * 60,
        f"模型路径：{model_dir}",
        f"推断诗词数：{total}",
        "",
        "5 类标签分布：",
    ]
    for zh, n in sorted(lbl_cnt.items(), key=lambda x: -x[1]):
        lines.append(f"  {zh:<8}: {n:>4}首  {n/total*100:>5.1f}%")

    lines += ["", "3 类极性分布："]
    for pol, n in sorted(pol_cnt.items(), key=lambda x: -x[1]):
        lines.append(f"  {pol:<12}: {n:>4}首  {n/total*100:>5.1f}%")

    lines += [
        "",
        f"平均置信度：{avg_conf:.4f}",
        "",
        "说明：",
        "  - ccpoem_label/ccpoem_label_zh: 5 类预测标签",
        "  - ccpoem_polarity: 3 类极性映射",
        "  - ccpoem_prob_0..4: 每类 softmax 概率，可用于后续融合",
    ]

    report = "\n".join(lines)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n报告已保存至：{REPORT_OUT}")


if __name__ == "__main__":
    main()
