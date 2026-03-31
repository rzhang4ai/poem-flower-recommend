"""
step2e_sentiment / 05_inference_siku.py
========================================
使用微调后的 SikuRoBERTa 模型对项目 1075 首诗词做情感推断，
并将结果追加到 sentiment_per_poem.csv。

前置条件
--------
  04_finetune_siku.py 已成功运行，models/siku_sentiment_ft/ 目录存在。

输入
----
  models/siku_sentiment_ft/           : 微调后的模型
  00.poems_dataset/poems_dataset_merged_done.csv : 1075 首原始诗词
  output/results/sentiment_per_poem.csv          : Phase 1 结果（将追加新列）

新增列（追加到 sentiment_per_poem.csv）
--------------------------------------
  siku_label         : 5 类标签 id（0=负面 … 4=正面）
  siku_label_zh      : 中文标签（负面/隐性负面/中性/隐性正面/正面）
  siku_conf          : 最大 softmax 概率（置信度）
  siku_polarity      : 3 类极性（positive / neutral / negative）
  siku_prob_0 … _4  : 各类 softmax 概率（5列，供后续分析）

输出
----
  output/results/sentiment_per_poem.csv   : 已追加 siku_* 列
  output/finetune/inference_report.txt    : 推断结果统计

运行方式
--------
  source flower_env/bin/activate
  cd 02.sample_label_phase2/step2e_sentiment
  python 05_inference_siku.py
"""

import os, warnings, re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
ROOT_DIR   = BASE_DIR.parent.parent
MODEL_PATH = ROOT_DIR / "models" / "siku_sentiment_ft"
POEMS_CSV  = ROOT_DIR / "00.poems_dataset" / "poems_dataset_merged_done.csv"
RESULT_CSV = BASE_DIR / "output" / "results" / "sentiment_per_poem.csv"
REPORT_OUT = BASE_DIR / "output" / "finetune" / "inference_report.txt"

# ── 标签定义 ──────────────────────────────────────────────────
LABEL_LIST = ["negative", "implicit_negative", "neutral", "implicit_positive", "positive"]
LABEL_ZH   = ["负面", "隐性负面", "中性", "隐性正面", "正面"]
# 3类极性映射
POLARITY_MAP = {
    0: "negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "positive",
}

MAX_LEN    = 128   # 项目诗词比 FSPC 长，给宽裕空间
BATCH_SIZE = 32

# ── 检测硬件 ──────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Apple Silicon GPU) 已启用")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA GPU 已启用")
else:
    device = torch.device("cpu")
    print("使用 CPU 推断")

# ── 检查模型目录 ──────────────────────────────────────────────
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"微调模型不存在：{MODEL_PATH}\n"
        "请先运行 04_finetune_siku.py 完成训练。"
    )

# ── 加载 Tokenizer & 模型 ─────────────────────────────────────
print(f"加载微调模型：{MODEL_PATH}")
tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.to(device)
model.eval()
print(f"模型标签数：{model.config.num_labels}")

# ── 读取 1075 首诗词 ──────────────────────────────────────────
poems_df = pd.read_csv(POEMS_CSV)
print(f"原始诗词数据：{len(poems_df)} 行，列：{list(poems_df.columns)}")

def clean_poem_text(text: str) -> str:
    """清洗诗词正文：去除换行/空格/标点，只保留汉字"""
    if not isinstance(text, str):
        return ""
    # 保留汉字，丢弃标点和空白
    return re.sub(r"[^\u4e00-\u9fff]", "", text)

poems_df["_text"] = poems_df["正文"].apply(clean_poem_text)

# 去除空文本
valid_mask = poems_df["_text"].str.len() > 0
print(f"有效诗文：{valid_mask.sum()} / {len(poems_df)}")

# ── 批量推断 ──────────────────────────────────────────────────
texts = poems_df["_text"].tolist()
all_probs  = []

print(f"开始推断（batch_size={BATCH_SIZE}）...")
for start in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[start:start + BATCH_SIZE]
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
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()

    all_probs.append(probs)
    if (start // BATCH_SIZE + 1) % 5 == 0:
        print(f"  已处理 {start + len(batch_texts)}/{len(texts)}")

all_probs  = np.vstack(all_probs)   # (1075, 5)
pred_ids   = np.argmax(all_probs, axis=-1)
pred_conf  = all_probs.max(axis=-1)

# ── 整理结果 ──────────────────────────────────────────────────
poems_df["siku_label"]    = pred_ids
poems_df["siku_label_zh"] = [LABEL_ZH[i] for i in pred_ids]
poems_df["siku_conf"]     = pred_conf.round(4)
poems_df["siku_polarity"] = [POLARITY_MAP[i] for i in pred_ids]
for c in range(5):
    poems_df[f"siku_prob_{c}"] = all_probs[:, c].round(4)

# ── 追加到 sentiment_per_poem.csv ────────────────────────────
result_df = pd.read_csv(RESULT_CSV)
print(f"\nsentiment_per_poem.csv 当前行数：{len(result_df)}")

# 用 id 对齐（poems_df 的 ID 列）
siku_cols = ["siku_label","siku_label_zh","siku_conf","siku_polarity",
             "siku_prob_0","siku_prob_1","siku_prob_2","siku_prob_3","siku_prob_4"]

# 移除已有的 siku_* 列（防止重复追加）
for col in siku_cols:
    if col in result_df.columns:
        result_df.drop(columns=[col], inplace=True)

# 按行序对齐（两者行数相同，按索引对齐）
for col in siku_cols:
    result_df[col] = poems_df[col].values

result_df.to_csv(RESULT_CSV, index=False)
print(f"已更新 sentiment_per_poem.csv，追加列：{siku_cols}")

# ── 统计报告 ──────────────────────────────────────────────────
from collections import Counter
total = len(result_df)
lbl_cnt = Counter(result_df["siku_label_zh"])
pol_cnt = Counter(result_df["siku_polarity"])

lines = [
    "=" * 60,
    "SikuRoBERTa 推断报告（1075 首项目诗词）",
    "=" * 60,
    f"模型路径：{MODEL_PATH}",
    f"推断诗词数：{total}",
    "",
    "5 类标签分布：",
]
for zh, n in sorted(lbl_cnt.items(), key=lambda x: -x[1]):
    lines.append(f"  {zh:<8}: {n:>4}首  {n/total*100:>5.1f}%")

lines += ["", "3 类极性分布："]
for pol, n in sorted(pol_cnt.items(), key=lambda x: -x[1]):
    lines.append(f"  {pol:<12}: {n:>4}首  {n/total*100:>5.1f}%")

avg_conf = result_df["siku_conf"].mean()
lines += [
    "",
    f"平均置信度：{avg_conf:.4f}",
    "",
    "说明：",
    "  - siku_label/siku_label_zh: 5 类预测标签（基于 FSPC 微调）",
    "  - siku_polarity: 3 类极性（0+1→negative, 2→neutral, 3+4→positive）",
    "  - siku_prob_0..4: 每类 softmax 概率，可用于软标签融合",
    "  - 与 Phase 1 的 15 维 C3 向量互补：",
    "    siku_polarity 修正 polarity 判断",
    "    c3_* 维度提供细粒度情感特征",
]

report = "\n".join(lines)
print("\n" + report)

REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_OUT, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\n报告已保存至：{REPORT_OUT}")
