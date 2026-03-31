"""
step2e_sentiment / 04_finetune_siku.py
=======================================
在 FSPC 5000 首标注诗词语料上对 SikuRoBERTa 做情感分类微调。

输入
----
  FSPC_V1.0.json          : 5000 首诗，holistic 标签 1-5（JSONL 格式）
  models/siku_sentiment_ft : 微调后的模型保存目录

标签映射（5 类，0-based）
---
  FSPC 原始  → 模型 id  → 中文含义
  1          → 0        → 负面 (explicit negative)
  2          → 1        → 隐性负面 (implicit negative)
  3          → 2        → 中性 (neutral)
  4          → 3        → 隐性正面 (implicit positive)
  5          → 4        → 正面 (explicit positive)

输出
----
  models/siku_sentiment_ft/          : 微调后的模型 + tokenizer
  output/finetune/train_log.txt      : 训练过程指标
  output/finetune/test_metrics.txt   : 测试集评估指标（accuracy / macro-F1）
  output/finetune/confusion_matrix.txt

运行方式（在项目根目录）
----
  source flower_env/bin/activate
  cd 02.sample_label_phase2/step2e_sentiment
  python 04_finetune_siku.py

硬件说明
----
  - Mac Mini (Apple Silicon)：自动检测并启用 MPS 加速
  - 预计训练时间（3 epoch，4000 条）：MPS 约 10-20 分钟
"""

import json, os, random, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (
    BertTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
ROOT_DIR    = BASE_DIR.parent.parent
FSPC_PATH   = BASE_DIR / "output" / "lexicon" / "FSPC_V1.0.json"
OUTPUT_DIR  = BASE_DIR / "output" / "finetune"
MODEL_SAVE  = ROOT_DIR / "models" / "siku_sentiment_ft"

SIKU_LOCAL  = str(
    ROOT_DIR / "models" / ".hf_cache" / "hub"
    / "models--SIKU-BERT--sikuroberta"
    / "snapshots" / "bb25260d5c321924fe4fb353c09191c0aaf5c5c6"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE.mkdir(parents=True, exist_ok=True)

# ── 随机种子 ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 标签定义 ──────────────────────────────────────────────────
LABEL_LIST = ["negative", "implicit_negative", "neutral", "implicit_positive", "positive"]
LABEL_ZH   = ["负面", "隐性负面", "中性", "隐性正面", "正面"]
NUM_LABELS = 5
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
# FSPC 原始标签 "1"-"5" → 0-4
FSPC2ID    = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

# ── 超参数 ────────────────────────────────────────────────────
MAX_LEN    = 64     # FSPC 诗词最长 ~28 字，64 足够
BATCH_SIZE = 16
NUM_EPOCHS = 5
LR         = 2e-5
WARMUP_RATIO = 0.1

# ── 读取 FSPC 数据 ────────────────────────────────────────────
print("读取 FSPC 数据...")
data = []
with open(FSPC_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            item = json.loads(line)
            poem_text = item["poem"].replace("|", " ")  # 竖线分隔 → 空格
            label_id  = FSPC2ID[item["setiments"]["holistic"]]
            data.append({"text": poem_text, "label": label_id})

print(f"总样本数：{len(data)}")
label_dist = Counter(d["label"] for d in data)
print("标签分布：", {LABEL_LIST[k]: v for k, v in sorted(label_dist.items())})

# ── 分层划分 8:1:1 ────────────────────────────────────────────
labels_all = [d["label"] for d in data]
train_data, tmp = train_test_split(data, test_size=0.2, stratify=labels_all, random_state=SEED)
labels_tmp = [d["label"] for d in tmp]
val_data, test_data = train_test_split(tmp, test_size=0.5, stratify=labels_tmp, random_state=SEED)

print(f"划分：train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

# ── Dataset ───────────────────────────────────────────────────
class PoemSentimentDataset(Dataset):
    def __init__(self, records, tokenizer, max_len):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(rec["label"], dtype=torch.long),
        }

# ── 加载 Tokenizer & 模型 ─────────────────────────────────────
print(f"\n加载 SikuRoBERTa tokenizer，路径：{SIKU_LOCAL}")
tokenizer = BertTokenizer.from_pretrained(SIKU_LOCAL)

print("加载 SikuRoBERTa 模型（SequenceClassification head）...")
model = AutoModelForSequenceClassification.from_pretrained(
    SIKU_LOCAL,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

# ── 类别权重（解决不均衡）────────────────────────────────────
# 权重 = 总样本 / (类别数 × 该类样本数)
class_counts = np.array([label_dist[i] for i in range(NUM_LABELS)], dtype=float)
class_weights = torch.tensor(len(train_data) / (NUM_LABELS * class_counts), dtype=torch.float)
print(f"类别权重：{dict(zip(LABEL_LIST, class_weights.numpy().round(3)))}")

# ── 自定义 Trainer（带加权交叉熵）────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── 评估指标 ──────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1}

# ── 构建 Dataset ──────────────────────────────────────────────
train_ds = PoemSentimentDataset(train_data, tokenizer, MAX_LEN)
val_ds   = PoemSentimentDataset(val_data,   tokenizer, MAX_LEN)
test_ds  = PoemSentimentDataset(test_data,  tokenizer, MAX_LEN)

# ── 检测硬件 ──────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device_str = "mps"
    print("MPS (Apple Silicon GPU) 已启用")
elif torch.cuda.is_available():
    device_str = "cuda"
    print("CUDA GPU 已启用")
else:
    device_str = "cpu"
    print("使用 CPU 训练（较慢）")

# ── 训练参数 ──────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir              = str(MODEL_SAVE / "checkpoints"),
    num_train_epochs        = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate           = LR,
    warmup_ratio            = WARMUP_RATIO,
    weight_decay            = 0.01,
    eval_strategy           = "epoch",
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "macro_f1",
    greater_is_better       = True,
    logging_dir             = str(OUTPUT_DIR / "logs"),
    logging_steps           = 50,
    save_total_limit        = 2,
    seed                    = SEED,
    report_to               = "none",
    use_mps_device          = (device_str == "mps"),
    fp16                    = False,  # MPS 不支持 fp16
)

# ── 训练 ──────────────────────────────────────────────────────
trainer = WeightedTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    class_weights   = class_weights,
)

print("\n开始训练...")
train_result = trainer.train()

# ── 保存最优模型 ──────────────────────────────────────────────
print(f"\n保存最优模型至：{MODEL_SAVE}")
trainer.save_model(str(MODEL_SAVE))
tokenizer.save_pretrained(str(MODEL_SAVE))

# ── 测试集评估 ────────────────────────────────────────────────
print("\n在测试集上评估...")
test_result = trainer.predict(test_ds)
preds  = np.argmax(test_result.predictions, axis=-1)
labels_true = np.array([d["label"] for d in test_data])

acc     = accuracy_score(labels_true, preds)
mac_f1  = f1_score(labels_true, preds, average="macro", zero_division=0)
wgt_f1  = f1_score(labels_true, preds, average="weighted", zero_division=0)
per_f1  = f1_score(labels_true, preds, average=None, zero_division=0)
cm      = confusion_matrix(labels_true, preds)

print(f"测试集 Accuracy : {acc:.4f}")
print(f"测试集 Macro-F1 : {mac_f1:.4f}")
print(f"测试集 Weighted-F1: {wgt_f1:.4f}")

# ── 写报告 ────────────────────────────────────────────────────
lines = [
    "=" * 60,
    "SikuRoBERTa 在 FSPC 上微调 — 测试集评估报告",
    "=" * 60,
    f"训练样本：{len(train_data)}  验证：{len(val_data)}  测试：{len(test_data)}",
    f"Epochs：{NUM_EPOCHS}  Batch：{BATCH_SIZE}  LR：{LR}",
    f"硬件：{device_str}",
    "",
    f"测试集 Accuracy   : {acc:.4f}",
    f"测试集 Macro-F1   : {mac_f1:.4f}",
    f"测试集 Weighted-F1: {wgt_f1:.4f}",
    "",
    "各类别 F1：",
]
for i, (lbl, zh, f) in enumerate(zip(LABEL_LIST, LABEL_ZH, per_f1)):
    n_test = sum(1 for d in test_data if d["label"] == i)
    lines.append(f"  {zh}({lbl}): F1={f:.4f}  (测试集样本数={n_test})")

lines += [
    "",
    "混淆矩阵（行=真实，列=预测）：",
    "  " + "  ".join(f"{l[:4]:>5}" for l in LABEL_ZH),
]
for i, row in enumerate(cm):
    lines.append(f"  {LABEL_ZH[i]:<6} " + "  ".join(f"{v:>5}" for v in row))

report = "\n".join(lines)
print("\n" + report)

with open(OUTPUT_DIR / "test_metrics.txt", "w", encoding="utf-8") as f:
    f.write(report)

# 训练过程日志
with open(OUTPUT_DIR / "train_log.txt", "w", encoding="utf-8") as f:
    f.write(f"train_runtime: {train_result.metrics.get('train_runtime', 0):.1f}s\n")
    for k, v in train_result.metrics.items():
        f.write(f"{k}: {v}\n")

print(f"\n报告已保存至：{OUTPUT_DIR}")
print("模型已保存至：", MODEL_SAVE)
print("\n下一步：运行 05_inference_siku.py 对 1075 首项目诗词做推断")
