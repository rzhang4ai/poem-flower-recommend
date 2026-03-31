"""
step2e_sentiment / 04_finetune_ccpoem.py
========================================
在 FSPC 5000 首标注诗词语料上对 BERT-CCPoem 做情感分类微调。

输入
----
  output/lexicon/FSPC_V1.0.json

输出
----
  models/ccpoem_sentiment_ft/                 : 微调后的模型 + tokenizer
  output/finetune_ccpoem/train_log.txt        : 训练日志
  output/finetune_ccpoem/test_metrics.txt     : 测试集指标
  output/finetune_ccpoem/predictions_test.csv : 测试集预测结果

运行方式（项目根目录）
----
  source flower_env/bin/activate
  cd 02.sample_label_phase2/step2e_sentiment
  python 04_finetune_ccpoem.py
"""

import json
import os
import random
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent.parent

FSPC_PATH = BASE_DIR / "output" / "lexicon" / "FSPC_V1.0.json"
OUTPUT_DIR = BASE_DIR / "output" / "finetune_ccpoem"
MODEL_SAVE = ROOT_DIR / "models" / "ccpoem_sentiment_ft"
CCPOEM_PATH = ROOT_DIR / "models" / "bert_ccpoem"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE.mkdir(parents=True, exist_ok=True)

# ── 随机种子 ──────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 标签定义 ──────────────────────────────────────────────────────────────
LABEL_LIST = ["negative", "implicit_negative", "neutral", "implicit_positive", "positive"]
LABEL_ZH = ["负面", "隐性负面", "中性", "隐性正面", "正面"]
NUM_LABELS = 5
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
FSPC2ID = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

# ── 超参数 ────────────────────────────────────────────────────────────────
MAX_LEN = 96
BATCH_SIZE = 16
NUM_EPOCHS = 5
LR = 2e-5
WARMUP_RATIO = 0.1
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
USE_WEIGHTED_SAMPLER = True

# 通过环境变量快速切换实验配置（避免频繁改代码）
#   CCPOEM_EXP=baseline      -> 仅加权CE（与上一版最接近）
#   CCPOEM_EXP=focal         -> Focal，不重采样
#   CCPOEM_EXP=sampler       -> 加权CE + 重采样
#   CCPOEM_EXP=focal_sampler -> Focal + 重采样（最激进）
EXP_MODE = os.getenv("CCPOEM_EXP", "baseline").strip().lower()
if EXP_MODE == "baseline":
    USE_FOCAL_LOSS = False
    USE_WEIGHTED_SAMPLER = False
elif EXP_MODE == "focal":
    USE_FOCAL_LOSS = True
    USE_WEIGHTED_SAMPLER = False
elif EXP_MODE == "sampler":
    USE_FOCAL_LOSS = False
    USE_WEIGHTED_SAMPLER = True
elif EXP_MODE == "focal_sampler":
    USE_FOCAL_LOSS = True
    USE_WEIGHTED_SAMPLER = True
else:
    print(f"[警告] 未识别 CCPOEM_EXP={EXP_MODE}，回退 baseline")
    EXP_MODE = "baseline"
    USE_FOCAL_LOSS = False
    USE_WEIGHTED_SAMPLER = False


class PoemSentimentDataset(Dataset):
    def __init__(self, records, tokenizer, max_len):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

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
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(rec["label"], dtype=torch.long),
        }


class WeightedTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights=None,
        use_focal_loss=False,
        focal_gamma=2.0,
        use_weighted_sampler=False,
        sampler_weights=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.use_weighted_sampler = use_weighted_sampler
        self.sampler_weights = sampler_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device)

        if self.use_focal_loss:
            ce_each = torch.nn.functional.cross_entropy(
                logits, labels, weight=weights, reduction="none"
            )
            pt = torch.exp(-ce_each).clamp(min=1e-8, max=1.0)
            focal_factor = (1.0 - pt) ** self.focal_gamma
            loss = (focal_factor * ce_each).mean()
        else:
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        """
        可选：按样本权重重采样，提升少数类出现频率。
        """
        if not self.use_weighted_sampler or self.sampler_weights is None:
            return super().get_train_dataloader()

        train_dataset = self.train_dataset
        sampler = WeightedRandomSampler(
            weights=self.sampler_weights,
            num_samples=len(self.sampler_weights),
            replacement=True,
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1}


def load_fspc(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"未找到 FSPC 文件: {path}")
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            poem_text = str(item.get("poem", "")).replace("|", " ").strip()
            holistic = str(item.get("setiments", {}).get("holistic", "3"))
            if not poem_text or holistic not in FSPC2ID:
                continue
            data.append({
                "text": poem_text,
                "label": FSPC2ID[holistic],
                "title": str(item.get("title", "")),
                "dynasty": str(item.get("dynasty", "")),
                "poet": str(item.get("poet", "")),
            })
    return data


def main():
    print("=" * 70)
    print("BERT-CCPoem 在 FSPC 上微调（5类情感）")
    print("=" * 70)
    print(f"FSPC_PATH: {FSPC_PATH}")
    print(f"CCPOEM_PATH: {CCPOEM_PATH}")
    print(f"EXP_MODE: {EXP_MODE}")

    if not CCPOEM_PATH.exists():
        raise FileNotFoundError(
            f"模型目录不存在: {CCPOEM_PATH}\n请先确认 BERT-CCPoem 已下载到 models/bert_ccpoem。"
        )

    print("\n[1/6] 读取 FSPC 数据...")
    data = load_fspc(FSPC_PATH)
    print(f"总样本数: {len(data)}")
    label_dist = Counter(d["label"] for d in data)
    print("标签分布:", {LABEL_LIST[k]: v for k, v in sorted(label_dist.items())})

    print("\n[2/6] 分层划分 train/val/test = 8/1/1 ...")
    labels_all = [d["label"] for d in data]
    train_data, tmp = train_test_split(
        data, test_size=0.2, stratify=labels_all, random_state=SEED
    )
    labels_tmp = [d["label"] for d in tmp]
    val_data, test_data = train_test_split(
        tmp, test_size=0.5, stratify=labels_tmp, random_state=SEED
    )
    print(f"train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    print("\n[3/6] 加载 tokenizer 和模型...")
    tokenizer = AutoTokenizer.from_pretrained(str(CCPOEM_PATH), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(CCPOEM_PATH),
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )

    print("\n[4/6] 构建数据集与训练参数...")
    train_ds = PoemSentimentDataset(train_data, tokenizer, MAX_LEN)
    val_ds = PoemSentimentDataset(val_data, tokenizer, MAX_LEN)
    test_ds = PoemSentimentDataset(test_data, tokenizer, MAX_LEN)

    train_label_dist = Counter(d["label"] for d in train_data)
    class_counts = np.array([train_label_dist[i] for i in range(NUM_LABELS)], dtype=float)
    class_weights = torch.tensor(
        len(train_data) / (NUM_LABELS * class_counts), dtype=torch.float32
    )
    print("类别权重:", dict(zip(LABEL_LIST, class_weights.numpy().round(3))))
    print("训练集标签分布:", {LABEL_LIST[k]: v for k, v in sorted(train_label_dist.items())})

    # 样本级权重（用于 WeightedRandomSampler）
    sample_weights = None
    if USE_WEIGHTED_SAMPLER:
        weights_np = class_weights.numpy()
        sample_weights = [float(weights_np[rec["label"]]) for rec in train_data]
        print("已启用 WeightedRandomSampler（按类别权重重采样）")
    else:
        print("未启用 WeightedRandomSampler")

    if torch.backends.mps.is_available():
        device_str = "mps"
        print("训练设备: MPS")
    elif torch.cuda.is_available():
        device_str = "cuda"
        print("训练设备: CUDA")
    else:
        device_str = "cpu"
        print("训练设备: CPU")

    training_args = TrainingArguments(
        output_dir=str(MODEL_SAVE / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=50,
        save_total_limit=2,
        max_grad_norm=1.0,
        seed=SEED,
        report_to="none",
        use_mps_device=(device_str == "mps"),
        fp16=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
        use_focal_loss=USE_FOCAL_LOSS,
        focal_gamma=FOCAL_GAMMA,
        use_weighted_sampler=USE_WEIGHTED_SAMPLER,
        sampler_weights=sample_weights,
    )

    print(
        f"损失函数: {'FocalLoss' if USE_FOCAL_LOSS else 'CrossEntropy'}"
        f"{f'(gamma={FOCAL_GAMMA})' if USE_FOCAL_LOSS else ''}"
    )
    print(f"采样策略: {'WeightedRandomSampler' if USE_WEIGHTED_SAMPLER else 'Standard shuffle'}")

    print("\n[5/6] 开始训练...")
    train_result = trainer.train()

    print("\n保存最优模型...")
    trainer.save_model(str(MODEL_SAVE))
    tokenizer.save_pretrained(str(MODEL_SAVE))

    print("\n[6/6] 测试集评估...")
    test_result = trainer.predict(test_ds)
    preds = np.argmax(test_result.predictions, axis=-1)
    labels_true = np.array([d["label"] for d in test_data])

    acc = accuracy_score(labels_true, preds)
    mac_f1 = f1_score(labels_true, preds, average="macro", zero_division=0)
    wgt_f1 = f1_score(labels_true, preds, average="weighted", zero_division=0)
    per_f1 = f1_score(labels_true, preds, average=None, zero_division=0)
    cm = confusion_matrix(labels_true, preds)

    lines = [
        "=" * 70,
        "BERT-CCPoem 在 FSPC 上微调 — 测试集评估报告",
        "=" * 70,
        f"训练样本: {len(train_data)}  验证: {len(val_data)}  测试: {len(test_data)}",
        f"Epochs: {NUM_EPOCHS}  Batch: {BATCH_SIZE}  LR: {LR}",
        f"硬件: {device_str}",
        "",
        f"测试集 Accuracy   : {acc:.4f}",
        f"测试集 Macro-F1   : {mac_f1:.4f}",
        f"测试集 Weighted-F1: {wgt_f1:.4f}",
        "",
        "各类别 F1:",
    ]
    for i, (lbl, zh, f) in enumerate(zip(LABEL_LIST, LABEL_ZH, per_f1)):
        n_test = sum(1 for d in test_data if d["label"] == i)
        lines.append(f"  {zh}({lbl}): F1={f:.4f}  (测试样本数={n_test})")

    lines += [
        "",
        "混淆矩阵（行=真实，列=预测）:",
        "  " + "  ".join(f"{l[:4]:>5}" for l in LABEL_ZH),
    ]
    for i, row in enumerate(cm):
        lines.append(f"  {LABEL_ZH[i]:<6} " + "  ".join(f"{v:>5}" for v in row))

    report = "\n".join(lines)
    print("\n" + report)

    with open(OUTPUT_DIR / "test_metrics.txt", "w", encoding="utf-8") as f:
        f.write(report)

    with open(OUTPUT_DIR / "train_log.txt", "w", encoding="utf-8") as f:
        f.write(f"train_runtime: {train_result.metrics.get('train_runtime', 0):.1f}s\n")
        for k, v in train_result.metrics.items():
            f.write(f"{k}: {v}\n")

    pred_df = []
    for rec, y_true, y_pred in zip(test_data, labels_true, preds):
        pred_df.append({
            "title": rec["title"],
            "dynasty": rec["dynasty"],
            "poet": rec["poet"],
            "text": rec["text"],
            "y_true": int(y_true),
            "y_pred": int(y_pred),
            "y_true_zh": LABEL_ZH[int(y_true)],
            "y_pred_zh": LABEL_ZH[int(y_pred)],
            "correct": int(y_true == y_pred),
        })
    import pandas as pd
    pd.DataFrame(pred_df).to_csv(
        OUTPUT_DIR / "predictions_test.csv", index=False, encoding="utf-8-sig"
    )

    print(f"\n报告已保存至: {OUTPUT_DIR}")
    print(f"模型已保存至: {MODEL_SAVE}")
    print("下一步：可仿照 05_inference_siku.py 新建 05_inference_ccpoem.py 做1075首推断")


if __name__ == "__main__":
    main()
