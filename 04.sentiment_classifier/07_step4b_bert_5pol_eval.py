"""
07_step4b_bert_5pol_eval.py
===========================
方法B：加载现有微调 BERT-CCPoem（models/ccpoem_sentiment_ft）在 fspc_test 上推理，
评估 5极性分类性能，并输出与方法A（SVM）的对比报告。

⚠ 近似比较说明：
  微调BERT 在全量 5000首 FSPC 上训练，fspc_test 对它并非真正盲测。
  SVM 的 train/test 严格隔离。
  因此微调BERT 的测试分数可能被轻微高估，结论仅供方向性参考。

输出：
  output/eval_results/method_b_bert_5pol.json  评估结果 JSON
  output/eval_results/comparison_5pol.txt       两种方法对比报告（若方法A结果已存在）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ===========================================================================
# 配置
# ===========================================================================
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

SPLIT_CSV     = _SCRIPT_DIR  / "output" / "splits"       / "fspc_split.csv"
BERT_FT_DIR   = _PROJECT_ROOT / "models" / "ccpoem_sentiment_ft"
OUT_EVAL      = _SCRIPT_DIR  / "output" / "eval_results"
OUT_EVAL.mkdir(parents=True, exist_ok=True)

RESULT_JSON_B = OUT_EVAL / "method_b_bert_5pol.json"
RESULT_JSON_A = OUT_EVAL / "method_a_svm_5pol.json"
COMPARE_TXT   = OUT_EVAL / "comparison_5pol.txt"

BATCH_SIZE    = 32
MAX_LENGTH    = 128

# 微调模型的标签（小写加下划线）-> 规范化映射
BERT_LABEL_NORM = {
    "negative":          "Negative",
    "implicit_negative": "Implicit Negative",
    "neutral":           "Neutral",
    "implicit_positive": "Implicit Positive",
    "positive":          "Positive",
}
POL_ORDER = [
    "Negative", "Implicit Negative", "Neutral",
    "Implicit Positive", "Positive",
]


# ===========================================================================
# 工具
# ===========================================================================
def resolve_model_dir(model_root: Path) -> Path:
    """优先用根目录 config.json，否则找最新 checkpoint。"""
    if (model_root / "config.json").exists():
        return model_root
    ckpt_dir = model_root / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(
            [d for d in ckpt_dir.iterdir() if d.is_dir()],
            key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0,
        )
        if ckpts:
            return ckpts[-1]
    raise FileNotFoundError(f"找不到有效的微调模型目录: {model_root}")


def batch_predict(
    texts: List[str],
    tok,
    model,
    device: str,
    id2label: dict,
) -> tuple[List[str], np.ndarray]:
    """批量推理，返回 (预测标签列表, 概率矩阵)。"""
    all_labels, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1).cpu().float().numpy()
            preds  = probs.argmax(axis=-1)
            for p in preds:
                raw = id2label.get(str(p), id2label.get(p, "neutral"))
                all_labels.append(BERT_LABEL_NORM.get(raw, raw))
            all_probs.append(probs)
            done = min(i + BATCH_SIZE, len(texts))
            if done % 500 == 0 or done == len(texts):
                print(f"  [{done}/{len(texts)}] 已推理...")
    return all_labels, np.vstack(all_probs)


# ===========================================================================
# 对比报告
# ===========================================================================
def print_comparison(result_a: dict, result_b: dict) -> str:
    lines = [
        "=" * 64,
        "  5极性分类 方法对比报告",
        "  ⚠ 微调BERT 在全量FSPC训练，fspc_test 非严格盲测",
        "=" * 64,
        f"  {'指标':<18s}  {'方法A: SVM':>12s}  {'方法B: 微调BERT':>14s}  {'差值(B-A)':>10s}",
        "-" * 64,
    ]
    for metric, label in [
        ("accuracy",    "Accuracy"),
        ("macro_f1",    "Macro-F1"),
        ("weighted_f1", "Weighted-F1"),
    ]:
        va = result_a[metric]
        vb = result_b[metric]
        diff = vb - va
        mark = "↑" if diff > 0.01 else ("↓" if diff < -0.01 else "≈")
        lines.append(f"  {label:<18s}  {va:>12.4f}  {vb:>14.4f}  {diff:>+8.4f} {mark}")

    lines += ["", f"  {'类别':<22s}  {'A F1':>6}  {'B F1':>6}  {'胜者':>6}"]
    lines.append("-" * 64)
    for pol in POL_ORDER:
        f1a = result_a["per_class"].get(pol, {}).get("f1", 0.0)
        f1b = result_b["per_class"].get(pol, {}).get("f1", 0.0)
        winner = "B" if f1b > f1a + 0.01 else ("A" if f1a > f1b + 0.01 else "≈")
        lines.append(f"  {pol:<22s}  {f1a:>6.4f}  {f1b:>6.4f}  {winner:>6}")

    lines += ["=" * 64]
    macro_a = result_a["macro_f1"]
    macro_b = result_b["macro_f1"]
    if macro_b > macro_a + 0.02:
        verdict = f"✓ 方法B（微调BERT）更优，建议用作5极性分类器（Macro-F1: {macro_b:.4f} vs {macro_a:.4f}）"
    elif macro_a > macro_b + 0.02:
        verdict = f"✓ 方法A（SVM）更优，建议用作5极性分类器（Macro-F1: {macro_a:.4f} vs {macro_b:.4f}）"
    else:
        verdict = (f"≈ 两种方法相近（Macro-F1 差 <0.02）\n"
                   f"  建议选方法A（SVM）：更轻量、推理快、无需GPU")
    lines += ["", f"  【结论】 {verdict}", "=" * 64]

    report = "\n".join(lines)
    print(report)
    return report


# ===========================================================================
# 主函数
# ===========================================================================
def main() -> None:
    # ── 检查依赖 ──────────────────────────────────────────────────
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"切分文件不存在: {SPLIT_CSV}\n请先运行 07_step3_split.py"
        )

    # ── 加载 fspc_test ────────────────────────────────────────────
    print(f"加载切分文件: {SPLIT_CSV}")
    df = pd.read_csv(SPLIT_CSV)
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"  fspc_test: {len(df_test)} 首")

    # ── 加载微调 BERT ─────────────────────────────────────────────
    model_dir = resolve_model_dir(BERT_FT_DIR)
    print(f"\n加载微调模型: {model_dir}")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"  device: {device}")

    tok   = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), trust_remote_code=True
    )
    model.to(device)

    # 读取 id2label
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    id2label = cfg.get("id2label", {})
    print(f"  id2label: {id2label}")

    # ── 推理 ──────────────────────────────────────────────────────
    print(f"\n推理 {len(df_test)} 首诗...")
    texts  = df_test["text"].tolist()
    y_pred_labels, y_prob = batch_predict(texts, tok, model, device, id2label)
    y_true = df_test["polarity"].tolist()

    # 规范化真实标签（FSPC 用首字母大写格式）
    y_true_norm = [BERT_LABEL_NORM.get(t.lower().replace(" ", "_"), t) for t in y_true]

    # ── 评估 ──────────────────────────────────────────────────────
    acc = accuracy_score(y_true_norm, y_pred_labels)
    rpt_dict = classification_report(
        y_true_norm, y_pred_labels,
        labels=POL_ORDER,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    rpt_str = classification_report(
        y_true_norm, y_pred_labels,
        labels=POL_ORDER,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_norm, y_pred_labels, labels=POL_ORDER)

    print(f"\n{'='*60}")
    print(f"  方法B：微调 BERT-CCPoem  (⚠ 非严格盲测)")
    print(f"  测试集: fspc_test ({len(y_true)} 首)")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.4f}")
    print(rpt_str)
    print("  混淆矩阵（行=真实，列=预测）:")
    header = "  " + "  ".join(f"{c[:8]:>8}" for c in POL_ORDER)
    print(header)
    for i, row_cls in enumerate(POL_ORDER):
        row_str = "  ".join(f"{v:8d}" for v in cm[i])
        print(f"  {row_cls[:8]:>8}: {row_str}")

    # ── 保存结果 ──────────────────────────────────────────────────
    per_class = {}
    for pol in POL_ORDER:
        per_class[pol] = {
            "precision": rpt_dict.get(pol, {}).get("precision", 0.0),
            "recall":    rpt_dict.get(pol, {}).get("recall",    0.0),
            "f1":        rpt_dict.get(pol, {}).get("f1-score",  0.0),
            "support":   rpt_dict.get(pol, {}).get("support",   0),
        }
    result_b = {
        "method":       "Fine-tuned BERT-CCPoem (approx, not blind test)",
        "test_size":    len(y_true),
        "accuracy":     float(acc),
        "macro_f1":     float(rpt_dict["macro avg"]["f1-score"]),
        "weighted_f1":  float(rpt_dict["weighted avg"]["f1-score"]),
        "per_class":    per_class,
        "confusion_matrix": cm.tolist(),
        "label_order":  POL_ORDER,
    }
    with open(RESULT_JSON_B, "w", encoding="utf-8") as f:
        json.dump(result_b, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {RESULT_JSON_B}")

    # ── 生成对比报告（若方法A结果已存在）────────────────────────
    if RESULT_JSON_A.exists():
        with open(RESULT_JSON_A) as f:
            result_a = json.load(f)
        print(f"\n检测到方法A结果，生成对比报告...")
        report_str = print_comparison(result_a, result_b)
        with open(COMPARE_TXT, "w", encoding="utf-8") as f:
            f.write(report_str + "\n")
        print(f"对比报告已保存: {COMPARE_TXT}")
    else:
        print(f"\n（方法A结果未找到，请先运行 07_step4a_svm_5pol.py 再运行本脚本以生成对比）")


if __name__ == "__main__":
    main()
