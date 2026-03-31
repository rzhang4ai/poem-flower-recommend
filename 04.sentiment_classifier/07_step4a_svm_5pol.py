"""
07_step4a_svm_5pol.py
=====================
方法A：BERT-CCPoem（冻结）+ Linear SVC 做 FSPC 5极性分类。

流程：
  1. 加载 fspc_split.csv（由 07_step3_split.py 生成）
  2. 提取全量 FSPC 5000首的 [CLS] 向量（首次运行耗时较长，自动缓存）
  3. 用 fspc_train 训练 SVC(linear, balanced)
  4. 在 fspc_test 上评估，保存结果供对比

输出：
  output/svm_models/fspc_cls_features.npy    FSPC全量CLS特征（缓存）
  output/svm_models/fspc_cls_index.csv       特征行↔诗歌对应表
  output/svm_models/svm_5pol.pkl             训练好的 SVM 5极性模型
  output/eval_results/method_a_svm_5pol.json 评估结果 JSON（供对比脚本读取）
"""

from __future__ import annotations

import json
import pickle
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
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer

# ===========================================================================
# 配置
# ===========================================================================
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

SPLIT_CSV    = _SCRIPT_DIR / "output" / "splits"       / "fspc_split.csv"
MODEL_DIR    = _PROJECT_ROOT / "models" / "bert_ccpoem"
OUT_SVM      = _SCRIPT_DIR / "output" / "svm_models"
OUT_EVAL     = _SCRIPT_DIR / "output" / "eval_results"
OUT_SVM.mkdir(parents=True, exist_ok=True)
OUT_EVAL.mkdir(parents=True, exist_ok=True)

FEAT_NPY     = OUT_SVM / "fspc_cls_features.npy"
FEAT_IDX     = OUT_SVM / "fspc_cls_index.csv"
SVM_PKL      = OUT_SVM / "svm_5pol.pkl"
RESULT_JSON  = OUT_EVAL / "method_a_svm_5pol.json"

BATCH_SIZE   = 32
MAX_LENGTH   = 128

POL_ORDER = [
    "Negative", "Implicit Negative", "Neutral",
    "Implicit Positive", "Positive",
]


# ===========================================================================
# BERT 特征提取
# ===========================================================================
def load_bert(model_path: str):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[BERT] 加载模型: {model_path}  (device={device})")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    mdl.to(device).eval()
    return tok, mdl, device


def extract_cls(texts: List[str], tok, mdl, device: str) -> np.ndarray:
    vecs = []
    n = len(texts)
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LENGTH)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            vecs.append(cls)
            done = min(i + BATCH_SIZE, n)
            if done % 500 == 0 or done == n:
                print(f"  [{done}/{n}] 已提取...")
    return np.vstack(vecs)


# ===========================================================================
# 主函数
# ===========================================================================
def main() -> None:
    # ── 检查依赖文件 ──────────────────────────────────────────────
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"切分文件不存在: {SPLIT_CSV}\n"
            "请先运行 07_step3_split.py 生成数据集切分。"
        )

    # ── 加载切分 ──────────────────────────────────────────────────
    print(f"加载切分文件: {SPLIT_CSV}")
    df = pd.read_csv(SPLIT_CSV)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_test  = df[df["split"] == "test"].reset_index(drop=True)
    print(f"  train: {len(df_train)}  test: {len(df_test)}")

    # ── 提取 / 读取 CLS 特征 ──────────────────────────────────────
    if FEAT_NPY.exists() and FEAT_IDX.exists():
        print(f"读取缓存特征: {FEAT_NPY}")
        X_all   = np.load(FEAT_NPY)
        idx_df  = pd.read_csv(FEAT_IDX)
        # 按 fspc_split 的行顺序对齐（fspc_split 包含全量5000首）
        # 检查行数一致
        if len(X_all) != len(df):
            raise RuntimeError(
                f"缓存特征行数 ({len(X_all)}) 与切分文件行数 ({len(df)}) 不符，"
                "请删除缓存后重新运行。"
            )
        print(f"  特征 shape: {X_all.shape}")
    else:
        tok, mdl, device = load_bert(str(MODEL_DIR))
        texts = df["text"].tolist()
        print(f"提取 {len(texts)} 首诗的 [CLS] 向量（首次运行，需要较长时间）...")
        X_all = extract_cls(texts, tok, mdl, device)
        np.save(FEAT_NPY, X_all)
        df[["poet", "title", "dynasty", "text", "polarity", "split"]].to_csv(
            FEAT_IDX, index=False, encoding="utf-8-sig"
        )
        print(f"  特征已缓存 shape={X_all.shape}")

    # ── 切分特征矩阵 ──────────────────────────────────────────────
    train_mask = df["split"] == "train"
    test_mask  = df["split"] == "test"
    X_train = X_all[train_mask.values]
    X_test  = X_all[test_mask.values]
    y_train = df_train["polarity"].tolist()
    y_test  = df_test["polarity"].tolist()

    # ── 训练 SVM ─────────────────────────────────────────────────
    print(f"\n训练 SVM（linear kernel, class_weight=balanced）...")
    le = LabelEncoder()
    le.fit(POL_ORDER)
    y_tr = le.transform(y_train)
    y_te = le.transform(y_test)

    clf = SVC(kernel="linear", probability=True,
              class_weight="balanced", random_state=42)
    clf.fit(X_train, y_tr)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # ── 评估 ──────────────────────────────────────────────────────
    acc = accuracy_score(y_te, y_pred)
    report_str = classification_report(
        y_te, y_pred,
        target_names=le.classes_,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_te, y_pred, labels=list(range(len(le.classes_))))

    print(f"\n{'='*60}")
    print(f"  方法A：SVM (BERT-CCPoem CLS, no finetune)")
    print(f"  测试集: fspc_test ({len(y_test)} 首)")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.4f}")
    print(report_str)
    print("  混淆矩阵（行=真实，列=预测）:")
    header = "  " + "  ".join(f"{c[:8]:>8}" for c in le.classes_)
    print(header)
    for i, row_cls in enumerate(le.classes_):
        row_str = "  ".join(f"{v:8d}" for v in cm[i])
        print(f"  {row_cls[:8]:>8}: {row_str}")

    # ── 保存结果 ──────────────────────────────────────────────────
    per_class = {}
    rpt_dict = classification_report(
        y_te, y_pred,
        target_names=le.classes_,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    for cls in le.classes_:
        per_class[cls] = {
            "precision": rpt_dict[cls]["precision"],
            "recall":    rpt_dict[cls]["recall"],
            "f1":        rpt_dict[cls]["f1-score"],
            "support":   rpt_dict[cls]["support"],
        }

    result = {
        "method":       "SVM (BERT-CCPoem CLS, no finetune)",
        "test_size":    len(y_test),
        "accuracy":     float(acc),
        "macro_f1":     float(rpt_dict["macro avg"]["f1-score"]),
        "weighted_f1":  float(rpt_dict["weighted avg"]["f1-score"]),
        "per_class":    per_class,
        "confusion_matrix": cm.tolist(),
        "label_order":  le.classes_.tolist(),
    }
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ── 保存 SVM 模型 ─────────────────────────────────────────────
    with open(SVM_PKL, "wb") as f:
        pickle.dump({"clf": clf, "le": le}, f)

    print(f"\n结果已保存: {RESULT_JSON}")
    print(f"模型已保存: {SVM_PKL}")
    print(f"{'='*60}")
    print(f"  Macro-F1: {result['macro_f1']:.4f}   Accuracy: {acc:.4f}")
    print(f"  （运行 07_step4b_bert_5pol_eval.py 后可查看两种方法对比）")


if __name__ == "__main__":
    main()
