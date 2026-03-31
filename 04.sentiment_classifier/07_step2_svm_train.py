"""
07_step2_svm_train.py
=====================
第二步：BERT-CCPoem [CLS] 特征提取 + 三层 SVM 训练与评估

目标：用黄金训练集训练古诗词情感分类器，分别评估 C3（细粒度）和 C2（粗类）
的有效性，并通过"层级一致性验证"判断细粒度分类是否可信。

流程：
  1. 加载黄金集（golden_dataset_step1.csv）
  2. 用 BERT-CCPoem 批量提取 [CLS] 向量（无微调）
  3. 分三个粒度独立训练并用分层 5-fold CV 评估：
       SVM_C1: 正/负 二分类（2类，全量数据）
       SVM_C2: 粗类 4分类（排除 surprise 类，样本<30）
       SVM_C3: 细粒度 10分类（排除样本<MIN_C3_SAMPLES 的类别）
  4. 层级一致性验证：
       将 SVM_C3 的预测折叠到 C2 层级，与 SVM_C2 直接预测对比
       → 如果折叠后准确率接近，说明 C3 模型"内部一致"

输出：
  output/svm_models/    模型 pickle 文件 + 特征矩阵
  output/svm_models/svm_eval_report.txt  各层级评估报告
"""

from __future__ import annotations

import json
import pickle
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer

# ===========================================================================
# 路径配置
# ===========================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

GOLDEN_CSV   = _SCRIPT_DIR / "output" / "pseudo_label" / "golden_dataset_step1.csv"
FCCPSL_CSV   = _SCRIPT_DIR / "output" / "lexicon"      / "fccpsl_terms_only.csv"
MODEL_DIR    = _PROJECT_ROOT / "models" / "bert_ccpoem"
OUT_DIR      = _SCRIPT_DIR / "output" / "svm_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# 可调参数
# ===========================================================================
MIN_C3_SAMPLES  = 30    # C3 类别最少样本数，低于此阈值的类别在 C3 实验中排除
CV_FOLDS        = 5     # 分层 k-fold 折数
BERT_BATCH_SIZE = 32    # BERT 批推理大小
MAX_LENGTH      = 128   # BERT 最大序列长度

# C2 中/英文说明
C2_ZH = {
    "pleasure": "愉悦",
    "favour":   "好感赞赏",
    "surprise": "惊奇",
    "sadness":  "悲伤",
    "disgust":  "厌恶苦闷",
}
C1_ZH = {"positive": "积极", "negative": "消极"}

# ===========================================================================
# 工具：加载标签映射
# ===========================================================================
def build_label_maps(fccpsl_path: Path) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    返回：
      c3_to_c2  : {c3_en -> c2_en}
      c3_to_c1  : {c3_en -> c1_en}
      c3_to_zh  : {c3_en -> c3_zh}
      c2_to_c1  : {c2_en -> c1_en}
    """
    df = pd.read_csv(fccpsl_path)
    c3_to_c2 = dict(zip(df["C3"], df["C2"]))
    c3_to_c1 = dict(zip(df["C3"], df["C1"]))
    c3_to_zh = dict(zip(df["C3"], df["C3_zh"]))
    c2_to_c1: Dict[str, str] = {}
    for c2, grp in df.groupby("C2"):
        c2_to_c1[c2] = grp["C1"].iloc[0]
    return c3_to_c2, c3_to_c1, c3_to_zh, c2_to_c1


# ===========================================================================
# 工具：加载黄金集，丰富层级标签
# ===========================================================================
def load_golden(golden_path: Path, fccpsl_path: Path) -> pd.DataFrame:
    df = pd.read_csv(golden_path)
    c3_to_c2, c3_to_c1, c3_to_zh, _ = build_label_maps(fccpsl_path)
    df["c2"] = df["pseudo_label"].map(c3_to_c2)
    df["c1"] = df["pseudo_label"].map(c3_to_c1)
    return df


# ===========================================================================
# BERT-CCPoem：批量提取 [CLS] 向量
# ===========================================================================
def load_bert(model_path: str, device: Optional[str] = None):
    if device is None:
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


def extract_cls(
    texts: List[str],
    tok,
    mdl,
    device: str,
    batch_size: int = BERT_BATCH_SIZE,
    max_len: int = MAX_LENGTH,
) -> np.ndarray:
    """批量提取 [CLS] 向量，输出 shape=(N, hidden_size)。"""
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            vecs.append(cls)
            if (i // batch_size + 1) % 5 == 0:
                print(f"  [{i + len(batch)}/{len(texts)}] 已提取...")
    return np.vstack(vecs) if vecs else np.zeros((0, 768), dtype=np.float32)


# ===========================================================================
# SVM 训练 + 分层 k-fold CV 评估
# ===========================================================================
def train_and_evaluate(
    X: np.ndarray,
    y_str: List[str],
    label_zh: Optional[Dict[str, str]] = None,
    title: str = "SVM 评估",
    n_splits: int = CV_FOLDS,
) -> Tuple[SVC, LabelEncoder, np.ndarray, str]:
    """
    训练 SVC(kernel='linear', class_weight='balanced')，
    用分层 k-fold CV 得到 out-of-fold 预测，打印分类报告。

    返回：
      (fitted_svm_on_full_data, label_encoder, oof_predictions, report_str)
    """
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes = le.classes_

    # 中文类名（用于报告）
    if label_zh:
        target_names = [f"{c}({label_zh.get(c, '')})" for c in classes]
    else:
        target_names = list(classes)

    clf = SVC(kernel="linear", probability=True,
              class_weight="balanced", random_state=42)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # out-of-fold 预测
    y_oof = cross_val_predict(clf, X, y, cv=skf)

    report_lines = [
        f"\n{'='*62}",
        f"  {title}  ({n_splits}-fold 分层 CV，class_weight=balanced)",
        f"{'='*62}",
        f"样本数: {len(y)}  类别数: {len(classes)}",
        "",
        classification_report(
            y, y_oof,
            target_names=target_names,
            digits=4,
            zero_division=0,
        ),
    ]
    report_str = "\n".join(report_lines)
    print(report_str)

    # 在全量数据上 fit 一个最终模型（用于后续推理）
    clf_final = SVC(kernel="linear", probability=True,
                    class_weight="balanced", random_state=42)
    clf_final.fit(X, y)

    return clf_final, le, y_oof, report_str


# ===========================================================================
# 层级一致性验证
# ===========================================================================
def coherence_check(
    y_c3_oof: np.ndarray,
    le_c3: LabelEncoder,
    y_c2_oof: np.ndarray,
    le_c2: LabelEncoder,
    df_eval: pd.DataFrame,
    c3_to_c2: Dict[str, str],
) -> str:
    """
    将 C3 的 OOF 预测折叠到 C2 层，与 C2 OOF 直接预测对比：
      折叠准确率越接近 C2 直接训练的准确率 -> C3 模型内部一致
      折叠准确率远低于 C2 -> C3 分类边界混乱，细粒度不可信
    """
    # C3 预测折叠到 C2
    c3_pred_labels = le_c3.inverse_transform(y_c3_oof)
    c2_from_c3     = np.array([c3_to_c2.get(c, "unknown") for c in c3_pred_labels])

    # C2 直接预测
    c2_direct = le_c2.inverse_transform(y_c2_oof)

    # 真实 C2（在 C3 有效子集上）
    c2_true = df_eval["c2"].values

    acc_c2_direct = accuracy_score(c2_true, c2_direct)
    acc_c2_fold   = accuracy_score(c2_true, c2_from_c3)
    gap = acc_c2_direct - acc_c2_fold

    if gap <= 0.03:
        verdict = "✓ 一致性好：C3 细粒度分类可信"
    elif gap <= 0.08:
        verdict = "⚠ 一致性一般：C3 有部分 C2 内部混淆，建议以 C2 结果为主"
    else:
        verdict = "✗ 一致性差：C3 分类边界紊乱，不建议直接使用 C3 标签"

    lines = [
        f"\n{'='*62}",
        f"  层级一致性验证（C3 折叠 -> C2 vs C2 直接训练）",
        f"{'='*62}",
        f"  C2 直接训练准确率:        {acc_c2_direct:.4f}",
        f"  C3→C2 折叠准确率:         {acc_c2_fold:.4f}",
        f"  Gap (直接 - 折叠):         {gap:+.4f}",
        f"  结论: {verdict}",
        "",
        "  折叠后 C2 混淆矩阵（行=真实 C2, 列=从C3折叠预测）:",
    ]
    c2_classes_sorted = sorted(set(c2_true))
    cm = confusion_matrix(c2_true, c2_from_c3, labels=c2_classes_sorted)
    header = "  " + "  ".join(f"{c[:8]:>8}" for c in c2_classes_sorted)
    lines.append(header)
    for i, row_cls in enumerate(c2_classes_sorted):
        row_str = "  ".join(f"{v:8d}" for v in cm[i])
        lines.append(f"  {row_cls[:8]:>8}: {row_str}")
    lines.append("=" * 62)

    result = "\n".join(lines)
    print(result)
    return result


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    all_reports: List[str] = []

    # ── 数据加载 ────────────────────────────────────────────────────────
    print(f"加载黄金集: {GOLDEN_CSV}")
    df = load_golden(GOLDEN_CSV, FCCPSL_CSV)
    c3_to_c2, c3_to_c1, c3_to_zh, c2_to_c1 = build_label_maps(FCCPSL_CSV)
    print(f"  -> 共 {len(df)} 首诗")

    # ── BERT 特征提取（或读取缓存）──────────────────────────────────────
    feat_cache = OUT_DIR / "cls_features.npy"
    idx_cache  = OUT_DIR / "cls_index.csv"

    if feat_cache.exists() and idx_cache.exists():
        print(f"读取缓存特征: {feat_cache}")
        X_all = np.load(feat_cache)
        idx_df = pd.read_csv(idx_cache)
        # 按缓存顺序对齐
        df = df.reset_index(drop=True)
        print(f"  -> 特征矩阵 shape: {X_all.shape}")
    else:
        tok, mdl, device = load_bert(str(MODEL_DIR))
        texts = df["text"].tolist()
        print(f"提取 {len(texts)} 首诗的 [CLS] 向量...")
        X_all = extract_cls(texts, tok, mdl, device)
        np.save(feat_cache, X_all)
        df.reset_index(drop=True).to_csv(idx_cache, index=False, encoding="utf-8-sig")
        print(f"  -> 特征已缓存: {feat_cache}  shape={X_all.shape}")

    # ================================================================
    # 实验一：SVM_C1（正/负 二分类，全量）
    # ================================================================
    print("\n" + "=" * 62)
    print("实验一：SVM_C1（C1 正/负 二分类，全量 1896 首）")
    clf_c1, le_c1, oof_c1, rpt_c1 = train_and_evaluate(
        X_all,
        df["c1"].tolist(),
        label_zh=C1_ZH,
        title="SVM_C1  正/负 二分类",
    )
    all_reports.append(rpt_c1)

    # ================================================================
    # 实验二：SVM_C2（粗类 4分类，排除 surprise）
    # ================================================================
    print("\n" + "=" * 62)
    print("实验二：SVM_C2（C2 粗类，排除 surprise<30 条）")

    mask_c2 = df["c2"] != "surprise"
    df_c2    = df[mask_c2].reset_index(drop=True)
    X_c2     = X_all[mask_c2.values]

    print(f"  C2 数据集: {len(df_c2)} 首  分布: {df_c2['c2'].value_counts().to_dict()}")
    clf_c2, le_c2, oof_c2, rpt_c2 = train_and_evaluate(
        X_c2,
        df_c2["c2"].tolist(),
        label_zh=C2_ZH,
        title="SVM_C2  粗类 4分类",
    )
    all_reports.append(rpt_c2)

    # ================================================================
    # 实验三：SVM_C3（细粒度 10分类，排除<MIN_C3_SAMPLES 的类别）
    # ================================================================
    print("\n" + "=" * 62)
    print(f"实验三：SVM_C3（C3 细粒度，排除 <{MIN_C3_SAMPLES} 条的类别）")

    c3_counts = df["pseudo_label"].value_counts()
    valid_c3  = c3_counts[c3_counts >= MIN_C3_SAMPLES].index.tolist()
    excluded  = c3_counts[c3_counts < MIN_C3_SAMPLES]

    print(f"  保留 C3 类别 ({len(valid_c3)} 个): {valid_c3}")
    print(f"  排除类别: {excluded.to_dict()}")

    mask_c3 = df["pseudo_label"].isin(valid_c3)
    df_c3   = df[mask_c3].reset_index(drop=True)
    X_c3    = X_all[mask_c3.values]

    print(f"  C3 数据集: {len(df_c3)} 首")
    clf_c3, le_c3, oof_c3, rpt_c3 = train_and_evaluate(
        X_c3,
        df_c3["pseudo_label"].tolist(),
        label_zh=c3_to_zh,
        title=f"SVM_C3  细粒度 {len(valid_c3)}分类",
    )
    all_reports.append(rpt_c3)

    # ================================================================
    # 层级一致性验证（C3 → C2 折叠 vs C2 直接训练）
    # ================================================================
    print("\n层级一致性验证（需要 C3/C2 数据集对齐）...")

    # 取 C3 和 C2 数据集的交集（去掉 surprise 且 c3 样本充足）
    mask_both  = mask_c3 & mask_c2
    df_both    = df[mask_both].reset_index(drop=True)
    X_both     = X_all[mask_both.values]

    # C3 在 both 子集上重新做 CV
    _, le_c3b, oof_c3b, _ = train_and_evaluate(
        X_both,
        df_both["pseudo_label"].tolist(),
        title="SVM_C3（一致性子集，仅用于折叠验证）",
        n_splits=CV_FOLDS,
    )
    _, le_c2b, oof_c2b, _ = train_and_evaluate(
        X_both,
        df_both["c2"].tolist(),
        title="SVM_C2（一致性子集，仅用于折叠验证）",
        n_splits=CV_FOLDS,
    )

    rpt_coh = coherence_check(oof_c3b, le_c3b, oof_c2b, le_c2b, df_both, c3_to_c2)
    all_reports.append(rpt_coh)

    # ================================================================
    # 保存模型与特征
    # ================================================================
    def save_model(obj, name):
        p = OUT_DIR / name
        with open(p, "wb") as f:
            pickle.dump(obj, f)
        print(f"  已保存: {p}")

    print("\n保存模型...")
    save_model({"clf": clf_c1, "le": le_c1}, "svm_c1.pkl")
    save_model({"clf": clf_c2, "le": le_c2}, "svm_c2.pkl")
    save_model({"clf": clf_c3, "le": le_c3}, "svm_c3.pkl")

    # ================================================================
    # 保存完整评估报告
    # ================================================================
    report_all = "\n".join(all_reports)
    out_report = OUT_DIR / "svm_eval_report.txt"
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report_all)
    print(f"\n完整评估报告已保存: {out_report}")

    # ── 打印总结 ─────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  训练完成总结")
    print("=" * 62)
    print(f"  输出目录: {OUT_DIR}")
    print(f"  svm_c1.pkl   正/负 二分类（{len(df)} 首）")
    print(f"  svm_c2.pkl   C2 粗类 4分类（{len(df_c2)} 首）")
    print(f"  svm_c3.pkl   C3 细类 {len(valid_c3)}分类（{len(df_c3)} 首）")
    print(f"  cls_features.npy  BERT 特征矩阵缓存")
    print("=" * 62)


if __name__ == "__main__":
    main()
