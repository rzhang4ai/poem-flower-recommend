"""
07_step5_c2c3_svm.py
====================
用 Golden Dataset 的 gold_train 训练 SVM_C2 / SVM_C3，
在 gold_test 上评估（伪标签作为代理指标），输出各层级的决策报告。

决策规则：
  C2 Macro-F1 > 0.55  →  保留 C2 分类器
  C3 类别 F1 > 0.35   →  该 C3 类别在最终分类器中保留
  否则                →  不建议使用该层级

注意：
  - 测试分数是"伪标签准确率"（proxy metric），非人工金标
  - C3 训练只保留 golden 中样本数 >= MIN_C3_SAMPLES 的类别

输出：
  output/svm_models/svm_c2_final.pkl      C2 SVM 模型
  output/svm_models/svm_c3_final.pkl      C3 SVM 模型
  output/eval_results/c2c3_eval_report.txt 评估+决策报告
  output/eval_results/final_clf_config.json 最终分类器配置
"""

from __future__ import annotations

import json
import pickle
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
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer

# ===========================================================================
# 配置
# ===========================================================================
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

GOLD_SPLIT    = _SCRIPT_DIR / "output" / "splits"       / "golden_split.csv"
FCCPSL_CSV    = _SCRIPT_DIR / "output" / "lexicon"       / "fccpsl_terms_only.csv"
MODEL_DIR     = _PROJECT_ROOT / "models" / "bert_ccpoem"
OUT_SVM       = _SCRIPT_DIR / "output" / "svm_models"
OUT_EVAL      = _SCRIPT_DIR / "output" / "eval_results"
OUT_SVM.mkdir(parents=True, exist_ok=True)
OUT_EVAL.mkdir(parents=True, exist_ok=True)

# Gold CLS 缓存（来自 07_step2_svm_train.py，若存在则复用）
GOLD_FEAT_NPY = OUT_SVM / "cls_features.npy"
GOLD_FEAT_IDX = OUT_SVM / "cls_index.csv"

MIN_C3_SAMPLES = 30      # C3 类别最小样本数阈值
BATCH_SIZE     = 32
MAX_LENGTH     = 128

# 决策阈值
C2_KEEP_THRESHOLD = 0.55
C3_KEEP_THRESHOLD = 0.35

C2_ZH = {
    "pleasure": "愉悦",
    "favour":   "好感赞赏",
    "sadness":  "悲伤",
    "disgust":  "厌恶苦闷",
    "surprise": "惊奇",
}
C1_CONSTRAINT = {
    "pleasure": "positive",
    "favour":   "positive",
    "surprise": "positive",
    "sadness":  "negative",
    "disgust":  "negative",
}


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
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LENGTH)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            vecs.append(out.last_hidden_state[:, 0, :].cpu().float().numpy())
    return np.vstack(vecs) if vecs else np.zeros((0, 512), dtype=np.float32)


# ===========================================================================
# SVM 训练 + 评估
# ===========================================================================
def train_eval_svm(
    X_train: np.ndarray,
    y_train: List[str],
    X_test: np.ndarray,
    y_test: List[str],
    label_zh: Optional[Dict[str, str]] = None,
    title: str = "SVM",
) -> Tuple[SVC, LabelEncoder, dict, str]:
    """训练 SVM，在 test 上评估，返回 (模型, 编码器, 结果字典, 报告字符串)。"""
    le = LabelEncoder()
    all_labels = sorted(set(y_train) | set(y_test))
    le.fit(all_labels)

    y_tr = le.transform(y_train)
    y_te = le.transform(y_test)

    clf = SVC(kernel="linear", probability=True,
              class_weight="balanced", random_state=42)
    clf.fit(X_tr := X_train, y_tr)
    y_pred = clf.predict(X_test)

    if label_zh:
        target_names = [f"{c}({label_zh.get(c,'')})" for c in le.classes_]
    else:
        target_names = list(le.classes_)

    acc = accuracy_score(y_te, y_pred)
    rpt_dict = classification_report(
        y_te, y_pred, target_names=target_names,
        digits=4, zero_division=0, output_dict=True,
    )
    rpt_str = classification_report(
        y_te, y_pred, target_names=target_names,
        digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_te, y_pred)

    header = [
        f"\n{'='*60}",
        f"  {title}  (train={len(y_train)}, test={len(y_test)})",
        f"{'='*60}",
        f"  Accuracy:  {acc:.4f}",
        f"  Macro-F1:  {rpt_dict['macro avg']['f1-score']:.4f}",
        "",
        rpt_str,
        "  混淆矩阵（行=真实，列=预测）:",
    ]
    for i, cls in enumerate(le.classes_):
        row_str = "  ".join(f"{v:5d}" for v in cm[i])
        header.append(f"  {cls[:12]:>12}: {row_str}")
    report = "\n".join(header)
    print(report)

    # 整理 per_class（去掉括号前缀，还原为英文类名）
    per_class = {}
    for cls in le.classes_:
        key = f"{cls}({label_zh.get(cls,'')})" if label_zh else cls
        per_class[cls] = {
            "f1":        rpt_dict.get(key, {}).get("f1-score", 0.0),
            "precision": rpt_dict.get(key, {}).get("precision", 0.0),
            "recall":    rpt_dict.get(key, {}).get("recall", 0.0),
            "support":   rpt_dict.get(key, {}).get("support", 0),
        }

    result = {
        "accuracy":    float(acc),
        "macro_f1":    float(rpt_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(rpt_dict["weighted avg"]["f1-score"]),
        "per_class":   per_class,
        "cm":          cm.tolist(),
        "labels":      le.classes_.tolist(),
    }
    return clf, le, result, report


# ===========================================================================
# 决策报告
# ===========================================================================
def build_decision_report(
    c2_result: dict,
    c3_result: dict,
    c3_excluded: dict,
    method_a_result: Optional[dict],
) -> Tuple[str, dict]:
    """生成决策报告，返回 (报告文本, final_clf_config)。"""
    lines = [
        "=" * 64,
        "  最终分类器配置决策报告",
        "=" * 64,
        "",
        "【层级1：5极性分类】",
    ]
    if method_a_result:
        mf1 = method_a_result.get("macro_f1", 0.0)
        lines.append(f"  SVM Macro-F1 = {mf1:.4f}  （详见 comparison_5pol.txt）")
        best_5pol = "finetune_bert" if False else "svm"  # 由对比决定，默认SVM
    else:
        lines.append("  （需运行 step4a/4b 后查看）")
        best_5pol = "svm"

    lines += [
        "",
        "【层级2：C2 粗类分类】",
        f"  测试集 Macro-F1 = {c2_result['macro_f1']:.4f}  "
        f"(阈值 {C2_KEEP_THRESHOLD})",
    ]
    c2_keep = c2_result["macro_f1"] >= C2_KEEP_THRESHOLD
    lines.append(f"  → {'✓ 保留 C2 分类器' if c2_keep else '✗ C2 整体表现不达标，不建议使用'}")

    c2_keep_classes = {}
    lines += ["", "  各 C2 类别 F1："]
    for cls, stats in c2_result["per_class"].items():
        f1 = stats["f1"]
        keep = f1 >= C2_KEEP_THRESHOLD
        c2_keep_classes[cls] = keep
        mark = "✓" if keep else "✗"
        lines.append(f"  {mark} {cls:<12s}({C2_ZH.get(cls,''):6s})  F1={f1:.4f}  n={stats['support']}")

    lines += [
        "",
        "【层级3：C3 细粒度分类】",
        f"  测试集 Macro-F1 = {c3_result['macro_f1']:.4f}  "
        f"(各类阈值 {C3_KEEP_THRESHOLD})",
    ]

    c3_keep_classes = {}
    lines.append("\n  各 C3 类别 F1（训练样本充足的10类）：")
    for cls, stats in sorted(c3_result["per_class"].items(),
                              key=lambda x: -x[1]["f1"]):
        f1 = stats["f1"]
        keep = f1 >= C3_KEEP_THRESHOLD
        c3_keep_classes[cls] = keep
        mark = "✓" if keep else "✗"
        lines.append(f"  {mark} {cls:<12s}  F1={f1:.4f}  n={stats['support']}")

    if c3_excluded:
        lines += ["", "  排除类别（样本不足）："]
        for cls, n in c3_excluded.items():
            lines.append(f"    {cls}: {n}条")

    kept_c3 = [c for c, v in c3_keep_classes.items() if v]
    lines += [
        "",
        f"  → C3 建议保留类别 ({len(kept_c3)}/{len(c3_keep_classes)}): {kept_c3}",
    ]

    lines += [
        "",
        "=" * 64,
        "  最终推理优先级建议",
        "=" * 64,
        "  层1（5极性）    → 始终输出，置信度=SVM概率",
        "  层2（C2）       → " + ("输出，但 pleasure 类需注意 F1 偏低" if c2_keep else "暂不推荐输出"),
        "  层3（C3）       → 仅输出以下类别（F1≥0.35）：",
        f"                    {kept_c3}",
        "  一致性约束      → C2/C3 输出必须与层1极性匹配",
        "                    (正极性→favour/pleasure，负极性→sadness/disgust)",
        "=" * 64,
    ]

    report = "\n".join(lines)
    print(report)

    # 最终配置 JSON
    config = {
        "version": "1.0",
        "layer1_5pol": {
            "method": best_5pol,
            "model":  "svm_5pol.pkl",
        },
        "layer2_c2": {
            "enabled":       c2_keep,
            "model":         "svm_c2_final.pkl",
            "keep_classes":  [c for c, v in c2_keep_classes.items() if v],
            "macro_f1_test": c2_result["macro_f1"],
        },
        "layer3_c3": {
            "enabled":       len(kept_c3) > 0,
            "model":         "svm_c3_final.pkl",
            "keep_classes":  kept_c3,
            "macro_f1_test": c3_result["macro_f1"],
        },
        "polarity_constraint": {
            "positive_c2": ["favour", "pleasure", "surprise"],
            "negative_c2": ["sadness", "disgust"],
        },
    }
    return report, config


# ===========================================================================
# 主函数
# ===========================================================================
def main() -> None:
    # ── 检查依赖 ──────────────────────────────────────────────────
    if not GOLD_SPLIT.exists():
        raise FileNotFoundError(
            f"Golden 切分文件不存在: {GOLD_SPLIT}\n请先运行 07_step3_split.py"
        )

    # ── 加载 Golden 切分 ──────────────────────────────────────────
    print(f"加载 Golden 切分: {GOLD_SPLIT}")
    df_gold = pd.read_csv(GOLD_SPLIT)

    # 加载标签映射
    df_lex = pd.read_csv(FCCPSL_CSV)
    c3_to_c2 = dict(zip(df_lex["C3"], df_lex["C2"]))
    c3_to_zh = dict(zip(df_lex["C3"], df_lex["C3_zh"]))
    df_gold["c2"] = df_gold["pseudo_label"].map(c3_to_c2)

    df_train = df_gold[df_gold["split"] == "train"].reset_index(drop=True)
    df_test  = df_gold[df_gold["split"] == "test"].reset_index(drop=True)
    print(f"  gold_train: {len(df_train)}  gold_test: {len(df_test)}")

    # ── 提取 / 读取 CLS 特征 ──────────────────────────────────────
    if GOLD_FEAT_NPY.exists() and GOLD_FEAT_IDX.exists():
        print(f"读取缓存特征: {GOLD_FEAT_NPY}")
        X_all  = np.load(GOLD_FEAT_NPY)
        idx_df = pd.read_csv(GOLD_FEAT_IDX)
        print(f"  shape: {X_all.shape}")
        if len(X_all) != len(df_gold):
            raise RuntimeError(
                f"缓存特征行数 ({len(X_all)}) 与 golden_split.csv 行数 ({len(df_gold)}) 不符。"
                "请删除缓存 cls_features.npy 后重新运行。"
            )
    else:
        tok, mdl, device = load_bert(str(MODEL_DIR))
        print(f"提取 {len(df_gold)} 首诗的 [CLS] 向量...")
        X_all = extract_cls(df_gold["text"].tolist(), tok, mdl, device)
        np.save(GOLD_FEAT_NPY, X_all)
        df_gold.to_csv(GOLD_FEAT_IDX, index=False, encoding="utf-8-sig")
        print(f"  已缓存 shape={X_all.shape}")

    train_mask = df_gold["split"] == "train"
    test_mask  = df_gold["split"] == "test"
    X_train = X_all[train_mask.values]
    X_test  = X_all[test_mask.values]

    all_reports = []

    # ================================================================
    # C2 SVM（4类，排除 surprise）
    # ================================================================
    print("\n" + "=" * 60)
    print("训练 SVM_C2（C2 粗类，排除 surprise）")

    c2_mask_tr = (df_train["c2"] != "surprise") & df_train["c2"].notna()
    c2_mask_te = (df_test["c2"]  != "surprise") & df_test["c2"].notna()
    X_c2_tr = X_train[c2_mask_tr.values]
    X_c2_te = X_test[c2_mask_te.values]
    y_c2_tr = df_train.loc[c2_mask_tr, "c2"].tolist()
    y_c2_te = df_test.loc[c2_mask_te,  "c2"].tolist()

    clf_c2, le_c2, c2_result, rpt_c2 = train_eval_svm(
        X_c2_tr, y_c2_tr, X_c2_te, y_c2_te,
        label_zh=C2_ZH,
        title="SVM_C2  粗类 4分类（proxy metric on gold_test）",
    )
    all_reports.append(rpt_c2)

    with open(OUT_SVM / "svm_c2_final.pkl", "wb") as f:
        pickle.dump({"clf": clf_c2, "le": le_c2, "c3_to_c2": c3_to_c2}, f)

    # ================================================================
    # C3 SVM（≥MIN_C3_SAMPLES 的类别）
    # ================================================================
    print("\n" + "=" * 60)
    print(f"训练 SVM_C3（C3 细粒度，保留 ≥{MIN_C3_SAMPLES} 条的类别）")

    c3_counts  = df_train["pseudo_label"].value_counts()
    valid_c3   = c3_counts[c3_counts >= MIN_C3_SAMPLES].index.tolist()
    c3_excluded = c3_counts[c3_counts < MIN_C3_SAMPLES].to_dict()
    print(f"  保留类别 ({len(valid_c3)}): {valid_c3}")
    print(f"  排除类别: {c3_excluded}")

    c3_mask_tr = df_train["pseudo_label"].isin(valid_c3)
    c3_mask_te = df_test["pseudo_label"].isin(valid_c3)
    X_c3_tr = X_train[c3_mask_tr.values]
    X_c3_te = X_test[c3_mask_te.values]
    y_c3_tr = df_train.loc[c3_mask_tr, "pseudo_label"].tolist()
    y_c3_te = df_test.loc[c3_mask_te,  "pseudo_label"].tolist()

    clf_c3, le_c3, c3_result, rpt_c3 = train_eval_svm(
        X_c3_tr, y_c3_tr, X_c3_te, y_c3_te,
        label_zh=c3_to_zh,
        title=f"SVM_C3  细粒度 {len(valid_c3)}分类（proxy metric on gold_test）",
    )
    all_reports.append(rpt_c3)

    with open(OUT_SVM / "svm_c3_final.pkl", "wb") as f:
        pickle.dump({
            "clf": clf_c3, "le": le_c3,
            "valid_classes": valid_c3,
            "c3_to_c2": c3_to_c2, "c3_to_zh": c3_to_zh,
        }, f)

    # ================================================================
    # 决策报告
    # ================================================================
    print("\n" + "=" * 60)
    print("生成决策报告...")

    method_a_result = None
    result_a_path = OUT_EVAL / "method_a_svm_5pol.json"
    if result_a_path.exists():
        with open(result_a_path) as f:
            method_a_result = json.load(f)

    dec_report, final_config = build_decision_report(
        c2_result, c3_result, c3_excluded, method_a_result
    )
    all_reports.append(dec_report)

    # ── 保存 ─────────────────────────────────────────────────────
    full_report = "\n".join(all_reports)
    out_report = OUT_EVAL / "c2c3_eval_report.txt"
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(full_report + "\n")

    out_config = OUT_EVAL / "final_clf_config.json"
    with open(out_config, "w", encoding="utf-8") as f:
        json.dump(final_config, f, ensure_ascii=False, indent=2)

    print(f"\n评估报告已保存: {out_report}")
    print(f"分类器配置已保存: {out_config}")
    print(f"\n{'='*60}")
    print(f"  C2 Macro-F1: {c2_result['macro_f1']:.4f}")
    print(f"  C3 Macro-F1: {c3_result['macro_f1']:.4f}")
    c3_keep_n = sum(1 for c,s in c3_result["per_class"].items()
                    if s["f1"] >= C3_KEEP_THRESHOLD)
    print(f"  C3 可信类别: {c3_keep_n}/{len(valid_c3)} 个 (F1≥{C3_KEEP_THRESHOLD})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
