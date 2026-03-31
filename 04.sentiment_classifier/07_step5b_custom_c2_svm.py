"""
07_step5b_custom_c2_svm.py
==========================
路线三：重新设计语义更合理的 C2 标签体系，对比三套方案：

  方案A（旧DUTIR C2）：favour / disgust / sadness / pleasure  [4类，已有基线]
  方案B（4类语义重构）：pos_praise / pos_joy / neg_sorrow / neg_anger
  方案C（3类合并正向）：positive / neg_sorrow / neg_anger

标签重构逻辑：
  原 DUTIR 的 pleasure(joy+ease) 与 favour(praise+like+faith+wish) 在特征空间高度重叠
  → 方案B：把 wish 从 favour 移到 pos_joy 组（wish=渴望，更接近内心情感而非赞美）
             把 peculiar(惊奇) 也并入 pos_joy（惊奇是积极内心反应）
  → 方案C：直接合并全部正向类，只保留负向二分（悲伤 vs 愤懑）

输出：
  output/eval_results/custom_c2_comparison.txt  三套方案对比报告
  output/svm_models/svm_c2_best.pkl             最优方案的模型
  output/eval_results/final_clf_config.json      更新后的最终分类器配置
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# ===========================================================================
# 配置
# ===========================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent

GOLD_SPLIT  = _SCRIPT_DIR / "output" / "splits"       / "golden_split.csv"
FCCPSL_CSV  = _SCRIPT_DIR / "output" / "lexicon"       / "fccpsl_terms_only.csv"
FEAT_NPY    = _SCRIPT_DIR / "output" / "svm_models"    / "cls_features.npy"
FEAT_IDX    = _SCRIPT_DIR / "output" / "svm_models"    / "cls_index.csv"
OUT_SVM     = _SCRIPT_DIR / "output" / "svm_models"
OUT_EVAL    = _SCRIPT_DIR / "output" / "eval_results"
OUT_EVAL.mkdir(parents=True, exist_ok=True)

C2_KEEP_THRESHOLD = 0.50    # 方案总体 Macro-F1 达标线
CLS_KEEP_THRESHOLD = 0.40   # 单类 F1 达标线

# ===========================================================================
# 三套 C3 → 自定义 C2 映射
# ===========================================================================

# 方案A：旧 DUTIR C2（基线，直接从 fccpsl_terms_only.csv 读取）
# （不在这里硬编码，运行时从 CSV 读取）

# 方案B：4类语义重构
# 核心改动：wish + peculiar 归入 pos_joy；其余与旧DUTIR同
CUSTOM_C2_4CLS: Dict[str, str] = {
    "praise":    "pos_praise",   # 正向-赞美（称颂/欣赏/信念，指向外部对象）
    "like":      "pos_praise",
    "faith":     "pos_praise",
    "joy":       "pos_joy",      # 正向-喜乐（喜悦/安适/渴望/惊奇，内心状态）
    "ease":      "pos_joy",
    "wish":      "pos_joy",      # 关键改动：渴望 → 内心情感，非赞美外物
    "peculiar":  "pos_joy",      # 关键改动：惊奇 → 积极内心反应
    "sorrow":    "neg_sorrow",   # 负向-悲伤（悲痛/愧疚/思念/恐惧）
    "guilt":     "neg_sorrow",
    "miss":      "neg_sorrow",
    "fear":      "neg_sorrow",
    "vexed":     "neg_anger",    # 负向-愤懑（烦恼/批判/愤怒/疑虑）
    "criticize": "neg_anger",
    "anger":     "neg_anger",
    "misgive":   "neg_anger",
}

# 方案C：3类（合并全部正向）
CUSTOM_C2_3CLS: Dict[str, str] = {
    "praise":    "positive",
    "like":      "positive",
    "faith":     "positive",
    "joy":       "positive",
    "ease":      "positive",
    "wish":      "positive",
    "peculiar":  "positive",
    "sorrow":    "neg_sorrow",
    "guilt":     "neg_sorrow",
    "miss":      "neg_sorrow",
    "fear":      "neg_sorrow",
    "vexed":     "neg_anger",
    "criticize": "neg_anger",
    "anger":     "neg_anger",
    "misgive":   "neg_anger",
}

# 中文标签
LABEL_ZH = {
    # 方案B
    "pos_praise":  "正向-赞美",
    "pos_joy":     "正向-喜乐",
    "neg_sorrow":  "负向-悲伤",
    "neg_anger":   "负向-愤懑",
    # 方案C
    "positive":    "积极",
    # 方案A（部分沿用）
    "favour":      "好感赞赏",
    "pleasure":    "愉悦",
    "sadness":     "悲伤",
    "disgust":     "厌恶苦闷",
}


# ===========================================================================
# 工具函数
# ===========================================================================
def train_eval(
    X_tr: np.ndarray, y_tr: List[str],
    X_te: np.ndarray, y_te: List[str],
    label_zh: Optional[Dict[str, str]] = None,
    title: str = "SVM",
) -> Tuple[SVC, LabelEncoder, dict, str]:
    le = LabelEncoder()
    le.fit(sorted(set(y_tr) | set(y_te)))
    y_tr_enc = le.transform(y_tr)
    y_te_enc = le.transform(y_te)

    clf = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)
    clf.fit(X_tr, y_tr_enc)
    y_pred = clf.predict(X_te)

    tn = [f"{c}({label_zh.get(c,'')})" if label_zh else c for c in le.classes_]
    acc = accuracy_score(y_te_enc, y_pred)
    rpt_dict = classification_report(
        y_te_enc, y_pred, target_names=tn, digits=4, zero_division=0, output_dict=True
    )
    rpt_str = classification_report(
        y_te_enc, y_pred, target_names=tn, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_te_enc, y_pred)

    lines = [
        f"\n{'='*60}",
        f"  {title}  (train={len(y_tr)}, test={len(y_te)})",
        f"{'='*60}",
        f"  Accuracy:  {acc:.4f}   Macro-F1: {rpt_dict['macro avg']['f1-score']:.4f}",
        "", rpt_str,
        "  混淆矩阵（行=真实，列=预测）:",
    ]
    for i, cls in enumerate(le.classes_):
        row_str = " ".join(f"{v:5d}" for v in cm[i])
        zh = label_zh.get(cls, "") if label_zh else ""
        lines.append(f"  {cls}({zh}): {row_str}")
    report = "\n".join(lines)
    print(report)

    per_class = {}
    for cls, tn_name in zip(le.classes_, tn):
        per_class[cls] = {
            "f1":        rpt_dict.get(tn_name, {}).get("f1-score",  0.0),
            "precision": rpt_dict.get(tn_name, {}).get("precision", 0.0),
            "recall":    rpt_dict.get(tn_name, {}).get("recall",    0.0),
            "support":   int(rpt_dict.get(tn_name, {}).get("support", 0)),
        }
    result = {
        "accuracy":    float(acc),
        "macro_f1":    float(rpt_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(rpt_dict["weighted avg"]["f1-score"]),
        "per_class":   per_class,
        "labels":      le.classes_.tolist(),
    }
    return clf, le, result, report


def make_split_data(
    df_train: pd.DataFrame, df_test: pd.DataFrame,
    X_all: np.ndarray, df_all: pd.DataFrame,
    c2_col: str,
) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """从 golden_split 切出 train/test 特征，排除 NaN 标签行。"""
    train_idx = df_all.index[df_all["split"] == "train"].tolist()
    test_idx  = df_all.index[df_all["split"] == "test"].tolist()

    df_tr = df_all.loc[train_idx].copy()
    df_te = df_all.loc[test_idx].copy()

    mask_tr = df_tr[c2_col].notna()
    mask_te = df_te[c2_col].notna()

    X_tr = X_all[np.array(train_idx)[mask_tr.values]]
    X_te = X_all[np.array(test_idx)[mask_te.values]]
    y_tr = df_tr.loc[mask_tr, c2_col].tolist()
    y_te = df_te.loc[mask_te, c2_col].tolist()
    return X_tr, y_tr, X_te, y_te


# ===========================================================================
# 对比汇总报告
# ===========================================================================
def comparison_summary(results: Dict[str, dict]) -> str:
    """生成三套方案的对比表。"""
    lines = [
        "\n" + "=" * 64,
        "  三套 C2 方案对比（proxy metric on gold_test）",
        "=" * 64,
        f"  {'方案':<30s}  {'类别数':>4}  {'Macro-F1':>8}  {'Accuracy':>8}",
        "-" * 64,
    ]
    best_macro = max(r["macro_f1"] for r in results.values())
    for name, r in results.items():
        n_cls = len(r["labels"])
        mf1   = r["macro_f1"]
        acc   = r["accuracy"]
        mark  = " ★" if mf1 == best_macro else ""
        lines.append(f"  {name:<30s}  {n_cls:>4}  {mf1:>8.4f}  {acc:>8.4f}{mark}")

    lines += ["", "  各方案单类 F1：", f"  {'类别':>14s}" + "".join(f"  {n[:6]:>8s}" for n in results)]
    # 收集所有类别
    all_cls_set = set()
    for r in results.values():
        all_cls_set |= set(r["per_class"].keys())
    for cls in sorted(all_cls_set):
        zh = LABEL_ZH.get(cls, "")
        row = f"  {cls}({zh})"[:18].ljust(18)
        for r in results.values():
            f1 = r["per_class"].get(cls, {}).get("f1", float("nan"))
            row += f"  {f1:>8.4f}" if not np.isnan(f1) else f"  {'--':>8s}"
        lines.append(row)

    # 决策
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_r    = results[best_name]
    lines += [
        "",
        "=" * 64,
        f"  最优方案: {best_name}（Macro-F1 = {best_r['macro_f1']:.4f}）",
    ]
    if best_r["macro_f1"] >= C2_KEEP_THRESHOLD:
        keep_cls = [c for c, s in best_r["per_class"].items() if s["f1"] >= CLS_KEEP_THRESHOLD]
        lines.append(f"  ✓ Macro-F1 ≥ {C2_KEEP_THRESHOLD}，建议保留 C2 分类器")
        lines.append(f"  ✓ F1≥{CLS_KEEP_THRESHOLD} 的类别: {keep_cls}")
    else:
        lines.append(f"  ✗ Macro-F1 < {C2_KEEP_THRESHOLD}，C2 仅作辅助参考输出")
    lines.append("=" * 64)

    report = "\n".join(lines)
    print(report)
    return report


# ===========================================================================
# 主函数
# ===========================================================================
def main() -> None:
    # ── 检查依赖 ──────────────────────────────────────────────────
    for p in [GOLD_SPLIT, FEAT_NPY, FEAT_IDX]:
        if not p.exists():
            raise FileNotFoundError(
                f"缺失文件: {p}\n"
                "请依次运行 07_step3_split.py → 07_step5_c2c3_svm.py 生成切分和特征缓存。"
            )

    # ── 加载 ──────────────────────────────────────────────────────
    print("加载数据与特征...")
    df = pd.read_csv(GOLD_SPLIT)
    X  = np.load(FEAT_NPY)

    if len(X) != len(df):
        raise RuntimeError(
            f"特征矩阵行数 ({len(X)}) ≠ golden_split.csv 行数 ({len(df)})，"
            "请删除 cls_features.npy 后重新运行 07_step5_c2c3_svm.py。"
        )

    # 加载旧 C2
    lex = pd.read_csv(FCCPSL_CSV)
    c3_to_c2_old = dict(zip(lex["C3"], lex["C2"]))
    df["c2_old"]  = df["pseudo_label"].map(c3_to_c2_old)
    df["c2_4cls"] = df["pseudo_label"].map(CUSTOM_C2_4CLS)
    df["c2_3cls"] = df["pseudo_label"].map(CUSTOM_C2_3CLS)

    print(f"  样本总数: {len(df)}  train: {(df['split']=='train').sum()}  test: {(df['split']=='test').sum()}")
    print(f"  特征 shape: {X.shape}")

    all_reports = []
    results: Dict[str, dict] = {}

    # ── 方案A：旧 DUTIR C2（排除 surprise）────────────────────────
    print("\n" + "=" * 60)
    print("方案A（基线）：旧 DUTIR C2（4类，排除 surprise）")
    mask_a = df["c2_old"] != "surprise"
    df_a   = df[mask_a].copy()
    X_a    = X[mask_a.values]

    tr_a = df_a["split"] == "train"
    te_a = df_a["split"] == "test"
    clf_a, le_a, res_a, rpt_a = train_eval(
        X_a[tr_a.values], df_a.loc[tr_a, "c2_old"].tolist(),
        X_a[te_a.values], df_a.loc[te_a, "c2_old"].tolist(),
        label_zh=LABEL_ZH,
        title="方案A  旧 DUTIR C2（favour/disgust/sadness/pleasure）",
    )
    results["A_旧DUTIR_4类"] = res_a
    all_reports.append(rpt_a)

    # ── 方案B：4类语义重构 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("方案B：4类语义重构（pos_praise / pos_joy / neg_sorrow / neg_anger）")

    tr_b = df["split"] == "train"
    te_b = df["split"] == "test"
    clf_b, le_b, res_b, rpt_b = train_eval(
        X[tr_b.values], df.loc[tr_b, "c2_4cls"].dropna().tolist(),
        X[te_b.values], df.loc[te_b, "c2_4cls"].dropna().tolist(),
        label_zh=LABEL_ZH,
        title="方案B  4类语义重构",
    )
    # 修正：需要正确对齐索引
    mask_b_tr = (df["split"] == "train") & df["c2_4cls"].notna()
    mask_b_te = (df["split"] == "test")  & df["c2_4cls"].notna()
    clf_b, le_b, res_b, rpt_b = train_eval(
        X[mask_b_tr.values], df.loc[mask_b_tr, "c2_4cls"].tolist(),
        X[mask_b_te.values], df.loc[mask_b_te, "c2_4cls"].tolist(),
        label_zh=LABEL_ZH,
        title="方案B  4类语义重构（pos_praise/pos_joy/neg_sorrow/neg_anger）",
    )
    results["B_语义重构_4类"] = res_b
    all_reports.append(rpt_b)

    # ── 方案C：3类（合并正向）────────────────────────────────────
    print("\n" + "=" * 60)
    print("方案C：3类（positive / neg_sorrow / neg_anger）")

    mask_c_tr = (df["split"] == "train") & df["c2_3cls"].notna()
    mask_c_te = (df["split"] == "test")  & df["c2_3cls"].notna()
    clf_c, le_c, res_c, rpt_c = train_eval(
        X[mask_c_tr.values], df.loc[mask_c_tr, "c2_3cls"].tolist(),
        X[mask_c_te.values], df.loc[mask_c_te, "c2_3cls"].tolist(),
        label_zh=LABEL_ZH,
        title="方案C  3类（合并正向：positive/neg_sorrow/neg_anger）",
    )
    results["C_合并正向_3类"] = res_c
    all_reports.append(rpt_c)

    # ── 对比报告 ──────────────────────────────────────────────────
    cmp_report = comparison_summary(results)
    all_reports.append(cmp_report)

    # ── 保存最优模型 ──────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    if best_name.startswith("A"):
        best_clf, best_le = clf_a, le_a
    elif best_name.startswith("B"):
        best_clf, best_le = clf_b, le_b
    else:
        best_clf, best_le = clf_c, le_c

    best_pkl = OUT_SVM / "svm_c2_best.pkl"
    with open(best_pkl, "wb") as f:
        pickle.dump({
            "clf":         best_clf,
            "le":          best_le,
            "scheme":      best_name,
            "c2_map":      (CUSTOM_C2_4CLS if best_name.startswith("B")
                            else (CUSTOM_C2_3CLS if best_name.startswith("C")
                                  else c3_to_c2_old)),
            "macro_f1":    results[best_name]["macro_f1"],
        }, f)
    print(f"\n最优模型已保存: {best_pkl}  ({best_name})")

    # ── 更新 final_clf_config.json ────────────────────────────────
    config_path = OUT_EVAL / "final_clf_config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    best_r    = results[best_name]
    keep_cls  = [c for c, s in best_r["per_class"].items() if s["f1"] >= CLS_KEEP_THRESHOLD]
    config["layer2_c2"] = {
        "enabled":        best_r["macro_f1"] >= C2_KEEP_THRESHOLD,
        "scheme":         best_name,
        "model":          "svm_c2_best.pkl",
        "keep_classes":   keep_cls,
        "macro_f1_test":  best_r["macro_f1"],
        "threshold_used": C2_KEEP_THRESHOLD,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"最终配置已更新: {config_path}")

    # ── 保存完整报告 ──────────────────────────────────────────────
    out_report = OUT_EVAL / "custom_c2_comparison.txt"
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(all_reports) + "\n")
    print(f"完整报告已保存: {out_report}")

    # ── 最终摘要 ──────────────────────────────────────────────────
    print(f"\n{'='*60}  总结  {'='*60}")
    print(f"  {'方案':<30s}  {'Macro-F1':>8}")
    for nm, r in results.items():
        flag = " ★" if nm == best_name else ""
        print(f"  {nm:<30s}  {r['macro_f1']:>8.4f}{flag}")
    print(f"\n  改进量（最优 vs 基线A）：{results[best_name]['macro_f1'] - results['A_旧DUTIR_4类']['macro_f1']:+.4f}")


if __name__ == "__main__":
    main()
