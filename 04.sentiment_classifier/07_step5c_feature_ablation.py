"""
07_step5c_feature_ablation.py
=============================
特征消融实验：测试 4 种特征组合 × 2 个分类任务

特征组合：
  F1  Base CLS       512维，来自 bert_ccpoem（冻结）               ← 已有基线
  F2  Finetuned CLS  512维，来自 ccpoem_sentiment_ft（微调后）     ← 路线1
  F3  Base CLS + Lex 512+15=527维，拼接 FCCPSL 15类 IDF 得分      ← 路线2
  F4  FT CLS + Lex   512+15=527维，微调CLS + 词典分数             ← 路线1+2

分类任务：
  Task-C2  3类：positive / neg_sorrow / neg_anger   （方案C，上次最优）
  Task-C3  10类：C3 细粒度，保留 ≥30 条的类别

每个实验用 StandardScaler 归一化特征，linear SVC class_weight=balanced。

输出：
  output/svm_models/ft_cls_features.npy         微调BERT CLS 特征缓存
  output/eval_results/feature_ablation_report.txt  完整消融报告
  output/svm_models/svm_c2_ablation_best.pkl     C2 消融最优模型
  output/svm_models/svm_c3_ablation_best.pkl     C3 消融最优模型
  output/eval_results/final_clf_config.json       更新最终配置
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer

# ===========================================================================
# 路径配置
# ===========================================================================
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

GOLD_SPLIT    = _SCRIPT_DIR / "output" / "splits"      / "golden_split.csv"
FCCPSL_CSV    = _SCRIPT_DIR / "output" / "lexicon"      / "fccpsl_terms_only.csv"
BASE_FEAT_NPY = _SCRIPT_DIR / "output" / "svm_models"   / "cls_features.npy"
FT_FEAT_NPY   = _SCRIPT_DIR / "output" / "svm_models"   / "ft_cls_features.npy"
BASE_MODEL    = _PROJECT_ROOT / "models" / "bert_ccpoem"
FT_MODEL      = _PROJECT_ROOT / "models" / "ccpoem_sentiment_ft"
OUT_SVM       = _SCRIPT_DIR / "output" / "svm_models"
OUT_EVAL      = _SCRIPT_DIR / "output" / "eval_results"
OUT_SVM.mkdir(parents=True, exist_ok=True)
OUT_EVAL.mkdir(parents=True, exist_ok=True)

BATCH_SIZE    = 32
MAX_LENGTH    = 128
MIN_C3_SAMPLES = 30

# C2 方案C 映射（路线三最优结果）
CUSTOM_C2_3CLS: Dict[str, str] = {
    "praise":"positive","like":"positive","faith":"positive",
    "joy":"positive","ease":"positive","wish":"positive","peculiar":"positive",
    "sorrow":"neg_sorrow","guilt":"neg_sorrow","miss":"neg_sorrow","fear":"neg_sorrow",
    "vexed":"neg_anger","criticize":"neg_anger","anger":"neg_anger","misgive":"neg_anger",
}
C2_LABEL_ZH = {"positive":"积极","neg_sorrow":"负向-悲伤","neg_anger":"负向-愤懑"}
C3_LABEL_ZH_MAP: Dict[str, str] = {}   # 运行时从 CSV 填充


# ===========================================================================
# BERT 特征提取
# ===========================================================================
def resolve_ft_model(model_root: Path) -> Path:
    """优先用根目录 config.json，否则找最新 checkpoint。"""
    if (model_root / "config.json").exists():
        return model_root
    ckpt_dir = model_root / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(
            [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint")],
            key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0,
        )
        if ckpts:
            return ckpts[-1]
    raise FileNotFoundError(f"找不到微调模型: {model_root}")


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_cls_batch(
    texts: List[str], tok, mdl, device: str
) -> np.ndarray:
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LENGTH)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc, output_hidden_states=False)
            vecs.append(out.last_hidden_state[:, 0, :].cpu().float().numpy())
            done = min(i + BATCH_SIZE, len(texts))
            if done % 320 == 0 or done == len(texts):
                print(f"    [{done}/{len(texts)}]...")
    return np.vstack(vecs)


def load_or_extract(
    npy_path: Path,
    texts: List[str],
    model_path: Path,
    label: str,
) -> np.ndarray:
    if npy_path.exists():
        X = np.load(npy_path)
        if len(X) == len(texts):
            print(f"  [{label}] 读取缓存: {npy_path}  shape={X.shape}")
            return X
        print(f"  [{label}] 缓存行数不符，重新提取...")

    device = _device()
    print(f"  [{label}] 提取 CLS 特征 (device={device})...")
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    mdl = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
    mdl.to(device).eval()
    X = extract_cls_batch(texts, tok, mdl, device)
    np.save(npy_path, X)
    print(f"  [{label}] 已缓存 shape={X.shape} -> {npy_path}")
    return X


# ===========================================================================
# 训练 + 评估
# ===========================================================================
def train_eval_pipeline(
    X_tr: np.ndarray, y_tr: List[str],
    X_te: np.ndarray, y_te: List[str],
    label_zh: Optional[Dict[str, str]] = None,
    title: str = "",
) -> Tuple[Pipeline, LabelEncoder, dict, str]:
    """StandardScaler + linear SVC，返回 (pipeline, le, result, report)。"""
    le = LabelEncoder()
    le.fit(sorted(set(y_tr) | set(y_te)))
    y_tr_enc = le.transform(y_tr)
    y_te_enc = le.transform(y_te)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    SVC(kernel="linear", probability=True,
                       class_weight="balanced", random_state=42)),
    ])
    pipe.fit(X_tr, y_tr_enc)
    y_pred = pipe.predict(X_te)

    acc = accuracy_score(y_te_enc, y_pred)
    tn  = [f"{c}({label_zh.get(c,'')})" if label_zh else c for c in le.classes_]
    rpt_dict = classification_report(
        y_te_enc, y_pred, target_names=tn,
        digits=4, zero_division=0, output_dict=True,
    )
    rpt_str = classification_report(
        y_te_enc, y_pred, target_names=tn,
        digits=4, zero_division=0,
    )
    per_class = {}
    for cls, tn_name in zip(le.classes_, tn):
        per_class[cls] = {
            "f1":      float(rpt_dict.get(tn_name, {}).get("f1-score",  0.0)),
            "recall":  float(rpt_dict.get(tn_name, {}).get("recall",    0.0)),
            "support": int(rpt_dict.get(tn_name, {}).get("support", 0)),
        }
    result = {
        "accuracy":    float(acc),
        "macro_f1":    float(rpt_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(rpt_dict["weighted avg"]["f1-score"]),
        "per_class":   per_class,
        "feat_dim":    int(X_tr.shape[1]),
    }

    header = [
        f"\n{'─'*60}",
        f"  {title}  (dim={X_tr.shape[1]}, train={len(y_tr)}, test={len(y_te)})",
        f"  Accuracy={acc:.4f}  Macro-F1={result['macro_f1']:.4f}",
        "", rpt_str,
    ]
    report = "\n".join(header)
    print(report)
    return pipe, le, result, report


# ===========================================================================
# 消融汇总表
# ===========================================================================
FEAT_NAMES = ["F1_BaseCLS", "F2_FinetunedCLS", "F3_BaseCLS+Lex", "F4_FT+Lex"]

def ablation_table(
    results_c2: Dict[str, dict],
    results_c3: Dict[str, dict],
) -> str:
    lines = [
        "\n" + "=" * 72,
        "  特征消融实验汇总",
        "  任务-C2（3类方案C）  ×  任务-C3（10类细粒度）",
        "=" * 72,
        f"  {'特征方案':<22s}  {'dim':>5}  "
        f"{'C2 Mac-F1':>10}  {'C2 Acc':>7}  "
        f"{'C3 Mac-F1':>10}  {'C3 Acc':>7}",
        "-" * 72,
    ]
    for fname in FEAT_NAMES:
        rc2 = results_c2.get(fname, {})
        rc3 = results_c3.get(fname, {})
        dim = rc2.get("feat_dim", 0)
        lines.append(
            f"  {fname:<22s}  {dim:>5}  "
            f"{rc2.get('macro_f1',0):>10.4f}  {rc2.get('accuracy',0):>7.4f}  "
            f"{rc3.get('macro_f1',0):>10.4f}  {rc3.get('accuracy',0):>7.4f}"
        )
    # 最优标记
    best_c2 = max(results_c2, key=lambda k: results_c2[k].get("macro_f1", 0))
    best_c3 = max(results_c3, key=lambda k: results_c3[k].get("macro_f1", 0))
    lines += [
        "-" * 72,
        f"  C2 最优方案: {best_c2}  Macro-F1={results_c2[best_c2]['macro_f1']:.4f}",
        f"  C3 最优方案: {best_c3}  Macro-F1={results_c3[best_c3]['macro_f1']:.4f}",
        "",
    ]
    # 单类 F1 对比（C2）
    lines += ["  C2 各类 F1 对比：",
              f"  {'类别':<16s}" + "".join(f"  {n[:12]:>12s}" for n in FEAT_NAMES)]
    sample_c2 = next(iter(results_c2.values()))
    for cls in sample_c2.get("per_class", {}):
        row = f"  {cls:<16s}"
        for fn in FEAT_NAMES:
            f1 = results_c2.get(fn, {}).get("per_class", {}).get(cls, {}).get("f1", float("nan"))
            row += f"  {f1:>12.4f}" if not (isinstance(f1, float) and np.isnan(f1)) else f"  {'--':>12s}"
        lines.append(row)

    # 单类 F1 对比（C3）
    lines += ["", "  C3 各类 F1 对比：",
              f"  {'类别':<16s}" + "".join(f"  {n[:12]:>12s}" for n in FEAT_NAMES)]
    sample_c3 = next(iter(results_c3.values()))
    for cls in sample_c3.get("per_class", {}):
        row = f"  {cls:<16s}"
        for fn in FEAT_NAMES:
            f1 = results_c3.get(fn, {}).get("per_class", {}).get(cls, {}).get("f1", float("nan"))
            row += f"  {f1:>12.4f}" if not (isinstance(f1, float) and np.isnan(f1)) else f"  {'--':>12s}"
        lines.append(row)

    lines += [
        "",
        "  分析结论：",
        f"  • 路线1（F2 vs F1）对 C2 改进: "
        f"{results_c2.get('F2_FinetunedCLS',{}).get('macro_f1',0)-results_c2.get('F1_BaseCLS',{}).get('macro_f1',0):+.4f}",
        f"  • 路线2（F3 vs F1）对 C2 改进: "
        f"{results_c2.get('F3_BaseCLS+Lex',{}).get('macro_f1',0)-results_c2.get('F1_BaseCLS',{}).get('macro_f1',0):+.4f}",
        f"  • 路线1+2（F4 vs F1）对 C2 改进: "
        f"{results_c2.get('F4_FT+Lex',{}).get('macro_f1',0)-results_c2.get('F1_BaseCLS',{}).get('macro_f1',0):+.4f}",
        f"  • 路线1（F2 vs F1）对 C3 改进: "
        f"{results_c3.get('F2_FinetunedCLS',{}).get('macro_f1',0)-results_c3.get('F1_BaseCLS',{}).get('macro_f1',0):+.4f}",
        f"  • 路线2（F3 vs F1）对 C3 改进: "
        f"{results_c3.get('F3_BaseCLS+Lex',{}).get('macro_f1',0)-results_c3.get('F1_BaseCLS',{}).get('macro_f1',0):+.4f}",
        f"  • 路线1+2（F4 vs F1）对 C3 改进: "
        f"{results_c3.get('F4_FT+Lex',{}).get('macro_f1',0)-results_c3.get('F1_BaseCLS',{}).get('macro_f1',0):+.4f}",
        "=" * 72,
    ]
    report = "\n".join(lines)
    print(report)
    return report


# ===========================================================================
# 主函数
# ===========================================================================
def main() -> None:
    # ── 加载数据 ──────────────────────────────────────────────────
    for p in [GOLD_SPLIT, BASE_FEAT_NPY]:
        if not p.exists():
            raise FileNotFoundError(f"缺失: {p}\n请先运行 07_step3_split.py 和 07_step5_c2c3_svm.py")

    print("=" * 60)
    print("加载数据...")
    df    = pd.read_csv(GOLD_SPLIT)
    X_base = np.load(BASE_FEAT_NPY)
    texts  = df["text"].tolist()

    # 词典分数矩阵（15列，N×15）
    score_cols = sorted([c for c in df.columns if c.startswith("score_")])
    LEX_SCORES = df[score_cols].values.astype(np.float32)
    print(f"  样本数: {len(df)}  base CLS shape: {X_base.shape}")
    print(f"  词典分数: {len(score_cols)} 列")

    # 加载标签映射
    lex_df = pd.read_csv(FCCPSL_CSV)
    global C3_LABEL_ZH_MAP
    C3_LABEL_ZH_MAP = dict(zip(lex_df["C3"], lex_df["C3_zh"]))

    # C2 标签
    df["c2_3cls"] = df["pseudo_label"].map(CUSTOM_C2_3CLS)

    # C3 过滤（≥30条）
    c3_counts = df[df["split"] == "train"]["pseudo_label"].value_counts()
    valid_c3  = c3_counts[c3_counts >= MIN_C3_SAMPLES].index.tolist()

    # ── 提取微调 BERT CLS（路线1）────────────────────────────────
    ft_model_dir = resolve_ft_model(FT_MODEL)
    X_ft = load_or_extract(FT_FEAT_NPY, texts, ft_model_dir, "微调BERT")

    # ── 构建 4 种特征矩阵 ─────────────────────────────────────────
    FEAT_SETS = {
        "F1_BaseCLS":        X_base,
        "F2_FinetunedCLS":   X_ft,
        "F3_BaseCLS+Lex":    np.hstack([X_base, LEX_SCORES]),
        "F4_FT+Lex":         np.hstack([X_ft,   LEX_SCORES]),
    }
    for name, X in FEAT_SETS.items():
        print(f"  {name}: dim={X.shape[1]}")

    # ── 切分辅助函数 ──────────────────────────────────────────────
    def split_xy(X_full: np.ndarray, label_col: str, valid_labels=None):
        mask_tr = (df["split"] == "train") & df[label_col].notna()
        mask_te = (df["split"] == "test")  & df[label_col].notna()
        if valid_labels is not None:
            mask_tr &= df[label_col].isin(valid_labels)
            mask_te &= df[label_col].isin(valid_labels)
        return (X_full[mask_tr.values], df.loc[mask_tr, label_col].tolist(),
                X_full[mask_te.values], df.loc[mask_te, label_col].tolist())

    # ── 消融实验 ──────────────────────────────────────────────────
    results_c2: Dict[str, dict] = {}
    results_c3: Dict[str, dict] = {}
    all_rpts: List[str] = []
    best_pipes: Dict[str, dict] = {}

    for fname, X_full in FEAT_SETS.items():

        print(f"\n{'='*60}")
        print(f"特征方案: {fname}  (dim={X_full.shape[1]})")

        # Task-C2
        X_tr, y_tr, X_te, y_te = split_xy(X_full, "c2_3cls")
        print(f"\n  [C2-3类]")
        pipe_c2, le_c2, res_c2, rpt_c2 = train_eval_pipeline(
            X_tr, y_tr, X_te, y_te,
            label_zh=C2_LABEL_ZH,
            title=f"C2（3类）× {fname}",
        )
        results_c2[fname] = res_c2
        all_rpts.append(rpt_c2)
        best_pipes[f"c2_{fname}"] = {"pipe": pipe_c2, "le": le_c2, "macro_f1": res_c2["macro_f1"]}

        # Task-C3
        X_tr3, y_tr3, X_te3, y_te3 = split_xy(X_full, "pseudo_label", valid_c3)
        print(f"\n  [C3-{len(valid_c3)}类]")
        pipe_c3, le_c3, res_c3, rpt_c3 = train_eval_pipeline(
            X_tr3, y_tr3, X_te3, y_te3,
            label_zh=C3_LABEL_ZH_MAP,
            title=f"C3（{len(valid_c3)}类）× {fname}",
        )
        results_c3[fname] = res_c3
        all_rpts.append(rpt_c3)
        best_pipes[f"c3_{fname}"] = {"pipe": pipe_c3, "le": le_c3, "macro_f1": res_c3["macro_f1"]}

    # ── 消融汇总 ──────────────────────────────────────────────────
    summary = ablation_table(results_c2, results_c3)
    all_rpts.append(summary)

    # ── 保存最优模型 ──────────────────────────────────────────────
    best_c2_name = max(results_c2, key=lambda k: results_c2[k]["macro_f1"])
    best_c3_name = max(results_c3, key=lambda k: results_c3[k]["macro_f1"])

    for task, name, suffix in [("c2", best_c2_name, "c2_ablation_best"),
                                ("c3", best_c3_name, "c3_ablation_best")]:
        info = best_pipes[f"{task}_{name}"]
        out_pkl = OUT_SVM / f"svm_{suffix}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump({
                "pipe":    info["pipe"],
                "le":      info["le"],
                "feat_scheme": name,
                "macro_f1":    info["macro_f1"],
                "valid_c3":    valid_c3 if task == "c3" else None,
                "c2_map":      CUSTOM_C2_3CLS if task == "c2" else None,
            }, f)
        print(f"最优 {task.upper()} 模型已保存: {out_pkl}  ({name})")

    # ── 更新 final_clf_config.json ────────────────────────────────
    config_path = OUT_EVAL / "final_clf_config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    def keep_classes(results: dict, name: str, threshold: float) -> list:
        return [c for c, s in results[name]["per_class"].items() if s["f1"] >= threshold]

    config["layer2_c2"]["ablation_best_feat"] = best_c2_name
    config["layer2_c2"]["ablation_macro_f1"]  = results_c2[best_c2_name]["macro_f1"]
    config["layer2_c2"]["keep_classes"]        = keep_classes(results_c2, best_c2_name, 0.40)
    config["layer2_c2"]["model"]               = "svm_c2_ablation_best.pkl"
    config["layer2_c2"]["enabled"]             = results_c2[best_c2_name]["macro_f1"] >= 0.50

    config["layer3_c3"]["ablation_best_feat"]  = best_c3_name
    config["layer3_c3"]["ablation_macro_f1"]   = results_c3[best_c3_name]["macro_f1"]
    config["layer3_c3"]["keep_classes"]        = keep_classes(results_c3, best_c3_name, 0.35)
    config["layer3_c3"]["model"]               = "svm_c3_ablation_best.pkl"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"最终配置已更新: {config_path}")

    # ── 保存报告 ──────────────────────────────────────────────────
    out_report = OUT_EVAL / "feature_ablation_report.txt"
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(all_rpts) + "\n")
    print(f"消融报告已保存: {out_report}")


if __name__ == "__main__":
    main()
