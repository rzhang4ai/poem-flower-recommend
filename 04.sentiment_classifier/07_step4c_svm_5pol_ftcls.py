"""
07_step4c_svm_5pol_ftcls.py
============================
补充对照实验：用微调BERT CLS特征训练5极性SVM（方法A'），
与方法A（基础BERT CLS SVM）和方法B（端到端微调BERT）三方对比。

背景：
  上一步消融实验（07_step5c_feature_ablation.py）证明，
  微调BERT CLS特征相比基础BERT CLS能提升约+0.07 Macro-F1。
  因此需要用相同的微调BERT提取FSPC全量5000首诗的CLS特征，
  再训练SVM做5极性分类，才能进行公平比较。

输出：
  output/svm_models/fspc_ft_cls_features.npy   FSPC微调BERT特征缓存
  output/eval_results/method_ap_svm_5pol.json  方法A'的评估结果
  output/eval_results/comparison_5pol_full.txt  三方最终对比报告
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List

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

FSPC_SPLIT    = _SCRIPT_DIR / "output" / "splits"     / "fspc_split.csv"
FT_MODEL      = _PROJECT_ROOT / "models" / "ccpoem_sentiment_ft"
FT_FEAT_NPY   = _SCRIPT_DIR / "output" / "svm_models" / "fspc_ft_cls_features.npy"
OUT_SVM       = _SCRIPT_DIR / "output" / "svm_models"
OUT_EVAL      = _SCRIPT_DIR / "output" / "eval_results"
OUT_SVM.mkdir(parents=True, exist_ok=True)
OUT_EVAL.mkdir(parents=True, exist_ok=True)

PREV_A_JSON   = OUT_EVAL / "method_a_svm_5pol.json"
PREV_B_JSON   = OUT_EVAL / "method_b_bert_5pol.json"

BATCH_SIZE  = 32
MAX_LENGTH  = 128

# FSPC 5极性标签（统一为 Title Case）
POLARITY_ORDER = ["Negative", "Implicit Negative", "Neutral", "Implicit Positive", "Positive"]


# ===========================================================================
# 工具函数
# ===========================================================================
def resolve_ft_model(model_root: Path) -> Path:
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


def extract_cls(texts: List[str], model_path: Path, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        X = np.load(cache_path)
        if len(X) == len(texts):
            print(f"  读取缓存: {cache_path}  shape={X.shape}")
            return X
        print("  缓存行数不符，重新提取...")

    device = _device()
    print(f"  提取微调BERT CLS（{len(texts)}首，device={device}）...")
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    mdl = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
    mdl.to(device).eval()

    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LENGTH)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            vecs.append(out.last_hidden_state[:, 0, :].cpu().float().numpy())
            done = min(i + BATCH_SIZE, len(texts))
            if done % 500 == 0 or done == len(texts):
                print(f"    [{done}/{len(texts)}]...")

    X = np.vstack(vecs)
    np.save(cache_path, X)
    print(f"  已缓存 shape={X.shape} -> {cache_path}")
    return X


# ===========================================================================
# 主函数
# ===========================================================================
def main() -> None:
    for p in [FSPC_SPLIT]:
        if not p.exists():
            raise FileNotFoundError(f"缺失: {p}\n请先运行 07_step3_split.py")

    # ── 加载 FSPC 切分 ──────────────────────────────────────────
    df = pd.read_csv(FSPC_SPLIT)
    # 统一标签格式（首字母大写）
    def normalize_label(s: str) -> str:
        return " ".join(w.capitalize() for w in str(s).replace("_", " ").split())

    df["polarity_norm"] = df["polarity"].apply(normalize_label)
    texts = df["text"].tolist()
    print(f"FSPC 总样本: {len(df)}  (train={df['split'].eq('train').sum()}, test={df['split'].eq('test').sum()})")

    # ── 提取微调BERT CLS（FSPC全量）─────────────────────────────
    ft_model_dir = resolve_ft_model(FT_MODEL)
    X_ft = extract_cls(texts, ft_model_dir, FT_FEAT_NPY)

    # ── 切分 train / test ──────────────────────────────────────
    mask_tr = df["split"] == "train"
    mask_te = df["split"] == "test"
    X_tr = X_ft[mask_tr.values];  y_tr = df.loc[mask_tr, "polarity_norm"].tolist()
    X_te = X_ft[mask_te.values];  y_te = df.loc[mask_te, "polarity_norm"].tolist()
    print(f"训练: {len(y_tr)}  测试: {len(y_te)}")

    # ── 训练 SVM ──────────────────────────────────────────────
    print("\n训练 SVM（微调BERT CLS，方法A'）...")
    le = LabelEncoder()
    le.fit(POLARITY_ORDER)
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
    rpt_dict = classification_report(
        y_te_enc, y_pred, target_names=le.classes_,
        digits=4, zero_division=0, output_dict=True,
    )
    rpt_str = classification_report(
        y_te_enc, y_pred, target_names=le.classes_,
        digits=4, zero_division=0,
    )

    result_ap = {
        "method":       "A_prime_SVM_FT_CLS",
        "accuracy":     float(acc),
        "macro_f1":     float(rpt_dict["macro avg"]["f1-score"]),
        "weighted_f1":  float(rpt_dict["weighted avg"]["f1-score"]),
        "per_class": {
            cls: {
                "f1":      float(rpt_dict.get(cls, {}).get("f1-score",  0.0)),
                "precision": float(rpt_dict.get(cls, {}).get("precision", 0.0)),
                "recall":  float(rpt_dict.get(cls, {}).get("recall",    0.0)),
            }
            for cls in le.classes_
        },
    }
    with open(OUT_EVAL / "method_ap_svm_5pol.json", "w", encoding="utf-8") as f:
        json.dump(result_ap, f, ensure_ascii=False, indent=2)

    # 保存模型
    with open(OUT_SVM / "svm_5pol_ftcls.pkl", "wb") as f:
        pickle.dump({"pipe": pipe, "le": le}, f)
    print(f"\n方法A' SVM（微调BERT CLS）")
    print(f"  Accuracy={acc:.4f}  Macro-F1={result_ap['macro_f1']:.4f}")
    print(rpt_str)

    # ── 三方对比报告 ──────────────────────────────────────────
    res_a  = json.loads(PREV_A_JSON.read_text()) if PREV_A_JSON.exists() else None
    res_b  = json.loads(PREV_B_JSON.read_text()) if PREV_B_JSON.exists() else None
    res_ap = result_ap

    rows = []
    if res_a:
        rows.append(("A:  SVM (基础BERT CLS)",  res_a))
    rows.append(    ("A': SVM (微调BERT CLS)",  res_ap))
    if res_b:
        rows.append(("B:  端到端微调BERT ⚠",    res_b))

    lines = [
        "=" * 72,
        "  5极性分类 三方最终对比报告",
        "  ⚠ 方法B 使用全量FSPC训练，fspc_test 非严格盲测（存在数据泄露）",
        "=" * 72,
        f"  {'方法':<30s}  {'Accuracy':>9}  {'Macro-F1':>9}  {'Weighted-F1':>12}",
        "-" * 72,
    ]
    for name, res in rows:
        lines.append(
            f"  {name:<30s}  "
            f"{res.get('accuracy',0):>9.4f}  "
            f"{res.get('macro_f1',0):>9.4f}  "
            f"{res.get('weighted_f1',0):>12.4f}"
        )
    lines += ["-" * 72, ""]

    # 各类 F1
    lines += [
        "  {:<22s}  {:>12}  {:>12}  {:>12}".format("类别", "A: Base SVM", "A': FT SVM", "B: BERT ft"),
        "-" * 72,
    ]
    for cls in POLARITY_ORDER:
        f_a  = res_a["per_class"].get(cls,  {}).get("f1", float("nan")) if res_a  else float("nan")
        f_ap = res_ap["per_class"].get(cls, {}).get("f1", float("nan"))
        f_b  = res_b["per_class"].get(cls,  {}).get("f1", float("nan")) if res_b  else float("nan")
        def fmt(v):
            return f"{v:>12.4f}" if not (isinstance(v, float) and v != v) else f"{'--':>12s}"
        lines.append(f"  {cls:<22s}  {fmt(f_a)}  {fmt(f_ap)}  {fmt(f_b)}")

    lines += ["", "-" * 72]

    # 决策逻辑
    best_svm = max(
        [("A: 基础BERT SVM", res_a["macro_f1"] if res_a else 0),
         ("A': 微调BERT SVM", res_ap["macro_f1"])],
        key=lambda x: x[1],
    )
    bert_f1  = res_b["macro_f1"] if res_b else 0
    svm_best_f1 = best_svm[1]
    delta = bert_f1 - svm_best_f1

    if delta > 0.05:
        verdict = (
            f"【决策】✓ 方法B（端到端微调BERT）更优  Macro-F1 {bert_f1:.4f} vs {svm_best_f1:.4f}\n"
            f"  → 建议5极性层使用微调BERT直接推断（注意：有数据泄露，真实优势可能略低于报告值）"
        )
        chosen = "bert_finetune"
    elif delta > 0:
        verdict = (
            f"【决策】⚠ 方法B略优但差距小（{delta:+.4f}），且存在数据泄露\n"
            f"  → 建议保守选择 {best_svm[0]}（Macro-F1={svm_best_f1:.4f}），数据更干净"
        )
        chosen = "svm_ft_cls" if "A'" in best_svm[0] else "svm_base_cls"
    else:
        verdict = (
            f"【决策】✓ SVM 更优  Macro-F1 {svm_best_f1:.4f} vs {bert_f1:.4f}\n"
            f"  → 建议5极性层使用 {best_svm[0]}"
        )
        chosen = "svm_ft_cls" if "A'" in best_svm[0] else "svm_base_cls"

    lines += [f"  {verdict}", "=" * 72, ""]
    report = "\n".join(lines)
    print("\n" + report)

    out_path = OUT_EVAL / "comparison_5pol_full.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"三方对比报告已保存: {out_path}")

    # ── 更新 final_clf_config.json ──────────────────────────
    cfg_path = OUT_EVAL / "final_clf_config.json"
    cfg = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())

    cfg["layer1_5pol"]["chosen_method"] = chosen
    cfg["layer1_5pol"]["macro_f1"]      = round(max(bert_f1, svm_best_f1), 4)
    cfg["layer1_5pol"]["svm_base_f1"]   = round(res_a["macro_f1"] if res_a else 0, 4)
    cfg["layer1_5pol"]["svm_ft_f1"]     = round(res_ap["macro_f1"], 4)
    cfg["layer1_5pol"]["bert_ft_f1"]    = round(bert_f1, 4)
    cfg["layer1_5pol"]["note"]          = (
        "bert_ft 使用全量FSPC训练，存在数据泄露，报告F1偏高"
        if chosen == "bert_finetune" else ""
    )

    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已更新 final_clf_config.json，5极性层选择: {chosen}")


if __name__ == "__main__":
    main()
