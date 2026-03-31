"""
07_step6_final_clf.py
=====================
最终分层情感分类器（三层级联）

架构：
  Layer 1  5极性      端到端微调BERT (ccpoem_sentiment_ft)
                      输出: Negative / Implicit Negative / Neutral /
                            Implicit Positive / Positive
  Layer 2  C2粗粒度   SVM（微调BERT CLS + FCCPSL 15维词典分数）
                      输出: positive / neg_sorrow / neg_anger
                      仅对 L1 != Neutral 的诗歌启用
  Layer 3  C3细粒度   SVM（微调BERT CLS + FCCPSL 15维词典分数）
                      输出: praise / like / faith / joy / ease / wish /
                            sorrow / guilt / miss / fear /
                            criticize / anger / vexed / misgive
                      仅对 L1 != Neutral 的诗歌启用

一致性约束：
  L1 Positive/Implicit Positive  →  L2 必须为 positive
  L1 Negative/Implicit Negative  →  L2 必须为 neg_sorrow 或 neg_anger
  L1 Neutral                     →  L2/L3 置空（标注 n/a）

  L2 positive    →  L3 必须在 {praise,like,faith,joy,ease,wish} 内
  L2 neg_sorrow  →  L3 必须在 {sorrow,guilt,miss,fear} 内
  L2 neg_anger   →  L3 必须在 {criticize,anger,vexed,misgive} 内
  若 SVM 预测的 L3 与 L2 约束冲突，则取对应约束集合内概率最高的类别。

用法：
  # 批量推断（默认 00.poems_dataset/poems_dataset_merged_done.csv）
  python 07_step6_final_clf.py

  # 单首推断
  python 07_step6_final_clf.py --text "春风又绿江南岸"

  # 自定义输入 CSV
  python 07_step6_final_clf.py --input /path/to/poems.csv --text_col 正文

输出：
  output/results/sentiment_final_predictions.csv   完整推断结果
  output/results/sentiment_final_summary.txt        标签分布汇总
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# ===========================================================================
# 路径配置
# ===========================================================================
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

FT_MODEL_DIR  = _PROJECT_ROOT / "models" / "ccpoem_sentiment_ft"
FCCPSL_PATH   = _SCRIPT_DIR / "output" / "lexicon" / "fccpsl_terms_only.csv"
SVM_C2_PKL    = _SCRIPT_DIR / "output" / "svm_models" / "svm_c2_ablation_best.pkl"
SVM_C3_PKL    = _SCRIPT_DIR / "output" / "svm_models" / "svm_c3_ablation_best.pkl"
# 项目诗歌全量表（更新后仍以本路径为准）
PROJECT_POEMS = _PROJECT_ROOT / "00.poems_dataset" / "poems_dataset_merged_done.csv"
OUT_RESULTS   = _SCRIPT_DIR / "output" / "results"
OUT_RESULTS.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
MAX_LENGTH = 128

# ── 词典匹配参数（与 07_step1_pseudo_labeling.py 完全对齐）───────────────
MIN_WORD_LEN = 2
MAX_WORD_LEN = 2
SINGLE_CHAR_WHITELIST = {"愁", "悲", "哀", "泪", "怨", "苦", "痛", "喜", "乐", "欢", "悦", "思", "念", "忆"}
NEGATION_CHARS = {"不", "无", "莫", "非", "未", "休", "勿"}

# ── C3 → C2 映射（方案C，3类合并正向）────────────────────────────────────
C3_TO_C2: Dict[str, str] = {
    "praise": "positive", "like":    "positive", "faith":  "positive",
    "joy":    "positive", "ease":    "positive", "wish":   "positive", "peculiar": "positive",
    "sorrow": "neg_sorrow", "guilt": "neg_sorrow", "miss":   "neg_sorrow", "fear":  "neg_sorrow",
    "vexed":  "neg_anger",  "criticize": "neg_anger", "anger": "neg_anger", "misgive": "neg_anger",
}

# ── C2 → 允许的 C3 集合 ────────────────────────────────────────────────────
C2_TO_C3_ALLOWED: Dict[str, set] = {
    "positive":   {"praise", "like", "faith", "joy", "ease", "wish", "peculiar"},
    "neg_sorrow": {"sorrow", "guilt", "miss", "fear"},
    "neg_anger":  {"criticize", "anger", "vexed", "misgive"},
}

# ── L1 极性 → 允许的 C2 集合 ─────────────────────────────────────────────
POL_TO_C2_ALLOWED: Dict[str, set] = {
    "Positive":          {"positive"},
    "Implicit Positive": {"positive"},
    "Neutral":           set(),            # 置空
    "Implicit Negative": {"neg_sorrow", "neg_anger"},
    "Negative":          {"neg_sorrow", "neg_anger"},
}

C2_ZH = {"positive": "积极", "neg_sorrow": "负向-悲伤", "neg_anger": "负向-愤懑"}
C3_ZH_MAP: Dict[str, str] = {}   # 从 FCCPSL CSV 填充
POL_ZH = {
    "Negative": "消极", "Implicit Negative": "隐性消极",
    "Neutral": "中性", "Implicit Positive": "隐性积极", "Positive": "积极",
}


# ===========================================================================
# 设备
# ===========================================================================
def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ===========================================================================
# 词典加载（与 step1 对齐）
# ===========================================================================
def load_fccpsl(path: Path) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    global C3_ZH_MAP
    df = pd.read_csv(path)
    C3_ZH_MAP = dict(zip(df["C3"], df["C3_zh"]))

    vocab_by_c3: Dict[str, List[str]] = {}
    for c3, grp in df.groupby("C3"):
        words = []
        for _, row in grp.iterrows():
            w = str(row["词"]).strip()
            wlen = len(w)
            in_whitelist = (wlen == 1 and w in SINGLE_CHAR_WHITELIST)
            in_range = (wlen >= MIN_WORD_LEN and
                        (MAX_WORD_LEN is None or wlen <= MAX_WORD_LEN))
            if in_range or in_whitelist:
                words.append(w)
        words = sorted(set(words), key=lambda x: -len(x))
        vocab_by_c3[c3] = words

    idf_weight = {
        c3: 1.0 / math.log2(len(words) + 1) if words else 0.0
        for c3, words in vocab_by_c3.items()
    }
    return vocab_by_c3, idf_weight


# ===========================================================================
# 词典得分计算
# ===========================================================================
def _count_matches(text: str, term: str) -> int:
    cnt, start = 0, 0
    tlen = len(term)
    while True:
        idx = text.find(term, start)
        if idx == -1:
            break
        if not (idx > 0 and text[idx - 1] in NEGATION_CHARS):
            cnt += 1
        start = idx + 1
    return cnt


def _clean(text: str) -> str:
    return "".join(c for c in text if "\u4e00" <= c <= "\u9fff")


def compute_lex_scores(
    texts: List[str],
    vocab_by_c3: Dict[str, List[str]],
    idf_weight: Dict[str, float],
    c3_order: List[str],
) -> np.ndarray:
    """返回 (N, 15) 的 IDF 加权词典得分矩阵，列顺序 = c3_order。"""
    N = len(texts)
    K = len(c3_order)
    scores = np.zeros((N, K), dtype=np.float32)
    c3_idx = {c3: i for i, c3 in enumerate(c3_order)}

    for n, raw in enumerate(texts):
        text = _clean(raw)
        for c3, words in vocab_by_c3.items():
            if c3 not in c3_idx:
                continue
            col = c3_idx[c3]
            w = idf_weight.get(c3, 0.0)
            for word in words:
                cnt = _count_matches(text, word)
                if cnt > 0:
                    scores[n, col] += cnt * w
    return scores


# ===========================================================================
# BERT 特征提取（共用单个模型实例）
# ===========================================================================
class BertFeatureExtractor:
    """加载微调BERT，同时支持分类推断和 CLS 特征提取。"""

    def __init__(self, model_dir: Path, device: str):
        print(f"加载微调BERT: {model_dir}  (device={device})")
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

        # 分类模型（用于 L1）
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), trust_remote_code=True
        ).to(device).eval()

        # 读 id2label
        import json
        cfg_path = model_dir / "config.json"
        cfg = json.loads(cfg_path.read_text())
        raw_id2label: Dict[str, str] = cfg.get("id2label", {})
        # 统一转为 Title Case
        def _title(s: str) -> str:
            return " ".join(w.capitalize() for w in s.replace("_", " ").split())
        self.id2label = {int(k): _title(v) for k, v in raw_id2label.items()}

        # 骨干模型（用于 CLS 特征 → L2/L3）
        self.base_model = AutoModel.from_pretrained(
            str(model_dir), trust_remote_code=True
        ).to(device).eval()

        print(f"  id2label: {self.id2label}")

    def predict_5pol(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        """返回 (预测标签列表, 概率矩阵 N×5)。"""
        all_labels, all_probs = [], []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i: i + BATCH_SIZE]
                enc = self.tok(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=MAX_LENGTH)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.clf_model(**enc).logits
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
                preds  = probs.argmax(axis=1)
                all_probs.append(probs)
                all_labels.extend([self.id2label[p] for p in preds])
        return all_labels, np.vstack(all_probs)

    def extract_cls(self, texts: List[str]) -> np.ndarray:
        """返回 (N, 512) CLS 向量矩阵。"""
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i: i + BATCH_SIZE]
                enc = self.tok(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=MAX_LENGTH)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.base_model(**enc)
                vecs.append(out.last_hidden_state[:, 0, :].cpu().float().numpy())
        return np.vstack(vecs)


# ===========================================================================
# SVM 封装
# ===========================================================================
class SVMClassifier:
    def __init__(self, pkl_path: Path, name: str):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        self.pipe     = obj["pipe"]
        self.le       = obj["le"]
        self.name     = name
        self.classes  = list(self.le.classes_)
        print(f"加载 {name}: {pkl_path.name}  类别={self.classes}")

    def predict_proba(self, X: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """返回 (标签列表, 概率矩阵 N×K)。"""
        probs = self.pipe.predict_proba(X)
        preds = [self.classes[i] for i in probs.argmax(axis=1)]
        return preds, probs

    def predict_constrained(
        self,
        X: np.ndarray,
        allowed_per_sample: List[Optional[set]],
    ) -> List[str]:
        """
        带约束预测：对每个样本仅在 allowed_per_sample[i] 内选最高概率类别。
        若 allowed 为 None 或为空集，则返回 'n/a'。
        """
        _, probs = self.predict_proba(X)
        results = []
        for i, allowed in enumerate(allowed_per_sample):
            if not allowed:
                results.append("n/a")
                continue
            best_label, best_prob = "n/a", -1.0
            for cls in allowed:
                if cls in self.classes:
                    j = self.classes.index(cls)
                    if probs[i, j] > best_prob:
                        best_prob  = probs[i, j]
                        best_label = cls
            results.append(best_label)
        return results


# ===========================================================================
# 分层分类器
# ===========================================================================
class HierarchicalSentimentClassifier:
    """三层情感分类器：5极性 → C2粗粒度 → C3细粒度。"""

    def __init__(self):
        for p in [FT_MODEL_DIR, FCCPSL_PATH, SVM_C2_PKL, SVM_C3_PKL]:
            if not Path(p).exists():
                raise FileNotFoundError(f"缺失: {p}")

        device = _device()
        self.bert    = BertFeatureExtractor(FT_MODEL_DIR, device)
        self.svm_c2  = SVMClassifier(SVM_C2_PKL, "SVM-C2")
        self.svm_c3  = SVMClassifier(SVM_C3_PKL, "SVM-C3")

        print("加载 FCCPSL 词典...")
        self.vocab_by_c3, self.idf_weight = load_fccpsl(FCCPSL_PATH)
        # 保证特征列顺序与训练时完全一致（训练时用 sorted(score_cols)）
        self.c3_order = sorted(self.vocab_by_c3.keys())
        print(f"  C3 顺序 ({len(self.c3_order)}): {self.c3_order}")

    # ── 核心推断 ───────────────────────────────────────────────────────────
    def predict(self, texts: List[str]) -> pd.DataFrame:
        N = len(texts)
        print(f"\n推断 {N} 首诗词...")

        # ── Step A: L1 5极性 ──────────────────────────────────────────────
        print("  [L1] 5极性推断...")
        l1_labels, l1_probs = self.bert.predict_5pol(texts)

        # ── Step B: 提取 CLS + 词典分数（L2/L3 共用）────────────────────
        print("  [feat] 提取 CLS 向量...")
        X_cls = self.bert.extract_cls(texts)

        print("  [feat] 计算词典分数...")
        X_lex = compute_lex_scores(texts, self.vocab_by_c3, self.idf_weight, self.c3_order)

        X_feat = np.hstack([X_cls, X_lex])   # (N, 527)

        # ── Step C: L2 C2（带极性约束）───────────────────────────────────
        print("  [L2] C2 推断（含一致性约束）...")
        c2_allowed = [POL_TO_C2_ALLOWED.get(lbl, set()) for lbl in l1_labels]
        l2_labels  = self.svm_c2.predict_constrained(X_feat, c2_allowed)

        # ── Step D: L3 C3（带 C2 约束）───────────────────────────────────
        print("  [L3] C3 推断（含 C2 约束）...")
        c3_allowed = [
            C2_TO_C3_ALLOWED.get(c2, set()) if c2 != "n/a" else set()
            for c2 in l2_labels
        ]
        # 只预测 L1 非 Neutral 的样本（batch 剪枝以节省时间）
        l3_labels = ["n/a"] * N
        active_idx = [i for i, lbl in enumerate(l1_labels) if lbl != "Neutral"]
        if active_idx:
            X_active  = X_feat[active_idx]
            c3_active = [c3_allowed[i] for i in active_idx]
            preds_active = self.svm_c3.predict_constrained(X_active, c3_active)
            for i, pred in zip(active_idx, preds_active):
                l3_labels[i] = pred

        # ── Step E: 组装结果 ──────────────────────────────────────────────
        # L1 概率列（5列）
        pol_order = ["Negative", "Implicit Negative", "Neutral", "Implicit Positive", "Positive"]
        bert_id2idx = {lbl: i for i, lbl in self.bert.id2label.items()}
        l1_prob_cols = {}
        for pol in pol_order:
            col_key = f"prob_l1_{pol.lower().replace(' ', '_')}"
            if pol in bert_id2idx:
                l1_prob_cols[col_key] = l1_probs[:, bert_id2idx[pol]]
            else:
                l1_prob_cols[col_key] = np.zeros(N)

        df_out = pd.DataFrame({
            "text":           texts,
            "l1_polarity":    l1_labels,
            "l1_polarity_zh": [POL_ZH.get(lbl, lbl) for lbl in l1_labels],
            "l2_c2":          l2_labels,
            "l2_c2_zh":       [C2_ZH.get(lbl, lbl) if lbl != "n/a" else "n/a" for lbl in l2_labels],
            "l3_c3":          l3_labels,
            "l3_c3_zh":       [C3_ZH_MAP.get(lbl, lbl) if lbl != "n/a" else "n/a" for lbl in l3_labels],
            **l1_prob_cols,
        })
        return df_out

    # ── 单首推断（命令行调用）──────────────────────────────────────────────
    def predict_one(self, text: str) -> dict:
        df = self.predict([text])
        row = df.iloc[0].to_dict()
        return row


# ===========================================================================
# 汇总报告
# ===========================================================================
def make_summary(df: pd.DataFrame) -> str:
    lines = ["=" * 60, "  情感分类结果分布汇总", "=" * 60]

    # L1
    lines += ["", "  [L1] 5极性分布:"]
    for lbl, cnt in df["l1_polarity"].value_counts().items():
        zh = POL_ZH.get(lbl, lbl)
        lines.append(f"    {lbl:<22s} ({zh})  {cnt:>4d} ({cnt/len(df)*100:.1f}%)")

    # L2
    lines += ["", "  [L2] C2粗粒度分布:"]
    for lbl, cnt in df["l2_c2"].value_counts().items():
        zh = C2_ZH.get(lbl, lbl)
        lines.append(f"    {lbl:<16s} ({zh})  {cnt:>4d} ({cnt/len(df)*100:.1f}%)")

    # L3
    lines += ["", "  [L3] C3细粒度分布:"]
    for lbl, cnt in df["l3_c3"].value_counts().items():
        zh = C3_ZH_MAP.get(lbl, lbl)
        lines.append(f"    {lbl:<16s} ({zh})  {cnt:>4d} ({cnt/len(df)*100:.1f}%)")

    # L2 × L3 交叉（仅非 n/a）
    lines += ["", "  [L2×L3] 粗细粒度交叉（Top 15）:"]
    cross = df[df["l3_c3"] != "n/a"].groupby(["l2_c2", "l3_c3"]).size().sort_values(ascending=False)
    for (c2, c3), cnt in cross.head(15).items():
        lines.append(f"    {c2:<14s} → {c3:<14s}  {cnt:>4d}")

    lines += ["", "=" * 60]
    return "\n".join(lines)


# ===========================================================================
# 主入口
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="分层情感分类器")
    parser.add_argument("--text",     type=str, default=None, help="单首诗词正文")
    parser.add_argument("--input",    type=str, default=None, help="自定义输入 CSV")
    parser.add_argument("--text_col", type=str, default="正文", help="CSV 中的正文列名")
    args = parser.parse_args()

    clf = HierarchicalSentimentClassifier()

    # ── 单首模式 ──────────────────────────────────────────────────────────
    if args.text:
        result = clf.predict_one(args.text)
        print("\n" + "=" * 50)
        print(f"  诗词: {args.text[:40]}...")
        print(f"  L1 5极性 : {result['l1_polarity']} ({result['l1_polarity_zh']})")
        print(f"  L2 C2粗类: {result['l2_c2']} ({result['l2_c2_zh']})")
        print(f"  L3 C3细类: {result['l3_c3']} ({result['l3_c3_zh']})")
        print("=" * 50)
        return

    # ── 批量模式 ──────────────────────────────────────────────────────────
    if args.input:
        src_path = Path(args.input)
        if not src_path.exists():
            print(f"ERROR: 文件不存在: {src_path}", file=sys.stderr)
            sys.exit(1)
        df_src = pd.read_csv(src_path)
        if args.text_col not in df_src.columns:
            print(f"ERROR: 列 '{args.text_col}' 不在 {src_path.name} 中", file=sys.stderr)
            sys.exit(1)
        texts  = df_src[args.text_col].fillna("").tolist()
        meta   = df_src
        id_col = "ID" if "ID" in df_src.columns else None
    else:
        # 默认：项目全量 merged_done（当前仓库 1075 行；v5 精简版 999 行请 --input）
        if not PROJECT_POEMS.exists():
            print(f"ERROR: 找不到项目诗词文件: {PROJECT_POEMS}", file=sys.stderr)
            sys.exit(1)
        df_src = pd.read_csv(PROJECT_POEMS)
        texts  = df_src["正文"].fillna("").tolist()
        meta   = df_src
        id_col = "ID"

    # 清洗：去除全空文本
    valid_mask = [bool(t.strip()) for t in texts]
    texts_clean = [t for t, v in zip(texts, valid_mask) if v]
    print(f"有效样本: {len(texts_clean)} / {len(texts)}")

    df_pred = clf.predict(texts_clean)

    # 合并元数据
    meta_clean = meta[valid_mask].reset_index(drop=True)
    key_cols = [c for c in ["ID", "诗名", "作者", "朝代", "花名", "月份"] if c in meta_clean.columns]
    df_out = pd.concat([meta_clean[key_cols].reset_index(drop=True), df_pred], axis=1)

    # 保存
    out_csv = OUT_RESULTS / "sentiment_final_predictions.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n推断结果已保存: {out_csv}  ({len(df_out)} 行)")

    # 汇总报告
    summary = make_summary(df_pred)
    print("\n" + summary)
    summary_path = OUT_RESULTS / "sentiment_final_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"汇总报告已保存: {summary_path}")


if __name__ == "__main__":
    main()
