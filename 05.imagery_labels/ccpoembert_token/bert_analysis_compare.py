"""
step2b_bert/bert_analysis_compare.py
─────────────────────────────────────────────────────────────────────────────
任务：
  1. 用 BERT-base-Chinese（Google）对【赏析】提取 embedding + top-10 token
  2. 与已有 SikuRoBERTa 的赏析 embedding/token 做逐首对比

输出（output/）：
  bert_base_chinese_ana_embeddings.npy   shape (N, 768) 赏析 embedding（Google）
  bert_base_chinese_ana_tokens.csv       每首诗赏析 top-10 token（Google）
  analysis_token_compare.csv            两模型 top-10 token 并排对比（200首）
  analysis_token_compare_report.txt     文字版对比报告（含分歧分析）
  output/figures/fig_ana_token_compare.png   典型示例可视化

运行：
    cd /Users/rzhang/Documents/poem-flower-recommend
    source flower_env/bin/activate
    python 02.sample_label_phase2/step2b_bert/bert_analysis_compare.py
─────────────────────────────────────────────────────────────────────────────
"""

import json, os, time, warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

# ─── 路径 ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUT_DIR   = SCRIPT_DIR / "output"
FIG_DIR      = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HF_CACHE = MODELS_DIR / ".hf_cache"
os.environ["HF_HOME"]               = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"]    = str(HF_CACHE / "hub")
os.environ["MPLCONFIGDIR"]          = str(MODELS_DIR / ".matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

SAMPLE_CSV        = PROJECT_ROOT / "01.sample_label" / "output" / "sample_200.csv"
BERT_BASE_ZH_PATH = MODELS_DIR / "bert_base_chinese"
SIKUROBERTA_PATH  = MODELS_DIR / "sikuroberta"

# ─── 设备 ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 MPS (Apple Silicon)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🚀 CUDA")
else:
    DEVICE = torch.device("cpu")
    print("💻 CPU")

# ─── 停用过滤集 ───────────────────────────────────────────────────────────────
SKIP_TOKENS = {
    "[CLS]","[SEP]","[PAD]","[UNK]","[MASK]",
    "，","。","！","？","、","；","：","（","）","(",")",
    "「","」","『","』","·","—","…","《","》"," ","\n",
}


def setup_chinese_font():
    for p in ["/System/Library/Fonts/STHeiti Medium.ttc",
              "/System/Library/Fonts/PingFang.ttc",
              "/System/Library/Fonts/STHeiti Light.ttc"]:
        if Path(p).exists():
            fm.fontManager.addfont(p)
            prop = fm.FontProperties(fname=p)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

setup_chinese_font()


# ─── 核心函数（与 bert_embed.py 保持一致）────────────────────────────────────

def load_model(path: Path, desc: str):
    print(f"  [加载] {desc}")
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    mdl = AutoModel.from_pretrained(str(path), trust_remote_code=True,
                                    output_attentions=True).to(DEVICE)
    mdl.eval()
    hidden = getattr(mdl.config, "hidden_size", 768)
    print(f"         hidden_size={hidden}  device={DEVICE}")
    return tok, mdl


def get_embedding(text: str, tok, mdl) -> np.ndarray:
    hidden_size = getattr(mdl.config, "hidden_size", 768)
    text = str(text).strip()
    if not text:
        return np.zeros(hidden_size, dtype=np.float32)
    inputs = tok(text, return_tensors="pt", max_length=512,
                 truncation=True, padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = mdl(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze().cpu().float().numpy()


def get_token_importance(text: str, tok, mdl, top_k: int = 10) -> list:
    text = str(text).strip()
    if not text:
        return []
    inputs = tok(text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0].cpu().tolist())
    with torch.no_grad():
        out = mdl(**inputs, output_attentions=True)
    attn = (out.attentions[-1].squeeze().mean(0).mean(0).cpu().float().numpy())
    from collections import defaultdict
    merged: dict = defaultdict(float)
    for tok_str, score in zip(tokens, attn):
        if tok_str in SKIP_TOKENS:
            continue
        clean = tok_str[2:] if tok_str.startswith("##") else tok_str
        if not clean.strip():
            continue
        merged[clean] = max(merged[clean], float(score))
    return sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 65)
    print("  赏析 Embedding 对比：BERT-base-Chinese vs SikuRoBERTa")
    print("=" * 65)

    df = pd.read_csv(SAMPLE_CSV)
    n  = len(df)
    print(f"\n样本：{n} 首诗")

    # ── 加载两个模型 ──────────────────────────────────────────────────────────
    print("\n[1/4] 加载模型 ...")
    tok_zh, mdl_zh  = load_model(BERT_BASE_ZH_PATH, "BERT-base-Chinese（Google，赏析）")
    tok_sk, mdl_sk  = load_model(SIKUROBERTA_PATH,  "SikuRoBERTa（四库全书，赏析）")

    # ── 逐首提取 ──────────────────────────────────────────────────────────────
    print(f"\n[2/4] 提取赏析 embedding 和 top-10 token（{n} 首）...")
    embs_zh, embs_sk = [], []
    rows_zh, rows_sk = [], []
    every = max(1, n // 10)

    for i, row in df.iterrows():
        ana = str(row.get("赏析", ""))

        emb_zh = get_embedding(ana, tok_zh, mdl_zh)
        emb_sk = get_embedding(ana, tok_sk, mdl_sk)
        embs_zh.append(emb_zh)
        embs_sk.append(emb_sk)

        tok10_zh = get_token_importance(ana, tok_zh, mdl_zh)
        tok10_sk = get_token_importance(ana, tok_sk, mdl_sk)

        base = {"ID": row.get("ID", i), "诗名": row.get("诗名", ""),
                "花名": row.get("花名", ""), "朝代": row.get("朝代", ""),
                "赏析_preview": ana[:40].replace("\n", " ")}

        rows_zh.append({**base,
                        "top10_tokens": json.dumps(tok10_zh, ensure_ascii=False),
                        "top5_preview": "、".join([t for t, _ in tok10_zh[:5]])})
        rows_sk.append({**base,
                        "top10_tokens": json.dumps(tok10_sk, ensure_ascii=False),
                        "top5_preview": "、".join([t for t, _ in tok10_sk[:5]])})

        if (i + 1) % every == 0 or (i + 1) == n:
            print(f"  {i+1}/{n}  {row.get('诗名','')[:8]}  "
                  f"zh_top3={'、'.join([t for t,_ in tok10_zh[:3]])}  "
                  f"sk_top3={'、'.join([t for t,_ in tok10_sk[:3]])}")

    embs_zh = np.array(embs_zh, dtype=np.float32)
    embs_sk = np.array(embs_sk, dtype=np.float32)
    df_zh   = pd.DataFrame(rows_zh)
    df_sk   = pd.DataFrame(rows_sk)

    # ── 保存 embedding ────────────────────────────────────────────────────────
    np.save(str(OUTPUT_DIR / "bert_base_chinese_ana_embeddings.npy"), embs_zh)
    df_zh.to_csv(str(OUTPUT_DIR / "bert_base_chinese_ana_tokens.csv"),
                 index=False, encoding="utf-8-sig")
    print(f"\n  ✅ BERT-base-Chinese 赏析 embedding: {embs_zh.shape}")

    # ── 生成并排对比 CSV ──────────────────────────────────────────────────────
    print("\n[3/4] 生成 token 对比表 ...")
    compare_rows = []
    for i in range(n):
        r_zh = rows_zh[i]
        r_sk = rows_sk[i]
        zh_toks = [t for t, _ in json.loads(r_zh["top10_tokens"])]
        sk_toks = [t for t, _ in json.loads(r_sk["top10_tokens"])]
        shared  = set(zh_toks) & set(sk_toks)
        agree   = len(shared)

        compare_rows.append({
            "ID":           r_zh["ID"],
            "诗名":         r_zh["诗名"],
            "花名":         r_zh["花名"],
            "朝代":         r_zh["朝代"],
            "赏析_preview": r_zh["赏析_preview"],
            # BERT-base-Chinese
            "zh_top5":      "、".join(zh_toks[:5]),
            "zh_top10":     " ".join(zh_toks),
            # SikuRoBERTa
            "sk_top5":      "、".join(sk_toks[:5]),
            "sk_top10":     " ".join(sk_toks),
            # 一致性
            "共同token数":  agree,
            "共同tokens":   "、".join(shared),
            "一致率":       round(agree / 10, 2),
        })

    df_cmp = pd.DataFrame(compare_rows)
    df_cmp.to_csv(str(OUTPUT_DIR / "analysis_token_compare.csv"),
                  index=False, encoding="utf-8-sig")

    # ── 文字报告 ──────────────────────────────────────────────────────────────
    mean_agree = df_cmp["共同token数"].mean()
    high_agree = (df_cmp["共同token数"] >= 5).sum()
    zero_agree = (df_cmp["共同token数"] == 0).sum()

    # 全局高频词对比
    all_zh = []
    all_sk = []
    for r in compare_rows:
        all_zh.extend(r["zh_top10"].split())
        all_sk.extend(r["sk_top10"].split())
    top20_zh = Counter(all_zh).most_common(20)
    top20_sk = Counter(all_sk).most_common(20)
    only_zh  = set(t for t,_ in top20_zh) - set(t for t,_ in top20_sk)
    only_sk  = set(t for t,_ in top20_sk) - set(t for t,_ in top20_zh)
    both     = set(t for t,_ in top20_zh) & set(t for t,_ in top20_sk)

    lines = [
        "=" * 70,
        "赏析 top-10 token 对比报告",
        "  模型A：BERT-base-Chinese（Google，现代中文预训练）",
        "  模型B：SikuRoBERTa（四库全书古文预训练）",
        "=" * 70,
        "",
        "── 整体一致性统计 ─────────────────────────────────────────",
        f"  平均共同 token 数（满分10）：{mean_agree:.2f}",
        f"  共同 ≥5 个 token 的诗：{high_agree} / {n} 首",
        f"  完全无共同 token 的诗：{zero_agree} / {n} 首",
        "",
        "── 全局 top-20 高频词差异 ─────────────────────────────────",
        f"  两模型共有词（top-20 交集）：{'  '.join(sorted(both))}",
        f"  仅 BERT-base-Chinese 独有：{'  '.join(sorted(only_zh))}",
        f"  仅 SikuRoBERTa 独有：    {'  '.join(sorted(only_sk))}",
        "",
        "── 解读 ────────────────────────────────────────────────────",
        "  ● 共有词 → 两模型都认为重要：赏析中的核心概念词",
        "  ● 仅BERT-base独有 → 现代汉语用词、口语化表达、数量词",
        "  ● 仅SikuRoBERTa独有 → 古典词汇、文言虚词、诗歌意象词",
        "",
        "  两模型对赏析的'关注点'不同：",
        "    BERT-base-Chinese 倾向关注现代汉语的话题词和主题词",
        "    SikuRoBERTa 倾向关注古典词汇和文言表达",
        "  → 建议：两路赏析 embedding 加权合并，互补信息更丰富",
        "",
        "── 逐首典型示例（共同最少 vs 最多）─────────────────────",
        "",
    ]

    # 挑分歧最大的5首（共同token最少，排除nan）
    non_nan = df_cmp[df_cmp["赏析_preview"].str.len() > 5].copy()
    bottom5 = non_nan.nsmallest(5, "共同token数")
    top5    = non_nan.nlargest(5, "共同token数")

    def poem_block(row, label):
        return [
            f"  [{label}] 【{row['花名']}】{row['诗名']}（{row['朝代']}）",
            f"    赏析片段：{row['赏析_preview']}...",
            f"    BERT-zh  top-5：{row['zh_top5']}",
            f"    SikuBERT top-5：{row['sk_top5']}",
            f"    共同 {row['共同token数']} 个：{row['共同tokens'] or '（无）'}",
            "",
        ]

    lines.append("  ▼ 分歧最大（最少共同 token）：")
    for _, row in bottom5.iterrows():
        lines += poem_block(row, "分歧")
    lines.append("  ▼ 共识最强（最多共同 token）：")
    for _, row in top5.iterrows():
        lines += poem_block(row, "共识")

    (OUTPUT_DIR / "analysis_token_compare_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"  ✅ 对比报告已保存")

    # ── 可视化 ────────────────────────────────────────────────────────────────
    print("\n[4/4] 生成可视化图 ...")
    _plot_compare(df_cmp, compare_rows, top20_zh, top20_sk)

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print(f"✅ 完成！总用时 {elapsed:.1f}s")
    print(f"  平均共同 token：{mean_agree:.2f}/10   共识≥5：{high_agree}首   完全分歧：{zero_agree}首")
    outputs = [
        "bert_base_chinese_ana_embeddings.npy",
        "bert_base_chinese_ana_tokens.csv",
        "analysis_token_compare.csv",
        "analysis_token_compare_report.txt",
        "figures/fig_ana_token_compare.png",
    ]
    for o in outputs:
        p = OUTPUT_DIR / o
        if p.exists():
            print(f"  - {o}  ({p.stat().st_size/1024:.0f} KB)")
    print("=" * 65)


def _plot_compare(df_cmp, compare_rows, top20_zh, top20_sk):
    """生成 4 格对比图"""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("赏析 top-10 token 对比：BERT-base-Chinese vs SikuRoBERTa",
                 fontsize=14, y=0.98)

    # ── 子图1：一致率分布直方图 ──────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    agree_counts = df_cmp["共同token数"].values
    bins = range(0, 12)
    ax1.hist(agree_counts, bins=bins, color="#4472C4", edgecolor="white", alpha=0.85)
    ax1.axvline(agree_counts.mean(), color="red", lw=2,
                label=f"均值 {agree_counts.mean():.2f}")
    ax1.set_xlabel("两模型共同 top-10 token 数量")
    ax1.set_ylabel("诗首数")
    ax1.set_title("两模型 top-10 token 一致性分布\n（0=完全不同，10=完全相同）")
    ax1.set_xticks(range(0, 11))
    ax1.legend()

    # ── 子图2：全局高频词对比横条图（并排）────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    # 取各自 top-15
    top15_zh = top20_zh[:15]
    top15_sk = top20_sk[:15]
    # 合并词表
    all_words = list({t for t, _ in top15_zh} | {t for t, _ in top15_sk})
    zh_dict = dict(top20_zh)
    sk_dict = dict(top20_sk)
    # 按两者之和排序
    all_words.sort(key=lambda w: zh_dict.get(w, 0) + sk_dict.get(w, 0), reverse=True)
    all_words = all_words[:18]

    x = range(len(all_words))
    w = 0.4
    bars_zh = [zh_dict.get(word, 0) for word in all_words]
    bars_sk = [sk_dict.get(word, 0) for word in all_words]
    ax2.bar([i - w/2 for i in x], bars_zh, width=w, label="BERT-base-Chinese",
            color="#4472C4", alpha=0.85)
    ax2.bar([i + w/2 for i in x], bars_sk, width=w, label="SikuRoBERTa",
            color="#ED7D31", alpha=0.85)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(all_words, fontsize=9)
    ax2.set_ylabel("出现次数（200首合计）")
    ax2.set_title("全局高频 token 对比\n（蓝=BERT-zh，橙=SikuRoBERTa）")
    ax2.legend(fontsize=9)

    # ── 子图3：6首典型诗的并排 token 展示 ────────────────────────────────────
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.axis("off")

    # 挑选：3首高共识 + 3首高分歧
    non_nan = df_cmp[df_cmp["赏析_preview"].str.len() > 5]
    sel_high = non_nan.nlargest(3, "共同token数")[
        ["诗名", "花名", "朝代", "zh_top5", "sk_top5", "共同token数", "共同tokens"]
    ]
    sel_low  = non_nan.nsmallest(3, "共同token数")[
        ["诗名", "花名", "朝代", "zh_top5", "sk_top5", "共同token数", "共同tokens"]
    ]
    sel = pd.concat([sel_high, sel_low]).reset_index(drop=True)

    col_labels = ["诗名/花名", "朝代", "BERT-zh top-5", "Siku top-5", "共同数", "共同词"]
    cell_data  = []
    for _, r in sel.iterrows():
        cell_data.append([
            f"{r['诗名'][:8]}\n({r['花名']})",
            r["朝代"],
            r["zh_top5"],
            r["sk_top5"],
            str(r["共同token数"]),
            r["共同tokens"] or "—",
        ])

    row_colors = [["#DDEEFF"] * 6] * 3 + [["#FFEEEE"] * 6] * 3

    tbl = ax3.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.8)
    ax3.set_title("典型示例（蓝底=高共识，红底=高分歧）",
                  fontsize=10, pad=8, y=0.92)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = FIG_DIR / "fig_ana_token_compare.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存：{out_path.name}")


if __name__ == "__main__":
    main()
