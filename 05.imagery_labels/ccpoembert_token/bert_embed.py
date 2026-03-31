"""
step2b_bert/bert_embed.py
─────────────────────────────────────────────────────────────────────────────
正文：BERT-CCPoem 系列（synpjh/BERT_CCPoem_v1-finetuned-poem 或
      ethanyt/guwenbert-base 等本地保存到 models/bert_ccpoem/ 的模型）
赏析：SikuRoBERTa（SIKU-BERT/sikuroberta，本地保存到 models/sikuroberta/）

输出文件（保存到 output/）：
  bert_ccpoem_embeddings.npy      shape (N, D_poem)  正文 embedding
  bert_analysis_embeddings.npy    shape (N, D_ana)   赏析 embedding
  bert_similarity_top10.csv       每首诗语义最近的前10首（正文+赏析各一套）
  bert_token_importance.csv       每首诗 top-10 重要 token（正文）
  bert_embed_meta.json            记录模型名称、维度、设备等元信息

注意：embedding 维度取决于所用模型，
  - synpjh/BERT_CCPoem_v1：hidden_size=512
  - ethanyt/guwenbert-base：hidden_size=768
  - SIKU-BERT/sikuroberta：hidden_size=768
代码自动检测，不硬编码维度。

运行方式：
    cd /Users/rzhang/Documents/poem-flower-recommend
    source flower_env/bin/activate
    python 02.sample_label_phase2/step2b_bert/bert_embed.py

GPU 加速：Mac Mini M4 会自动检测 MPS；无 MPS 则回退 CPU。
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

# ─── 路径配置 ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUT_DIR   = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 将 HuggingFace 缓存重定向到项目内，保持环境一致
HF_CACHE = MODELS_DIR / ".hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"]               = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"]    = str(HF_CACHE / "hub")

# 输入数据：沿用 01.sample_label 已生成的 200 条样本
SAMPLE_CSV = PROJECT_ROOT / "01.sample_label" / "output" / "sample_200.csv"

# 模型本地路径
BERT_CCPOEM_PATH  = MODELS_DIR / "bert_ccpoem"
SIKUROBERTA_PATH  = MODELS_DIR / "sikuroberta"

# ─── 设备选择（MPS → CPU） ───────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 使用 MPS（Apple Silicon GPU）加速")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🚀 使用 CUDA GPU 加速")
else:
    DEVICE = torch.device("cpu")
    print("💻 使用 CPU 推理（无 GPU 加速）")

# ─── 标点 & 特殊 token 过滤集合 ─────────────────────────────────────────────
SKIP_TOKENS = {
    "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]",
    "，", "。", "！", "？", "、", "；", "：",
    "（", "）", "(", ")", "「", "」", "『", "』",
    "·", "—", "…", "《", "》", " ", "\n",
    "##",    # BERT wordpiece 前缀单独出现时过滤
}


# ─── 核心函数 ─────────────────────────────────────────────────────────────────

def load_model(model_path: Path, desc: str):
    """加载 tokenizer 和 model 到指定设备"""
    print(f"\n[加载] {desc} ← {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"模型目录不存在：{model_path}\n"
            "请先运行 download_models.py 下载模型。"
        )
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    mdl = AutoModel.from_pretrained(
        str(model_path), trust_remote_code=True, output_attentions=True
    ).to(DEVICE)
    mdl.eval()
    print(f"  ✅ {desc} 加载完成（device={DEVICE}）")
    return tok, mdl


def get_embedding(text: str, tok, mdl) -> np.ndarray:
    """
    取 [CLS] token 最后一层 hidden state 作为句子 embedding。
    返回 shape (hidden_size,) 的 float32 数组；空文本返回零向量。
    hidden_size 自动从模型 config 读取，不硬编码 768。
    """
    hidden_size = getattr(mdl.config, "hidden_size", 768)
    text = str(text).strip()
    if not text:
        return np.zeros(hidden_size, dtype=np.float32)

    inputs = tok(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )
    # 移动到目标设备
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = mdl(**inputs)

    # [CLS] 位置（index 0）的最后层 hidden state → CPU → numpy
    emb = out.last_hidden_state[:, 0, :].squeeze().cpu().float().numpy()
    return emb


def get_token_importance(text: str, tok, mdl, top_k: int = 10) -> list:
    """
    用最后一层所有注意力头的平均权重作为 token 重要性分数。
    返回 top_k 个 (token_str, score) 对，排除特殊符号和标点。
    """
    text = str(text).strip()
    if not text:
        return []

    inputs = tok(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0].cpu().tolist())

    with torch.no_grad():
        out = mdl(**inputs, output_attentions=True)

    # attentions[-1]: (1, num_heads, seq_len, seq_len)
    # 对所有头和所有来源取均值 → 每个 token 的重要性分数
    attn = (
        out.attentions[-1]
        .squeeze()          # (num_heads, seq_len, seq_len)
        .mean(dim=0)        # (seq_len, seq_len)  行=来源
        .mean(dim=0)        # (seq_len,)           被关注程度
        .cpu()
        .float()
        .numpy()
    )

    scored = []
    for tok_str, score in zip(tokens, attn):
        # 过滤：特殊 token、标点、空格、wordpiece ##开头子词
        if tok_str in SKIP_TOKENS:
            continue
        if tok_str.startswith("##"):
            tok_str = tok_str[2:]   # 去掉前缀，保留子词内容
        if not tok_str.strip():
            continue
        scored.append((tok_str, float(score)))

    # 同一个词可能被 wordpiece 拆分，合并同名得分（取最大值）
    from collections import defaultdict
    merged: dict = defaultdict(float)
    for tok_str, score in scored:
        merged[tok_str] = max(merged[tok_str], score)

    sorted_tokens = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return sorted_tokens[:top_k]


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """计算 (N, N) 余弦相似度矩阵"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    normed = embeddings / norms
    return normed @ normed.T


def get_top10_similar(sim_matrix: np.ndarray, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    对每首诗找语义最近的前10首（排除自身）。
    返回 DataFrame，列：poem_id, 诗名, 花名, top10_ids, top10_names, top10_scores
    """
    rows = []
    n = len(df)
    for i in range(n):
        scores = sim_matrix[i].copy()
        scores[i] = -1  # 排除自身
        top_idx = np.argsort(scores)[::-1][:10]
        top_scores = scores[top_idx]

        row_data = df.iloc[i]
        rows.append({
            f"{prefix}_poem_id":     row_data.get("ID", i),
            "诗名":                  row_data.get("诗名", ""),
            "花名":                  row_data.get("花名", ""),
            "朝代":                  row_data.get("朝代", ""),
            f"{prefix}_top10_ids":   json.dumps(
                [int(df.iloc[j].get("ID", j)) for j in top_idx],
                ensure_ascii=False,
            ),
            f"{prefix}_top10_names": json.dumps(
                [str(df.iloc[j].get("诗名", "")) for j in top_idx],
                ensure_ascii=False,
            ),
            f"{prefix}_top10_scores": json.dumps(
                [round(float(s), 4) for s in top_scores],
                ensure_ascii=False,
            ),
        })
    return pd.DataFrame(rows)


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 65)
    print("  step2b_bert：BERT 特征提取")
    print(f"  正文模型：BERT-CCPoem  ({BERT_CCPOEM_PATH.name})")
    print(f"  赏析模型：SikuRoBERTa  ({SIKUROBERTA_PATH.name})")
    print("=" * 65)

    # 1. 读取数据 ──────────────────────────────────────────────────────────────
    print(f"\n[1/5] 读取样本数据：{SAMPLE_CSV}")
    if not SAMPLE_CSV.exists():
        raise FileNotFoundError(f"样本文件不存在：{SAMPLE_CSV}")
    df = pd.read_csv(SAMPLE_CSV)
    n = len(df)
    print(f"  共 {n} 首诗")
    print(f"  列名：{list(df.columns)}")

    # 2. 加载模型 ──────────────────────────────────────────────────────────────
    print("\n[2/5] 加载模型 ...")
    tok_poem, mdl_poem = load_model(BERT_CCPOEM_PATH,  "BERT-CCPoem（正文）")
    tok_ana,  mdl_ana  = load_model(SIKUROBERTA_PATH,  "SikuRoBERTa（赏析）")

    # 3. 逐条提取 embedding 与 token 重要性 ───────────────────────────────────
    print(f"\n[3/5] 提取 embedding（{n} 首诗）...")
    poem_embs   = []
    ana_embs    = []
    token_rows  = []

    report_every = max(1, n // 10)

    for i, row in df.iterrows():
        # —— 正文 embedding（BERT-CCPoem）
        poem_text = str(row.get("正文", ""))
        poem_emb  = get_embedding(poem_text, tok_poem, mdl_poem)
        poem_embs.append(poem_emb)

        # —— 赏析 embedding（SikuRoBERTa）
        ana_text = str(row.get("赏析", ""))
        ana_emb  = get_embedding(ana_text, tok_ana, mdl_ana)
        ana_embs.append(ana_emb)

        # —— 正文 token 重要性
        imp = get_token_importance(poem_text, tok_poem, mdl_poem, top_k=10)
        token_rows.append({
            "ID":           row.get("ID", i),
            "诗名":         row.get("诗名", ""),
            "花名":         row.get("花名", ""),
            "朝代":         row.get("朝代", ""),
            "月份":         row.get("月份", ""),
            "正文_preview": poem_text[:30],
            "top10_tokens": json.dumps(imp, ensure_ascii=False),
            "top5_preview": "、".join([t for t, _ in imp[:5]]),
        })

        if (i + 1) % report_every == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            print(f"  进度：{i+1}/{n}  已用时 {elapsed:.1f}s  "
                  f"（{row.get('诗名','')[:8]}）")

    poem_embs = np.array(poem_embs, dtype=np.float32)  # (N, 768)
    ana_embs  = np.array(ana_embs,  dtype=np.float32)  # (N, 768)

    print(f"\n  正文 embedding shape: {poem_embs.shape}")
    print(f"  赏析 embedding shape: {ana_embs.shape}")

    # 4. 计算语义相似度 Top-10 ─────────────────────────────────────────────────
    print("\n[4/5] 计算语义相似度 Top-10 ...")
    sim_poem = compute_similarity_matrix(poem_embs)
    sim_ana  = compute_similarity_matrix(ana_embs)

    df_sim_poem = get_top10_similar(sim_poem, df, prefix="poem")
    df_sim_ana  = get_top10_similar(sim_ana,  df, prefix="ana")

    # 合并两套相似度到同一个 CSV
    df_sim = pd.concat(
        [df_sim_poem.reset_index(drop=True),
         df_sim_ana[["ana_top10_ids", "ana_top10_names", "ana_top10_scores"]].reset_index(drop=True)],
        axis=1,
    )

    # 5. 保存输出 ──────────────────────────────────────────────────────────────
    print("\n[5/5] 保存输出文件 ...")

    path_poem_emb = OUTPUT_DIR / "bert_ccpoem_embeddings.npy"
    path_ana_emb  = OUTPUT_DIR / "bert_analysis_embeddings.npy"
    path_sim      = OUTPUT_DIR / "bert_similarity_top10.csv"
    path_token    = OUTPUT_DIR / "bert_token_importance.csv"

    np.save(str(path_poem_emb), poem_embs)
    np.save(str(path_ana_emb),  ana_embs)
    df_sim.to_csv(str(path_sim),   index=False, encoding="utf-8-sig")
    pd.DataFrame(token_rows).to_csv(str(path_token), index=False, encoding="utf-8-sig")

    # 元信息记录
    poem_model_name = getattr(
        mdl_poem.config, "_name_or_path",
        str(BERT_CCPOEM_PATH.name)
    )
    ana_model_name = getattr(
        mdl_ana.config, "_name_or_path",
        str(SIKUROBERTA_PATH.name)
    )
    meta = {
        "poem_model":      poem_model_name,
        "poem_model_path": str(BERT_CCPOEM_PATH),
        "poem_hidden_size": int(poem_embs.shape[1]),
        "ana_model":       ana_model_name,
        "ana_model_path":  str(SIKUROBERTA_PATH),
        "ana_hidden_size": int(ana_embs.shape[1]),
        "n_poems":         int(n),
        "device":          str(DEVICE),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    path_meta = OUTPUT_DIR / "bert_embed_meta.json"
    path_meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    elapsed_total = time.time() - t0
    print("\n" + "=" * 65)
    print("✅ 全部完成！")
    print(f"   总用时：{elapsed_total:.1f}s  ({elapsed_total/60:.1f} 分钟)")
    print(f"\n   输出文件：")
    for p in [path_poem_emb, path_ana_emb, path_sim, path_token, path_meta]:
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"   - {p.relative_to(PROJECT_ROOT)}  ({size_mb:.2f} MB)")
    print(f"\n   正文 embedding : shape {poem_embs.shape}  (模型: {poem_model_name})")
    print(f"   赏析 embedding : shape {ana_embs.shape}  (模型: {ana_model_name})")
    print("=" * 65)


if __name__ == "__main__":
    main()
