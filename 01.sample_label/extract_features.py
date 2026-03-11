"""
step2_features/extract_features.py
=====================================
特征提取：TF-IDF + TextRank关键句 + PMI花名-词共现

输出：
  tfidf_poem.csv / tfidf_analysis.csv / tfidf_combined.csv
    → 每首诗的TF-IDF特征向量（top词汇）
  textrank_keyphrases.csv
    → 每首诗赏析的关键词/关键句（TextRank）
  pmi_flower_word.csv
    → 花名-词汇的PMI共现强度矩阵

用法：
    python3 extract_features.py
"""

import argparse
import os
import json
import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── TF-IDF 参数 ───────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 300   # 每个版本保留的最大词汇数
TFIDF_MIN_DF       = 2     # 至少出现在2篇文档中（过滤极低频词）
TFIDF_MAX_DF       = 0.85  # 最多出现在85%文档中（过滤通用词）
TOP_WORDS_PER_DOC  = 20    # 每篇文档保存的top TF-IDF词数

# ── TextRank 参数 ─────────────────────────────────────────────────────────────
TEXTRANK_TOP_K     = 10    # 每篇文档提取的关键词数
TEXTRANK_WINDOW    = 4     # 共现窗口大小
TEXTRANK_DAMPING   = 0.85  # 阻尼系数
TEXTRANK_ITER      = 30    # 迭代次数

# ── PMI 参数 ──────────────────────────────────────────────────────────────────
PMI_MIN_COUNT      = 2     # 词汇最小出现次数（过滤噪音）
PMI_TOP_K          = 30    # 每个花名保留的top PMI词数


# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF
# ══════════════════════════════════════════════════════════════════════════════

def compute_tfidf(df: pd.DataFrame, version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算TF-IDF特征
    返回：(dense_top_words_df, sparse_matrix_df)
    """
    # 从tokens列还原成空格分隔的字符串（sklearn TF-IDF输入格式）
    corpus = []
    for tokens_json in df['tokens']:
        try:
            tokens = json.loads(tokens_json) if tokens_json else []
        except Exception:
            tokens = []
        corpus.append(' '.join(tokens) if tokens else '')

    # 处理空文档
    non_empty = [i for i, c in enumerate(corpus) if c.strip()]
    if len(non_empty) < 5:
        print(f"  ⚠️  {version} 版本有效文档不足5条，跳过TF-IDF")
        return pd.DataFrame(), pd.DataFrame()

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        token_pattern=r'(?u)\S+',   # 任何非空白字符序列（兼容中文）
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # ── 输出1：稀疏矩阵（全量特征）──────────────────────────────────────────
    matrix_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names
    )
    matrix_df.insert(0, 'ID', df['ID'].values)
    matrix_df.insert(1, 'sample_id', df['sample_id'].values)
    matrix_df.insert(2, '花名', df['花名'].values)
    matrix_df.insert(3, '月份', df['月份'].values)

    # ── 输出2：每篇文档top词（可读性强，供人工分析）────────────────────────
    rows = []
    for i, (_, doc_row) in enumerate(df.iterrows()):
        scores = tfidf_matrix[i].toarray()[0]
        top_idx = scores.argsort()[-TOP_WORDS_PER_DOC:][::-1]
        top_words = [(feature_names[j], round(float(scores[j]), 4))
                     for j in top_idx if scores[j] > 0]
        rows.append({
            'ID':        doc_row['ID'],
            'sample_id': doc_row['sample_id'],
            '花名':      doc_row['花名'],
            '月份':      doc_row['月份'],
            '朝代':      doc_row['朝代'],
            '作者':      doc_row['作者'],
            '诗名':      doc_row['诗名'],
            f'top{TOP_WORDS_PER_DOC}_words': json.dumps(top_words, ensure_ascii=False),
            'top5_preview': '、'.join([w for w, _ in top_words[:5]]),
        })
    top_df = pd.DataFrame(rows)

    return top_df, matrix_df


# ══════════════════════════════════════════════════════════════════════════════
# TextRank
# ══════════════════════════════════════════════════════════════════════════════

def textrank_keywords(tokens: list[str], top_k: int = TEXTRANK_TOP_K,
                      window: int = TEXTRANK_WINDOW,
                      damping: float = TEXTRANK_DAMPING,
                      n_iter: int = TEXTRANK_ITER) -> list[tuple[str, float]]:
    """
    基于TextRank的关键词提取
    共现窗口内的词建立无向图，PageRank迭代
    """
    if len(tokens) < 3:
        return [(t, 1.0) for t in tokens[:top_k]]

    # 构建共现图
    graph = defaultdict(lambda: defaultdict(float))
    for i, word in enumerate(tokens):
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if i != j:
                graph[word][tokens[j]] += 1.0

    # PageRank 迭代
    vocab = list(graph.keys())
    if not vocab:
        return []

    scores = {w: 1.0 for w in vocab}
    for _ in range(n_iter):
        new_scores = {}
        for w in vocab:
            neighbors = graph[w]
            total_out = sum(neighbors.values())
            contrib = sum(
                scores[nbr] * weight / max(sum(graph[nbr].values()), 1e-9)
                for nbr, weight in neighbors.items()
                if nbr in scores
            )
            new_scores[w] = (1 - damping) + damping * contrib
        scores = new_scores

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def compute_textrank(df: pd.DataFrame) -> pd.DataFrame:
    """对赏析版本做TextRank关键词提取"""
    rows = []
    for _, row in df.iterrows():
        try:
            tokens = json.loads(row['tokens']) if row['tokens'] else []
        except Exception:
            tokens = []

        keywords = textrank_keywords(tokens)
        rows.append({
            'ID':        row['ID'],
            'sample_id': row['sample_id'],
            '花名':      row['花名'],
            '月份':      row['月份'],
            '朝代':      row['朝代'],
            '作者':      row['作者'],
            '诗名':      row['诗名'],
            'textrank_keywords': json.dumps(keywords, ensure_ascii=False),
            'top5_keywords': '、'.join([w for w, _ in keywords[:5]]),
            'top10_keywords': '、'.join([w for w, _ in keywords[:10]]),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PMI 花名-词共现
# ══════════════════════════════════════════════════════════════════════════════

def compute_pmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算花名与词汇的PMI（点互信息）
    PMI(flower, word) = log P(flower,word) / (P(flower) * P(word))

    高PMI值表示该词与该花卉有强关联（超过随机共现的期望）
    """
    # 统计每个花名的文档集合
    flower_docs = defaultdict(list)
    all_word_counts = Counter()
    flower_word_counts = defaultdict(Counter)
    total_docs = len(df)

    for _, row in df.iterrows():
        flower = row['花名']
        try:
            tokens = json.loads(row['tokens']) if row['tokens'] else []
        except Exception:
            tokens = []
        unique_tokens = set(tokens)   # 每篇文档每个词只计一次

        flower_docs[flower].append(row['ID'])
        for word in unique_tokens:
            all_word_counts[word] += 1
            flower_word_counts[flower][word] += 1

    # 过滤低频词
    valid_words = {w for w, c in all_word_counts.items() if c >= PMI_MIN_COUNT}

    # 计算 PMI
    rows = []
    flower_doc_counts = {f: len(docs) for f, docs in flower_docs.items()}

    for flower, word_counts in flower_word_counts.items():
        flower_freq = flower_doc_counts[flower] / total_docs
        pmi_scores = []
        for word, co_count in word_counts.items():
            if word not in valid_words:
                continue
            word_freq   = all_word_counts[word] / total_docs
            joint_freq  = co_count / total_docs
            if word_freq > 0 and flower_freq > 0 and joint_freq > 0:
                pmi = math.log2(joint_freq / (flower_freq * word_freq))
                pmi_scores.append((word, round(pmi, 4), co_count))

        # 按PMI降序，取top K
        pmi_scores.sort(key=lambda x: x[1], reverse=True)
        top_pmi = pmi_scores[:PMI_TOP_K]

        rows.append({
            '花名':          flower,
            'doc_count':     flower_doc_counts[flower],
            'pmi_top_words': json.dumps(top_pmi, ensure_ascii=False),
            'top10_preview': '、'.join([w for w, _, _ in top_pmi[:10]]),
        })

    result = pd.DataFrame(rows).sort_values('doc_count', ascending=False)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess_dir', default='../step1_preprocess/output')
    args = parser.parse_args()

    preprocess_dir = os.path.join(os.path.dirname(__file__), args.preprocess_dir)

    print("=" * 55)
    print("Step 2: 特征提取")
    print("=" * 55)

    # ── TF-IDF（三版本） ──────────────────────────────────────────────────────
    print("\n【TF-IDF 特征提取】")
    for version in ['poem', 'analysis', 'combined']:
        csv_path = os.path.join(preprocess_dir, f"tokens_{version}.csv")
        if not os.path.exists(csv_path):
            print(f"  ⚠️  找不到 tokens_{version}.csv，跳过")
            continue
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"\n  {version} ({len(df)} 条)")
        top_df, matrix_df = compute_tfidf(df, version)
        if top_df.empty:
            continue
        top_path = os.path.join(OUTPUT_DIR, f"tfidf_{version}.csv")
        top_df.to_csv(top_path, index=False, encoding='utf-8-sig')
        print(f"  💾 tfidf_{version}.csv  ({len(top_df)} 条, {len(matrix_df.columns)-4} 特征维度)")

        matrix_path = os.path.join(OUTPUT_DIR, f"tfidf_{version}_matrix.csv")
        matrix_df.to_csv(matrix_path, index=False, encoding='utf-8-sig')
        print(f"  💾 tfidf_{version}_matrix.csv（完整特征矩阵）")

    # ── TextRank（赏析版本） ──────────────────────────────────────────────────
    print("\n【TextRank 关键词提取（赏析版）】")
    analysis_path = os.path.join(preprocess_dir, "tokens_analysis.csv")
    if os.path.exists(analysis_path):
        df_analysis = pd.read_csv(analysis_path, encoding='utf-8-sig')
        tr_df = compute_textrank(df_analysis)
        tr_path = os.path.join(OUTPUT_DIR, "textrank_keyphrases.csv")
        tr_df.to_csv(tr_path, index=False, encoding='utf-8-sig')
        print(f"  💾 textrank_keyphrases.csv  ({len(tr_df)} 条)")
        print(f"  示例（前3条）:")
        for _, row in tr_df.head(3).iterrows():
            print(f"    [{row['花名']}] {row['诗名'][:15]}... → {row['top5_keywords']}")
    else:
        print("  ⚠️  找不到 tokens_analysis.csv，跳过 TextRank")

    # ── PMI（赏析版本） ───────────────────────────────────────────────────────
    print("\n【PMI 花名-词共现】")
    if os.path.exists(analysis_path):
        df_analysis = pd.read_csv(analysis_path, encoding='utf-8-sig')
        pmi_df = compute_pmi(df_analysis)
        pmi_path = os.path.join(OUTPUT_DIR, "pmi_flower_word.csv")
        pmi_df.to_csv(pmi_path, index=False, encoding='utf-8-sig')
        print(f"  💾 pmi_flower_word.csv  ({len(pmi_df)} 个花名)")
        print(f"  示例（前5个花名）:")
        for _, row in pmi_df.head(5).iterrows():
            print(f"    [{row['花名']:6s} {row['doc_count']:2d}篇] → {row['top10_preview']}")

    print("\n✅ Step 2 完成")
    print(f"   输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
