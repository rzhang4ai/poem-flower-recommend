"""
step4_sentiment/sentiment.py
==============================
情感基线分析：SnowNLP极性分数 + NTUSD词典8维情感向量

8维情感分类（对应项目标注体系）：
  joy       喜悦  → 庆祝/富贵/赞美
  sorrow    哀情  → 悲伤/哀思/凄婉
  longing   思念  → 相思/怀人/离愁
  bold      豪迈  → 壮志/昂扬/英气
  elegant   清雅  → 高洁/雅致/淡泊
  serene    平和  → 宁静/自然/恬淡
  resentment 怨思 → 郁郁/不平/失意
  anxiety   忧惧  → 忧虑/惶恐/无常

输出：
  sentiment_scores.csv          → 每首诗的情感特征向量
  figures/sentiment_dist.png    → 情感分布可视化

用法：
    python3 sentiment.py
"""

import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from snownlp import SnowNLP

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 中文字体 ──────────────────────────────────────────────────────────────────
def setup_chinese_font():
    candidates = ['PingFang SC', 'Heiti TC', 'STHeiti', 'SimHei',
                  'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            return font
    plt.rcParams['font.family'] = 'DejaVu Sans'
    return None


# ══════════════════════════════════════════════════════════════════════════════
# NTUSD 情感词典（内置核心词汇，可被外部文件扩充）
# ══════════════════════════════════════════════════════════════════════════════
#
# 设计说明：
#   每个情感维度包含：正向触发词（权重+1）和 负向触发词（通过否定词反转）
#   词汇来源：NTUSD情感词典 + 古典诗词赏析高频词人工整理
#   格式：{词: 权重}，权重范围 0.5-2.0（核心词权重更高）
#
EMOTION_LEXICON = {
    'joy': {
        # 积极情绪：喜悦、庆祝、赞美
        '喜悦': 1.5, '欢乐': 1.5, '欢快': 1.2, '欢欣': 1.2, '喜庆': 1.5,
        '祥和': 1.0, '吉祥': 1.2, '富贵': 1.2, '繁盛': 1.0, '兴旺': 1.0,
        '明媚': 1.0, '灿烂': 1.0, '热烈': 1.0, '盎然': 0.8, '生机': 0.8,
        '赞美': 1.2, '赞叹': 1.2, '称颂': 1.0, '颂扬': 1.0,
        '喜': 0.8, '乐': 0.8, '欢': 0.8, '笑': 0.7,
        '绚丽': 0.8, '娇艳': 0.8, '鲜艳': 0.8, '芬芳': 0.7,
    },
    'sorrow': {
        '悲': 1.5, '哀': 1.5, '凄': 1.5, '惨': 1.2, '苦': 1.0,
        '悲伤': 1.5, '悲哀': 1.5, '哀愁': 1.5, '凄婉': 1.5, '凄凉': 1.5,
        '哀思': 1.5, '悼念': 1.8, '伤逝': 1.5, '哀悼': 1.8, '泣': 1.2,
        '零落': 1.2, '凋零': 1.2, '凋谢': 1.2, '飘零': 1.0, '残': 1.0,
        '泪': 1.2, '哭': 1.2, '痛': 1.0, '愁': 1.0,
        '断肠': 1.8, '肝肠寸断': 2.0, '满目疮痍': 1.5,
    },
    'longing': {
        '思': 1.2, '念': 1.2, '忆': 1.2, '怀': 1.0,
        '思念': 1.5, '相思': 1.8, '思乡': 1.5, '怀念': 1.5, '忆旧': 1.2,
        '离别': 1.5, '别离': 1.5, '送别': 1.5, '惜别': 1.5, '话别': 1.2,
        '相隔': 1.0, '远方': 0.8, '故乡': 1.0, '故人': 1.0, '旧友': 1.0,
        '魂牵梦萦': 1.8, '朝思暮想': 1.8, '望断': 1.5,
        '归': 0.8, '还': 0.5, '寄': 0.8, '赠': 0.8,
    },
    'bold': {
        '豪迈': 1.8, '壮志': 1.8, '豪情': 1.8, '英气': 1.5, '雄浑': 1.5,
        '昂扬': 1.5, '慷慨': 1.5, '激昂': 1.5, '壮阔': 1.2, '气势': 1.2,
        '英雄': 1.2, '功名': 1.0, '征战': 1.0, '报国': 1.5, '忠勇': 1.5,
        '奋发': 1.2, '进取': 1.0, '抱负': 1.2, '理想': 0.8,
        '磅礴': 1.5, '苍劲': 1.2, '遒劲': 1.2,
        '男儿': 0.8, '热血': 1.0, '铮铮': 1.0,
    },
    'elegant': {
        '清雅': 1.8, '高洁': 1.8, '雅致': 1.5, '淡泊': 1.5, '脱俗': 1.5,
        '清逸': 1.5, '飘逸': 1.2, '清远': 1.2, '超然': 1.5, '出尘': 1.5,
        '幽静': 1.2, '幽雅': 1.2, '清幽': 1.2, '淡雅': 1.2, '素雅': 1.0,
        '君子': 1.5, '文人': 1.0, '雅士': 1.2, '隐士': 1.2,
        '清风': 0.8, '明月': 0.8, '竹': 0.8, '梅': 0.5, '兰': 0.5,
        '笔墨': 0.8, '书卷': 0.8, '翰墨': 1.0,
    },
    'serene': {
        '宁静': 1.5, '恬淡': 1.5, '平和': 1.5, '安详': 1.5, '静谧': 1.2,
        '悠然': 1.5, '闲适': 1.5, '自在': 1.2, '随性': 1.0,
        '自然': 1.0, '天然': 1.0, '质朴': 1.0, '朴素': 0.8,
        '山水': 0.8, '田园': 1.0, '隐逸': 1.5,
        '淡然': 1.2, '超脱': 1.2, '豁达': 1.5,
        '微风': 0.7, '轻盈': 0.7, '柔和': 0.7,
    },
    'resentment': {
        '怨': 1.5, '恨': 1.5, '愤': 1.5, '郁': 1.2,
        '怨恨': 1.8, '郁郁': 1.5, '愤懑': 1.8, '不平': 1.5, '怨愤': 1.8,
        '失意': 1.5, '落魄': 1.2, '潦倒': 1.2, '蹉跎': 1.2,
        '怀才不遇': 2.0, '壮志难酬': 1.8, '报国无门': 1.8,
        '牢骚': 1.5, '抑郁': 1.5, '压抑': 1.2,
        '政治': 0.5, '贬谪': 1.5, '流放': 1.5, '放逐': 1.5,
    },
    'anxiety': {
        '忧': 1.2, '惧': 1.2, '恐': 1.2, '虑': 0.8,
        '忧虑': 1.5, '忧愁': 1.5, '惶恐': 1.5, '担忧': 1.2, '焦虑': 1.2,
        '无常': 1.5, '幻灭': 1.5, '虚无': 1.2, '短暂': 1.0, '易逝': 1.2,
        '叹息': 1.2, '感慨': 0.8, '嗟叹': 1.2,
        '盛极而衰': 1.5, '昙花一现': 1.5, '人生苦短': 1.5,
        '惆怅': 1.5, '彷徨': 1.2, '迷茫': 1.0,
    },
}

# 否定词（出现后将当前情感权重反转）
NEGATION_WORDS = {'不', '无', '非', '未', '没', '莫', '勿', '别', '休', '难'}


# ══════════════════════════════════════════════════════════════════════════════
# SnowNLP 情感分析
# ══════════════════════════════════════════════════════════════════════════════

def snownlp_score(text: str) -> dict:
    """
    SnowNLP对文本的情感极性分析
    返回：正文、赏析、合并的polarity分数（0-1，>0.5为正面）
    注意：SnowNLP训练语料以现代中文为主，古文精度较低
    """
    if not text or not str(text).strip():
        return {'polarity': 0.5, 'sentiment': 'neutral', 'confidence': 0.0}
    try:
        s = SnowNLP(str(text))
        polarity = round(float(s.sentiments), 4)
        sentiment = 'positive' if polarity > 0.6 else ('negative' if polarity < 0.4 else 'neutral')
        confidence = abs(polarity - 0.5) * 2   # 0=完全中性, 1=强烈
        return {
            'polarity':   polarity,
            'sentiment':  sentiment,
            'confidence': round(confidence, 4)
        }
    except Exception:
        return {'polarity': 0.5, 'sentiment': 'neutral', 'confidence': 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# NTUSD 8维情感向量
# ══════════════════════════════════════════════════════════════════════════════

def compute_emotion_vector(tokens: list[str]) -> dict:
    """
    基于情感词典计算8维情感特征向量
    处理否定词：窗口内遇到否定词，下一个情感词权重取负
    """
    scores = {dim: 0.0 for dim in EMOTION_LEXICON}
    negation_window = 3   # 否定词影响后续N个token

    neg_countdown = 0
    for token in tokens:
        if token in NEGATION_WORDS:
            neg_countdown = negation_window
            continue
        neg = (neg_countdown > 0)
        neg_countdown = max(0, neg_countdown - 1)

        for dim, lex in EMOTION_LEXICON.items():
            if token in lex:
                weight = lex[token]
                scores[dim] += (-weight if neg else weight)

    # 归一化：每维度除以词数（避免长文本bias）
    n_tokens = max(len(tokens), 1)
    scores = {dim: round(v / n_tokens, 6) for dim, v in scores.items()}

    # 主导情感（最高分维度）
    dominant = max(scores, key=scores.get)
    dominant_score = scores[dominant]

    return {**scores, 'dominant_emotion': dominant, 'dominant_score': round(dominant_score, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# 三版本对比分析
# ══════════════════════════════════════════════════════════════════════════════

def analyze_sentiment(df_analysis: pd.DataFrame,
                       df_poem: pd.DataFrame,
                       df_combined: pd.DataFrame) -> pd.DataFrame:
    """对三个版本计算完整情感特征，合并为一个宽表"""
    results = []

    for i, row_a in df_analysis.iterrows():
        row_p = df_poem.iloc[i]
        row_c = df_combined.iloc[i]

        # tokens
        tokens_a = json.loads(row_a['tokens']) if row_a['tokens'] else []
        tokens_p = json.loads(row_p['tokens']) if row_p['tokens'] else []
        tokens_c = json.loads(row_c['tokens']) if row_c['tokens'] else []

        # SnowNLP（对原始清洗文本）
        snow_a = snownlp_score(row_a.get('text', ''))
        snow_p = snownlp_score(row_p.get('text', ''))
        snow_c = snownlp_score(row_c.get('text', ''))

        # NTUSD 8维向量
        emo_a = compute_emotion_vector(tokens_a)
        emo_p = compute_emotion_vector(tokens_p)
        emo_c = compute_emotion_vector(tokens_c)

        result = {
            'ID':        row_a['ID'],
            'sample_id': row_a['sample_id'],
            '花名':      row_a['花名'],
            '月份':      row_a['月份'],
            '朝代':      row_a['朝代'],
            '作者':      row_a['作者'],
            '诗名':      row_a['诗名'],
            # SnowNLP
            'snow_polarity_analysis':  snow_a['polarity'],
            'snow_polarity_poem':      snow_p['polarity'],
            'snow_polarity_combined':  snow_c['polarity'],
            'snow_sentiment_analysis': snow_a['sentiment'],
            # NTUSD主导情感（三版本）
            'dominant_emotion_analysis': emo_a['dominant_emotion'],
            'dominant_emotion_poem':     emo_p['dominant_emotion'],
            'dominant_emotion_combined': emo_c['dominant_emotion'],
            'dominant_score_analysis':   emo_a['dominant_score'],
        }

        # 展开8维情感分数（赏析版本，作为主要特征）
        for dim in EMOTION_LEXICON:
            result[f'emo_{dim}'] = emo_a[dim]

        # 三版本情感向量JSON（供后续模块使用）
        result['emotion_vec_analysis'] = json.dumps(emo_a, ensure_ascii=False)
        result['emotion_vec_poem']     = json.dumps(emo_p, ensure_ascii=False)
        result['emotion_vec_combined'] = json.dumps(emo_c, ensure_ascii=False)

        results.append(result)

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════════════

def plot_sentiment_distribution(df: pd.DataFrame, font_name: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. SnowNLP极性分布（三版本对比）
    ax = axes[0, 0]
    for version, color in [('analysis', '#2D4A6A'), ('poem', '#8B3A2A'), ('combined', '#2D6A4F')]:
        col = f'snow_polarity_{version}'
        if col in df.columns:
            ax.hist(df[col], bins=20, alpha=0.5, label=version, color=color)
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('SnowNLP Polarity'); ax.set_ylabel('Count')
    ax.set_title('SnowNLP Polarity Distribution (3 versions)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. 8维情感平均强度（赏析版）
    ax = axes[0, 1]
    dims = list(EMOTION_LEXICON.keys())
    means = [df[f'emo_{d}'].mean() for d in dims]
    colors = ['#E8C547', '#6B9ED2', '#9B59B6', '#E74C3C',
              '#2ECC71', '#1ABC9C', '#E67E22', '#95A5A6']
    bars = ax.bar(dims, means, color=colors, alpha=0.8)
    ax.set_xlabel('Emotion Dimension'); ax.set_ylabel('Mean Score')
    ax.set_title('8-dim Emotion Profile (analysis version)')
    ax.tick_params(axis='x', rotation=30); ax.grid(True, alpha=0.3, axis='y')

    # 3. 主导情感分布（饼图）
    ax = axes[1, 0]
    dominant_counts = df['dominant_emotion_analysis'].value_counts()
    wedge_colors = ['#E8C547', '#6B9ED2', '#9B59B6', '#E74C3C',
                    '#2ECC71', '#1ABC9C', '#E67E22', '#95A5A6']
    ax.pie(dominant_counts.values,
           labels=dominant_counts.index,
           colors=wedge_colors[:len(dominant_counts)],
           autopct='%1.1f%%', startangle=90)
    ax.set_title('Dominant Emotion Distribution (analysis version)')

    # 4. SnowNLP极性 vs 词典主导情感（散点）
    ax = axes[1, 1]
    emotion_order = list(EMOTION_LEXICON.keys())
    emotion_int = df['dominant_emotion_analysis'].map(
        {e: i for i, e in enumerate(emotion_order)}
    ).fillna(0)
    scatter_colors = [wedge_colors[int(x) % len(wedge_colors)] for x in emotion_int]
    ax.scatter(df['snow_polarity_analysis'], df['dominant_score_analysis'],
               c=scatter_colors, alpha=0.6, s=40)
    ax.set_xlabel('SnowNLP Polarity'); ax.set_ylabel('Emotion Lexicon Score')
    ax.set_title('SnowNLP vs Lexicon Emotion (analysis version)')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "sentiment_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  💾 figures/sentiment_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess_dir', default='../step1_preprocess/output')
    args = parser.parse_args()

    font = setup_chinese_font()
    preprocess_dir = os.path.join(os.path.dirname(__file__), args.preprocess_dir)

    print("=" * 55)
    print("Step 4: 情感基线分析")
    print("=" * 55)

    # 读取三版本
    dfs = {}
    for version in ['analysis', 'poem', 'combined']:
        path = os.path.join(preprocess_dir, f"tokens_{version}.csv")
        if not os.path.exists(path):
            print(f"⚠️  找不到 tokens_{version}.csv")
            continue
        dfs[version] = pd.read_csv(path, encoding='utf-8-sig')
        print(f"✅ 读取 tokens_{version}.csv ({len(dfs[version])} 条)")

    if 'analysis' not in dfs:
        print("❌ 缺少 tokens_analysis.csv，无法继续")
        return

    # 补齐缺失版本（用analysis代替）
    df_poem     = dfs.get('poem',     dfs['analysis'])
    df_combined = dfs.get('combined', dfs['analysis'])

    print("\n⚙️  计算SnowNLP + NTUSD情感向量...")
    result_df = analyze_sentiment(dfs['analysis'], df_poem, df_combined)

    # 输出
    out_path = os.path.join(OUTPUT_DIR, "sentiment_scores.csv")
    result_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 sentiment_scores.csv ({len(result_df)} 条)")

    # 统计报告
    print("\n── 情感统计 ─────────────────────────────────")
    print(f"  SnowNLP极性（赏析版）:")
    print(f"    正面(>0.6): {(result_df['snow_polarity_analysis'] > 0.6).sum()} 条")
    print(f"    负面(<0.4): {(result_df['snow_polarity_analysis'] < 0.4).sum()} 条")
    print(f"    中性:       {((result_df['snow_polarity_analysis'] >= 0.4) & (result_df['snow_polarity_analysis'] <= 0.6)).sum()} 条")
    print(f"\n  NTUSD主导情感分布（赏析版）:")
    for emotion, count in result_df['dominant_emotion_analysis'].value_counts().items():
        bar = '█' * count
        print(f"    {emotion:12s}: {count:3d} {bar}")

    # 三版本主导情感一致性
    agree = (result_df['dominant_emotion_analysis'] == result_df['dominant_emotion_poem']).sum()
    total = len(result_df)
    print(f"\n  情感一致性（赏析 vs 正文）: {agree}/{total} = {agree/total:.1%}")

    # 可视化
    plot_sentiment_distribution(result_df, font)

    print("\n✅ Step 4 完成")


if __name__ == '__main__':
    main()
