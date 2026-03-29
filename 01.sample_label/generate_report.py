"""
step6_report/generate_report.py
=================================
汇总所有步骤结果，生成：
  1. annotation_draft.csv   → 供LLM标注和人工审核的底稿（含所有传统AI特征）
  2. low_confidence.csv     → 重点人工审核条目
  3. figures/               → 综合可视化图表

底稿字段设计：
  基础信息 + TF-IDF关键词 + TextRank关键词 + LDA主题 +
  SnowNLP极性 + NTUSD情感向量 + 规则标注 +
  【待填写列】: occasion / relation / symbolism / emotion_tone（供LLM和人工填写）

用法：
    python3 generate_report.py
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

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# 各步骤输出目录（相对路径；本项目中所有步骤输出均在 output/ 下）
STEP_DIRS = {
    'step0': 'output',
    'step1': 'output',
    'step2': 'output',
    'step3': 'output',
    'step4': 'output',
    'step5': 'output',
}


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


def load_csv(base_dir: str, rel_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(base_dir, rel_dir, filename)
    if os.path.exists(path):
        return pd.read_csv(path, encoding='utf-8-sig')
    print(f"  ⚠️  找不到: {path}")
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 合并所有步骤结果
# ══════════════════════════════════════════════════════════════════════════════

def merge_all(base_dir: str) -> pd.DataFrame:
    """将所有步骤输出合并为一张宽表，以sample_200.csv为主键"""
    print("  读取各步骤输出...")

    # 主表：200条样本
    df = load_csv(base_dir, STEP_DIRS['step0'], 'sample_200.csv')
    if df.empty:
        print("❌ 找不到 sample_200.csv，无法继续")
        return pd.DataFrame()
    print(f"    主表: {len(df)} 条")

    # Step2: TF-IDF关键词（赏析版）
    tfidf = load_csv(base_dir, STEP_DIRS['step2'], 'tfidf_analysis.csv')
    if not tfidf.empty:
        tfidf = tfidf[['ID', 'top5_preview']].rename(columns={'top5_preview': 'tfidf_top5'})
        df = df.merge(tfidf, on='ID', how='left')
        print(f"    TF-IDF: {len(tfidf)} 条")

    # Step2: TextRank关键词
    textrank = load_csv(base_dir, STEP_DIRS['step2'], 'textrank_keyphrases.csv')
    if not textrank.empty:
        textrank = textrank[['ID', 'top5_keywords', 'top10_keywords']].rename(
            columns={'top5_keywords': 'textrank_top5', 'top10_keywords': 'textrank_top10'})
        df = df.merge(textrank, on='ID', how='left')
        print(f"    TextRank: {len(textrank)} 条")

    # Step3: LDA主题（赏析版）
    lda = load_csv(base_dir, STEP_DIRS['step3'], 'lda_topics_analysis.csv')
    if not lda.empty:
        lda_cols = ['ID', 'dominant_topic', 'dominant_topic_score']
        # 加入所有topic_X列
        topic_cols = [c for c in lda.columns if c.startswith('topic_')]
        lda = lda[lda_cols + topic_cols]
        df = df.merge(lda, on='ID', how='left')
        print(f"    LDA: {len(lda)} 条, {len(topic_cols)} 主题")

    # Step3: k-Means聚类
    kmeans = load_csv(base_dir, STEP_DIRS['step3'], 'kmeans_labels_analysis.csv')
    if not kmeans.empty:
        kmeans = kmeans[['ID', 'kmeans_cluster']].rename(
            columns={'kmeans_cluster': 'kmeans_cluster'})
        df = df.merge(kmeans, on='ID', how='left')
        print(f"    k-Means: {len(kmeans)} 条")

    # Step4: 情感分析
    sentiment = load_csv(base_dir, STEP_DIRS['step4'], 'sentiment_scores.csv')
    if not sentiment.empty:
        sent_cols = ['ID', 'snow_polarity_analysis', 'snow_sentiment_analysis',
                     'dominant_emotion_analysis', 'dominant_score_analysis']
        # 加入8维情感分数
        emo_cols = [c for c in sentiment.columns if c.startswith('emo_')]
        sentiment = sentiment[sent_cols + emo_cols]
        df = df.merge(sentiment, on='ID', how='left')
        print(f"    情感: {len(sentiment)} 条")

    # Step5: 规则标注
    rules = load_csv(base_dir, STEP_DIRS['step5'], 'rule_labels.csv')
    if not rules.empty:
        rule_cols = ['ID', 'occasion', 'occasion_cn', 'occasion_conf',
                     'relation', 'relation_cn', 'relation_conf',
                     'symbolism_preview', 'overall_conf', 'is_high_conf', 'needs_review']
        rules = rules[rule_cols]
        df = df.merge(rules, on='ID', how='left')
        print(f"    规则标注: {len(rules)} 条")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 生成标注底稿（annotation_draft.csv）
# ══════════════════════════════════════════════════════════════════════════════

def build_annotation_draft(df: pd.DataFrame) -> pd.DataFrame:
    """
    重新排列字段，加入【待填写】列，生成供LLM和人工标注的底稿。

    底稿列顺序设计原则：
    1. 基础信息（人工能直接看懂的）
    2. 传统AI特征（供参考）
    3. 规则初标注（高置信度可直接采用）
    4. 待填写列（LLM和人工要填的）
    """
    draft = pd.DataFrame()

    # ── 基础信息 ──────────────────────────────────────────────────────────────
    for col in ['ID', 'sample_id', '花名', '月份', '朝代', '作者', '诗名', '正文', '赏析']:
        if col in df.columns:
            draft[col] = df[col]

    # ── 传统AI特征（供参考，标注时可见）────────────────────────────────────────
    for col in ['tfidf_top5', 'textrank_top5', 'textrank_top10']:
        if col in df.columns:
            draft[col] = df[col]

    for col in ['dominant_topic', 'dominant_topic_score', 'kmeans_cluster']:
        if col in df.columns:
            draft[col] = df[col]

    for col in ['snow_polarity_analysis', 'snow_sentiment_analysis',
                'dominant_emotion_analysis', 'dominant_score_analysis']:
        if col in df.columns:
            draft[col] = df[col]

    emo_cols = [c for c in df.columns if c.startswith('emo_')]
    for col in emo_cols:
        draft[col] = df[col]

    # ── 规则初标注（可直接采用或修改）──────────────────────────────────────────
    for col in ['occasion', 'occasion_cn', 'occasion_conf',
                'relation', 'relation_cn', 'relation_conf',
                'symbolism_preview', 'overall_conf', 'is_high_conf', 'needs_review']:
        if col in df.columns:
            draft[col] = df[col]

    # ── 待填写列（LLM第一轮标注后，人工审核时修改这些列）──────────────────────
    # 规则已标注的条目可预填，其余留空
    draft['label_occasion']      = df.get('occasion_cn', '')      # 赠送场合（待确认/修改）
    draft['label_relation']      = df.get('relation_cn', '')      # 赠送关系（待确认/修改）
    draft['label_emotion_tone']  = df.get('dominant_emotion_analysis', '')  # 情感基调
    draft['label_symbolism']     = df.get('symbolism_preview', '') # 花卉象征（待补充）
    draft['label_color']         = ''                              # 颜色联想（人工填）
    draft['llm_occasion']        = ''                              # LLM标注：场合
    draft['llm_relation']        = ''                              # LLM标注：关系
    draft['llm_emotion']         = ''                              # LLM标注：情感
    draft['llm_symbolism']       = ''                              # LLM标注：象征
    draft['human_review_note']   = ''                              # 人工审核备注
    draft['final_occasion']      = ''                              # 最终确认：场合
    draft['final_relation']      = ''                              # 最终确认：关系
    draft['final_emotion']       = ''                              # 最终确认：情感
    draft['final_symbolism']     = ''                              # 最终确认：象征
    draft['annotation_status']   = draft['is_high_conf'].apply(
        lambda x: 'high_conf_rule' if x else 'needs_llm'
    ) if 'is_high_conf' in draft.columns else 'needs_llm'

    return draft


# ══════════════════════════════════════════════════════════════════════════════
# 综合可视化
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(df: pd.DataFrame, draft: pd.DataFrame, font_name: str):
    """生成综合分析图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. 各月份情感极性箱线图
    ax = axes[0, 0]
    if 'snow_polarity_analysis' in df.columns and '月份' in df.columns:
        month_groups = {}
        month_order_map = {'正月':1,'二月':2,'三月':3,'四月':4,'五月':5,'六月':6,
                           '七月':7,'八月':8,'九月':9,'十月':10,'十一月':11,'十二月':12}
        for month, grp in df.groupby('月份'):
            month_groups[month] = grp['snow_polarity_analysis'].dropna().values
        sorted_months = sorted(month_groups.keys(), key=lambda m: month_order_map.get(m, 99))
        data_to_plot = [month_groups[m] for m in sorted_months if len(month_groups[m]) > 0]
        valid_months = [m for m in sorted_months if len(month_groups.get(m, [])) > 0]
        if data_to_plot:
            ax.boxplot(data_to_plot, labels=valid_months)
            ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
            ax.set_ylabel('SnowNLP Polarity')
            ax.set_title('Sentiment by Month')
            ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    # 2. 花名-主导情感热力图（top10花名）
    ax = axes[0, 1]
    if 'dominant_emotion_analysis' in df.columns and '花名' in df.columns:
        top_flowers = df['花名'].value_counts().head(10).index
        emotions = ['joy', 'sorrow', 'longing', 'bold', 'elegant', 'serene', 'resentment', 'anxiety']
        heat_data = np.zeros((len(top_flowers), len(emotions)))
        for i, flower in enumerate(top_flowers):
            sub = df[df['花名'] == flower]
            for j, emo in enumerate(emotions):
                col = f'emo_{emo}'
                if col in df.columns:
                    heat_data[i, j] = sub[col].mean()
        im = ax.imshow(heat_data, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(emotions))); ax.set_xticklabels(emotions, rotation=30, ha='right')
        ax.set_yticks(range(len(top_flowers))); ax.set_yticklabels(top_flowers)
        ax.set_title('Flower-Emotion Heatmap (Top 10 Flowers)')
        plt.colorbar(im, ax=ax)

    # 3. LDA主题分布（各主题文档数）
    ax = axes[0, 2]
    if 'dominant_topic' in df.columns:
        topic_counts = df['dominant_topic'].value_counts().sort_index()
        ax.bar(topic_counts.index.astype(str), topic_counts.values, color='#2D4A6A', alpha=0.8)
        ax.set_xlabel('Topic ID'); ax.set_ylabel('Document Count')
        ax.set_title('LDA Topic Distribution')
        ax.grid(True, alpha=0.3, axis='y')

    # 4. 场合标注覆盖率
    ax = axes[1, 0]
    if 'occasion_cn' in draft.columns:
        occ_counts = draft['label_occasion'].replace('', '未标注').value_counts()
        colors = ['#E8C547' if v != '未标注' else '#CCCCCC' for v in occ_counts.index]
        ax.barh(occ_counts.index, occ_counts.values, color=colors, alpha=0.85)
        ax.set_xlabel('Count'); ax.set_title('Occasion Label Distribution')
        ax.grid(True, alpha=0.3, axis='x')

    # 5. 置信度分布
    ax = axes[1, 1]
    if 'overall_conf' in draft.columns:
        conf_vals = draft['overall_conf'].fillna(0)
        ax.hist(conf_vals, bins=20, color='#2D6A4F', alpha=0.8)
        ax.axvline(0.65, color='#8B3A2A', linestyle='--', label='High conf threshold (0.65)')
        ax.axvline(0.40, color='orange', linestyle='--', label='Review threshold (0.40)')
        ax.set_xlabel('Overall Confidence'); ax.set_ylabel('Count')
        ax.set_title('Rule Label Confidence Distribution')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 6. annotation_status 分布
    ax = axes[1, 2]
    if 'annotation_status' in draft.columns:
        status_counts = draft['annotation_status'].value_counts()
        wedge_colors = ['#2D6A4F', '#E8C547', '#8B3A2A']
        ax.pie(status_counts.values,
               labels=status_counts.index,
               colors=wedge_colors[:len(status_counts)],
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Annotation Status Distribution')

    fig.suptitle('Poetic Flora Advisor — Traditional AI Analysis Summary', fontsize=14, y=1.01)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "summary_dashboard.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  💾 figures/summary_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    font = setup_chinese_font()
    base_dir = os.path.dirname(__file__)

    print("=" * 55)
    print("Step 6: 汇总报告 & 标注底稿生成")
    print("=" * 55)

    # 合并所有步骤
    print("\n【合并各步骤结果】")
    df = merge_all(base_dir)
    if df.empty:
        return

    # 生成标注底稿
    print("\n【生成标注底稿】")
    draft = build_annotation_draft(df)

    # 输出主文件
    draft_path = os.path.join(OUTPUT_DIR, "annotation_draft.csv")
    draft.to_csv(draft_path, index=False, encoding='utf-8-sig')
    print(f"  💾 annotation_draft.csv  ({len(draft)} 条, {len(draft.columns)} 列)")

    # 输出重点审核文件（置信度低 or 多规则矛盾）
    if 'needs_review' in draft.columns:
        low_conf = draft[draft['needs_review'] == True].copy()
        low_conf_path = os.path.join(OUTPUT_DIR, "low_confidence.csv")
        low_conf.to_csv(low_conf_path, index=False, encoding='utf-8-sig')
        print(f"  💾 low_confidence.csv    ({len(low_conf)} 条需重点审核)")

    # 高置信度规则标注（可直接采用）
    if 'is_high_conf' in draft.columns:
        high_conf = draft[draft['is_high_conf'] == True].copy()
        high_conf_path = os.path.join(OUTPUT_DIR, "high_confidence.csv")
        high_conf.to_csv(high_conf_path, index=False, encoding='utf-8-sig')
        print(f"  💾 high_confidence.csv   ({len(high_conf)} 条高置信度，可直接采用)")

    # 可视化
    print("\n【生成可视化图表】")
    plot_summary(df, draft, font)

    # 打印最终摘要
    total = len(draft)
    high_n = draft['is_high_conf'].sum() if 'is_high_conf' in draft.columns else 0
    low_n  = draft['needs_review'].sum() if 'needs_review' in draft.columns else 0
    mid_n  = total - high_n - low_n

    print(f"""
{'='*55}
传统AI分析完成！标注底稿摘要：
{'='*55}
  总条数:          {total}
  高置信度规则标注: {high_n} 条 ({high_n/total:.1%}) → 人工快速确认即可
  中等置信度:       {mid_n} 条 ({mid_n/total:.1%}) → LLM标注后人工审核
  需重点审核:       {low_n} 条 ({low_n/total:.1%}) → 优先人工处理

下一步：
  1. 将 annotation_draft.csv 送入LLM标注脚本（填写 llm_* 列）
  2. 人工审核 low_confidence.csv（优先）
  3. 合并LLM结果和人工审核结果 → 填写 final_* 列
  4. 将高质量标注数据集用于训练分类器
{'='*55}
""")


if __name__ == '__main__':
    main()
