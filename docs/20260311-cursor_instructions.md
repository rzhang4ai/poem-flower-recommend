
人工逐个步骤运行代码：
cd /Users/rzhang/Documents/poem-flower-recommend
source flower_env/bin/activate

# Step 1 预处理（显式指定刚生成的 sample_200）
python 01.sample_label/preprocess.py --input output/sample_200.csv

# Step 2 特征（从 Step1 的 output 读）
python 01.sample_label/extract_features.py --preprocess_dir output

# Step 3 聚类
python 01.sample_label/cluster_lda.py --preprocess_dir output --n_topics 7

# Step 4 情感
python 01.sample_label/sentiment.py --preprocess_dir output

# Step 5 规则（用 Step0 的 output/sample_200）
python 01.sample_label/rule_labeler.py --sample_csv output/sample_200.csv

# Step 6 汇总
python 01.sample_label/generate_report.py




# Cursor Agent 任务说明
## 诗花雅送 · Phase 2 传统AI标注流水线

---

## 你的任务

在项目 `/Users/rzhang/Documents/poem-flower-recommend` 中运行 `01.sample_label/` 下的传统AI标注流水线，逐步执行 Step 0–6，修复运行过程中遇到的任何错误，确保每一步成功输出文件后再进行下一步。

---

## 项目环境

```
项目根目录:   /Users/rzhang/Documents/poem-flower-recommend/
数据集:       /Users/rzhang/Documents/poem-flower-recommend/poems_dataset/poems_dataset_merged_done.csv 
流水线目录:   01.sample_label/
虚拟环境:     flower_env/  （需先激活）
```

---

## 执行前：环境准备

**第一步：激活虚拟环境**
```bash
cd /Users/rzhang/Documents/poem-flower-recommend
source flower_env/bin/activate
```

**第二步：安装依赖**
```bash
pip3 install -r 01.sample_label/requirements.txt
```

依赖清单：`jieba`, `snownlp`, `scikit-learn`, `numpy`, `scipy`, `pandas`, `matplotlib`

如果某个包安装失败，单独安装：
```bash
pip3 install jieba snownlp scikit-learn numpy scipy pandas matplotlib
```

**第三步：确认数据集存在**
```bash
ls -la poems_dataset_merged_done.csv
python3 -c "import pandas as pd; df=pd.read_csv('poems_dataset_v5.csv'); print(f'✅ {len(df)} 条数据，列：{list(df.columns)}')"
```

---

## 执行步骤（按顺序，逐步运行）

### Step 0 — 分层抽样 200 条

```bash
cd /Users/rzhang/Documents/poem-flower-recommend
python3 01.sample_label/step0_sample/sample_200.py
```

**预期输出：**
- `01.sample_label/step0_sample/output/sample_200.csv`
- `01.sample_label/step0_sample/output/sample_stats.txt`

**预期终端打印：**
- 月份分布表（每月约 16 条）
- 花名覆盖数（应接近 79 个）
- `💾 已保存: .../sample_200.csv`

**验证：**
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('01.sample_label/step0_sample/output/sample_200.csv')
print(f'抽样数: {len(df)}, 花名: {df[\"花名\"].nunique()}, 月份: {df[\"月份\"].nunique()}')
"
```

---

### Step 1 — 文本预处理（分词 + 词性 + 停用词）

```bash
python3 01.sample_label/step1_preprocess/preprocess.py
```

**预期输出（`step1_preprocess/output/`）：**
- `preprocessed.csv` — 完整预处理结果
- `tokens_poem.csv` — 正文版分词
- `tokens_analysis.csv` — 赏析版分词（主力版本）
- `tokens_combined.csv` — 合并版分词
- `stopwords_custom.txt` — 停用词文件

**预期终端打印：**
- 三版本 token 数统计（赏析版平均应在 30-80 个 token）
- 如有赏析为空的条目，会打印警告

**注意：** jieba 首次运行会加载词典，可能有几秒延迟，输出 `Building prefix dict...` 属正常。

---

### Step 2 — 特征提取（TF-IDF + TextRank + PMI）

```bash
python3 01.sample_label/step2_features/extract_features.py
```

**预期输出（`step2_features/output/`）：**
- `tfidf_poem.csv` / `tfidf_analysis.csv` / `tfidf_combined.csv`
- `tfidf_*_matrix.csv` — 完整特征矩阵
- `textrank_keyphrases.csv` — 每首诗的关键词
- `pmi_flower_word.csv` — 花名-词汇PMI共现

**验证重点：**
```bash
python3 -c "
import pandas as pd
pmi = pd.read_csv('01.sample_label/step2_features/output/pmi_flower_word.csv')
print('PMI示例（前3个花名）:')
print(pmi[['花名','doc_count','top10_preview']].head(3).to_string())
"
```

---

### Step 3 — 无监督聚类（LDA + k-Means + 层次聚类）

```bash
python3 01.sample_label/step3_unsupervised/cluster_lda.py
```

**预期输出（`step3_unsupervised/output/`）：**
- `lda_topics_analysis.csv` / `lda_topics_combined.csv`
- `lda_topic_keywords_analysis.csv` — 每个主题的 top 关键词
- `kmeans_labels_analysis.csv`
- `hierarchical_labels_analysis.csv`
- `figures/lda_coherence.png` — LDA主题数选择曲线
- `figures/kmeans_elbow.png` — k-Means肘部图
- `figures/dendrogram.png` — 层次聚类树状图

**这一步耗时最长（LDA搜索4-12个主题），约1-3分钟，属正常。**

终端会打印类似：
```
搜索最优主题数:
  n= 4  perplexity=1234.5
  n= 5  perplexity=1198.3
  ...
→ 推荐主题数: 7
Topic 0: 离别、相思、怀人、送别、故乡...
Topic 1: 高洁、清雅、傲寒、君子、隐逸...
```

**如果想跳过自动搜索，强制指定主题数（节省时间）：**
```bash
python3 01.sample_label/step3_unsupervised/cluster_lda.py --n_topics 7
```

---

### Step 4 — 情感基线（SnowNLP + NTUSD 8维）

```bash
python3 01.sample_label/step4_sentiment/sentiment.py
```

**预期输出（`step4_sentiment/output/`）：**
- `sentiment_scores.csv` — 每首诗的情感特征向量
- `figures/sentiment_distribution.png` — 情感分布图

**验证重点（三版本情感一致性）：**
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('01.sample_label/step4_sentiment/output/sentiment_scores.csv')
agree = (df['dominant_emotion_analysis'] == df['dominant_emotion_poem']).mean()
print(f'赏析 vs 正文情感一致率: {agree:.1%}')
print(df['dominant_emotion_analysis'].value_counts())
"
```

---

### Step 5 — 规则初标注

```bash
python3 01.sample_label/step5_rules/rule_labeler.py
```

**预期输出（`step5_rules/output/`）：**
- `rule_labels.csv`
- `rule_coverage_report.txt` — 覆盖率报告

**预期覆盖率：**
- 场合标注：30-50%（约 60-100 条）
- 关系标注：20-40%（约 40-80 条）
- 高置信度标签：约 20-35%

---

### Step 6 — 汇总报告 & 标注底稿

```bash
python3 01.sample_label/step6_report/generate_report.py
```

**预期输出（`step6_report/output/`）：**
- `annotation_draft.csv` — **核心输出，200条标注底稿**
- `high_confidence.csv`
- `low_confidence.csv`
- `figures/summary_dashboard.png`

**最终验证：**
```bash
python3 -c "
import pandas as pd
draft = pd.read_csv('01.sample_label/step6_report/output/annotation_draft.csv')
print(f'底稿总条数: {len(draft)}')
print(f'列数: {len(draft.columns)}')
print(f'高置信度: {draft[\"is_high_conf\"].sum()} 条')
print(f'需审核: {draft[\"needs_review\"].sum()} 条')
print(f'列名: {list(draft.columns)}')
"
```

---

## 常见报错处理

| 报错 | 原因 | 解决方式 |
|------|------|----------|
| `ModuleNotFoundError: No module named 'jieba'` | 依赖未安装 | `pip3 install jieba` |
| `ModuleNotFoundError: No module named 'snownlp'` | 依赖未安装 | `pip3 install snownlp` |
| `FileNotFoundError: poems_dataset_v5.csv` | 路径不对 | 确认在项目根目录运行，或用 `--input` 参数指定完整路径 |
| `FileNotFoundError: tokens_analysis.csv` | 上一步未完成 | 先运行 Step 1 |
| `ValueError: n_samples=X < n_clusters` | 有效文档太少 | Step 3 加 `--n_topics 5 --n_clusters 5` 降低参数 |
| `UnicodeDecodeError` | CSV编码问题 | 脚本已用 `utf-8-sig`，如仍报错检查文件编码 |
| matplotlib 中文显示方框 | 无中文字体 | 图表会自动降级为英文标签，不影响数据输出 |

---

## 完成后输出汇总

所有步骤成功后，关键文件位置：

```
01.sample_label/
├── step0_sample/output/sample_200.csv          ← 200条样本
├── step2_features/output/pmi_flower_word.csv   ← 花名-词PMI关联
├── step3_unsupervised/output/lda_topic_keywords_analysis.csv  ← LDA主题
├── step4_sentiment/output/sentiment_scores.csv ← 情感特征向量
├── step5_rules/output/rule_coverage_report.txt ← 规则覆盖报告
└── step6_report/output/
    ├── annotation_draft.csv    ← 标注底稿（最重要）
    ├── high_confidence.csv     ← 高置信度条目
    ├── low_confidence.csv      ← 需重点审核
    └── figures/summary_dashboard.png  ← 综合图表
```

完成后请将 `step6_report/output/` 整个文件夹上传到 Google Drive 团队共享文件夹。

