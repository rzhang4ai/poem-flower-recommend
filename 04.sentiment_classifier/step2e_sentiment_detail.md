# step2e_sentiment — 古诗词情感识别详细说明

> 工作阶段：Phase 2 升级版  
> 目标：用领域专属情感词典替代通用工具（SnowNLP），对 1075 首项目诗词做精准情感分析

---

## 1. 工作目标

### 1.1 要解决的核心问题

`01.sample_label/sentiment.py` 的原始方案有两个已知缺陷：

1. **SnowNLP 不适合古诗词**：SnowNLP 训练语料是现代汉语电商评论，对"枯藤老树昏鸦"这类古诗风格无法正确判断情感，会把本来悲凉的词汇判为正面。
2. **手写词典规模小**：原 `EMOTION_LEXICON` 约 200 词/维度，属于人工拍脑袋的结果，无文献支撑，覆盖率低。

### 1.2 本步骤产出

| 输出文件 | 内容 |
|----------|------|
| `output/lexicon/fccpsl_terms.csv` | FCCPSL 解析后的 14,368 术语表（词、C3类、C2类、C1类） |
| `output/lexicon/combined_lexicon.csv` | FCCPSL + NTUSD 合并词典，附来源标记 |
| `output/results/sentiment_per_poem.csv` | 1075 首诗的情感分析结果（FCCPSL三层得分 + 主导情感） |
| `output/results/flower_sentiment.csv` | 79 种植物的情感分布画像 |
| `output/evaluation/fspc_evaluation.csv` | 在 FSPC 5000 首标注语料上的准确率评估 |
| `output/figures/` | 情感分布图、植物情感热图 |

---

## 2. 三个资源概述与相互关系

### 2.1 NTUSD（台湾大学情感词典）

**来源**：台湾大学中文情感词典（Ku et al., 2006）  
**规模**：约 8,276 词（2,818 正向 + 8,276 负向，含部分重叠）  
**特点**：  
- 通用中文（繁简都有），未专门针对古诗词
- 覆盖面广，词条权重无细粒度分类
- 本项目已有该词典

**在本步骤的角色**：补充词典（FCCPSL 的来源词典之一，在 FCCPSL 构建时已被融合）。用于覆盖 FCCPSL 中缺失的常用情感词，尤其是用于赏析文本（现代汉语）的分析。

```
NTUSD 格式（已有）：
  词条  极性（1=正向，-1=负向）
  喜悦   1
  悲伤  -1
```

### 2.2 FCCPSL（古诗词细粒度情感词典）

**来源**：南京大学，张威等，Neural Computing and Applications 2022  
**规模**：14,368 词，OWL 本体格式（`FCCPSL.owl`，已下载至 `output/lexicon/`）  
**核心创新**：
- 以《唐诗鉴赏词典》为训练语料（领域专属）
- 用 BERT + 情感偏旁特征（Emotion Radical, ER）做情感词抽取（F1=96.09%）
- 用 BiLSTM-Attention + 层级分类做情感类别判断（Acc=91.38%）
- 融合了 NTUSD、HowNet、李军词典、DUTIR 四个通用词典

**四层 OWL 本体结构**（从 `FCCPSL.owl` 实际解析）：

```
C1（极性）：positive, negative
  └─ C2（情感族）：
       positive → pleasure, favour, surprise
       negative → sadness, disgust
         └─ C3（情感类）：
              pleasure → joy(PA), ease(PE)
              favour   → praise(PD/PH), like(PB), faith(PG), wish(PK)
              surprise → peculiar(PC)
              sadness  → sorrow(NB/NJ), miss(PF), fear(NI/NC), guilt(NG/NH)
              disgust  → criticize(NN/ND/NK), anger(NA), vexed(NE), misgive(NL)
                └─ C4（实际术语，14,368个）：
                     例：C3_ease 下 → 化干戈为玉帛、归隐田园、风清气爽、花气袭人...
                         C3_sorrow 下 → 断肠、望断、涕泗滂沱...
```

**为什么直接用 FCCPSL 的分类体系（不做映射）**：
- FCCPSL 的 15 类（C3 层）是数据驱动的，有学术依据，内部一致性强
- 原始 8 维是手工设计，与 FCCPSL 不完全对应（如"elegant/清雅"在 FCCPSL 中可能分布在 ease+praise 中，强行映射会失真）
- 直接用 C3 层 15 类输出更精准、更可解释、更利于后续分析

**FCCPSL 与 NTUSD 的关系**：NTUSD 是 FCCPSL 的**输入来源之一**，在构建 FCCPSL 时已被融合。理论上 FCCPSL 覆盖了 NTUSD 的核心词汇并进行了古诗词领域适配。

### 2.3 FSPC（清华细粒度情感诗词语料库）

**来源**：清华大学 THUNLP-AIPoet，Chen et al., IJCAI 2019  
**规模**：5,000 首古诗词，人工标注  
**标注维度**：
- 整首诗情感（holistic）+ 每句情感（line1~line4）
- 5 类：`1=negative, 2=implicit negative, 3=neutral, 4=implicit positive, 5=positive`

**格式**：
```json
{
  "poet": "韦庄",
  "poem": "自有春愁正断魂|不堪芳草思王孙|落花寂寂黄昏雨|深院无人独倚门",
  "dynasty": "唐",
  "setiments": {"holistic": "1", "line1": "1", "line2": "1", "line3": "2", "line4": "2"},
  "title": "春愁"
}
```

**关键洞察："隐性情感"（implicit）**

古诗词最难处理的是隐性情感——诗句中没有直接的情感词，但通过意象传达情绪：
- "枯藤老树昏鸦，小桥流水人家" → implicit negative（无情感词，但意境凄凉）
- "春风又绿江南岸" → implicit positive（无情感词，但意境生机盎然）

词典法（包括 FCCPSL）对隐性情感的识别有先天局限——这是**阶段二（SikuRoBERTa 微调）**要解决的问题。

**在本步骤的角色**：作为**评估基准**，检验我们词典法（FCCPSL）的实际准确率。FSPC 的 5 类标签可以映射到 FCCPSL 的 positive/negative/neutral 三极性上进行对比。

---

## 3. 三者与原始方案的关系图

```
原始方案                          本步骤升级
────────────────────────────────────────────────────────
SnowNLP（现代汉语）               [彻底移除]
手写EMOTION_LEXICON（~200词）     → FCCPSL 14,368词（古诗词专属）
NTUSD（通用词典，已有）           → 补充词典（覆盖赏析现代汉语部分）
无验证语料                        → FSPC 5000首验证准确率
8维主观框架                       → FCCPSL 15类（C3层，数据驱动）

评估方法                          对比方法
────────────────────────────────────────────────────────
无                                FSPC 3极性（positive/implicit/negative）
                                  → 将词典法结果映射到5类与FSPC对齐
```

---

## 4. 各脚本详细说明

### 4.1 `01_parse_lexicons.py` — 解析与合并词典

**输入**：`output/lexicon/FCCPSL.owl`  
**输出**：
- `fccpsl_terms.csv`：14,368 词 × (词、C3类、C2类、C1类)
- `combined_lexicon.csv`：FCCPSL + NTUSD 补充合并，附 source 列

**OWL 解析逻辑**（`xml.etree.ElementTree`）：

```python
# FCCPSL.owl 的 OWL 结构规律：
# 每个C4术语格式：
# <owl:Class rdf:ID="C4_化干戈为玉帛">
#   <rdfs:subClassOf rdf:resource="#C3_ease"/>
# </owl:Class>
#
# 层级映射表（硬编码）：
C3_TO_C2 = {
    'ease': 'pleasure', 'joy': 'pleasure',
    'praise': 'favour', 'like': 'favour', 'faith': 'favour', 'wish': 'favour',
    'peculiar': 'surprise',
    'sorrow': 'sadness', 'miss': 'sadness', 'fear': 'sadness', 'guilt': 'sadness',
    'criticize': 'disgust', 'anger': 'disgust', 'vexed': 'disgust', 'misgive': 'disgust',
}
C2_TO_C1 = {
    'pleasure': 'positive', 'favour': 'positive', 'surprise': 'positive',
    'sadness': 'negative', 'disgust': 'negative',
}
```

**可追溯性记录**：`lexicon_stats.txt`（各类词数、来源词典统计）

### 4.2 `02_sentiment_analyze.py` — 对 1075 首诗做情感分析

**输入**：`00.poems_dataset/poems_dataset_merged_done.csv` + `combined_lexicon.csv`  
**算法**：基于 FCCPSL 的加权词典匹配

**评分机制（改进版，去掉 SnowNLP）**：

```python
# 对每首诗，同时分析"正文"和"赏析"两个字段
# 正文字段：逐字匹配（古汉语，无分词）
# 赏析字段：分词后匹配（现代汉语，用 jieba）
#
# 对每首诗计算 C3 层 15 个情感类的得分：
# score(C3) = Σ(词权重) / 文本长度  （归一化）
#
# 否定词处理（改进版，窗口=2字）：
# 否定词表：不/无/非/未/没/莫/勿/别/休/难/岂
#
# 输出：
# - c3_scores: {ease: 0.12, joy: 0.05, sorrow: 0.34, ...}  15维向量
# - c2_scores: {pleasure: 0.17, sadness: 0.34, ...}         5维向量  
# - c1_scores: {positive: 0.17, negative: 0.34}             2维（极性）
# - dominant_c3: 'sorrow'     最高分C3类
# - dominant_c2: 'sadness'    最高分C2类
# - polarity: 'negative'      正/负/neutral（|pos-neg| < 0.05 视为 neutral）
# - polarity_confidence: 0.73  |pos_score - neg_score| / total
```

**为什么同时保留 C3/C2/C1 三层输出**：
- C3 层（15类）：最细粒度，适合研究分析（区分"sorrow哀情"和"miss思念"）
- C2 层（5类）：中粒度，适合展示（"sadness"类），与论文对齐
- C1 层（极性）：粗粒度，适合与 FSPC 比较评估

**赏析 vs 正文的分析策略**：

| 字段 | 处理方式 | 适用词典 | 理由 |
|------|----------|----------|------|
| 正文（古汉语） | 逐字匹配 | FCCPSL | 诗文字数少（≤50字），逐字最准确 |
| 赏析（现代汉语） | jieba分词后匹配 | FCCPSL + NTUSD | 赏析是现代白话，分词有益 |
| 合并版 | 二者加权（正文0.6+赏析0.4） | 两者 | 正文权重稍高（更直接反映情感） |

### 4.3 `03_evaluate.py` — 用 FSPC 评估准确率

**输入**：FSPC_V1.0.json（需手动下载）+ `combined_lexicon.csv`  
**评估逻辑**：

```
FSPC 标签映射到 C1 极性：
  1 (negative)          → negative
  2 (implicit negative) → negative（词典法弱点）
  3 (neutral)           → neutral
  4 (implicit positive) → positive（词典法弱点）
  5 (positive)          → positive

评估指标：
  - 3类准确率（positive/negative/neutral）
  - 混淆矩阵（特别关注 implicit 类的识别）
  - 词典覆盖率（FCCPSL 覆盖 FSPC 多少词）
```

**关键预期结果**：词典法对 implicit 类（隐性情感）准确率会明显低于显性情感，这正是阶段二（SikuRoBERTa 微调）要改进的依据。

---

## 5. 目录结构

```
step2e_sentiment/
├── step2e_sentiment_detail.md   # 本文件
├── 01_parse_lexicons.py         # 解析 FCCPSL.owl → fccpsl_terms.csv
├── 02_sentiment_analyze.py      # 对 1075 首诗做情感分析
├── 03_evaluate.py               # 在 FSPC 上评估词典法准确率
├── output/
│   ├── lexicon/
│   │   ├── FCCPSL.owl           # 原始 OWL 文件（已下载，14,368词）
│   │   ├── FSPC_V1.0.json       # FSPC 标注语料（需手动下载）
│   │   ├── fccpsl_terms.csv     # 解析后术语表（词+C3+C2+C1）
│   │   ├── combined_lexicon.csv # FCCPSL + NTUSD 合并词典
│   │   └── lexicon_stats.txt    # 词典统计（可追溯）
│   ├── results/
│   │   ├── sentiment_per_poem.csv  # 1075 首诗的情感向量
│   │   ├── flower_sentiment.csv    # 79 种植物的情感画像
│   │   └── summary_report.txt      # 汇总报告
│   ├── evaluation/
│   │   ├── fspc_evaluation.csv     # FSPC 评估结果
│   │   └── evaluation_report.txt   # 准确率/混淆矩阵报告
│   └── figures/
│       ├── sentiment_distribution.png  # 15类情感分布
│       ├── flower_sentiment_heatmap.png # 植物×情感热图
│       └── fspc_confusion_matrix.png    # FSPC 混淆矩阵
```

---

## 6. 当前方法的局限性与阶段二规划

### 6.1 词典法的固有局限

| 局限 | 表现 | 量化预期 |
|------|------|----------|
| 隐性情感盲区 | "枯藤老树昏鸦"→无情感词匹配 | FSPC implicit类 准确率 < 60% |
| 词典词数有限 | 1075首诗中约30-40%的诗情感词覆盖率<3词 | 约20%的诗输出 neutral（无法判断） |
| 语境无感知 | "不喜"被误判为正面（否定词窗口=2，但古诗句法复杂） | 约5-10%误判 |
| 多义词 | "清"在"清清泉水"（正面）和"清冷孤寂"（负面）意义不同 | 难量化 |

### 6.2 阶段二：SikuRoBERTa 情感微调

**方案**：在 FSPC 5000 首标注数据上微调 SikuRoBERTa 做 5 类分类

```
输入：古诗词全文（正文 + 赏析）
模型：SikuRoBERTa + SequenceClassification 头
标签：5类（negative/implicit negative/neutral/implicit positive/positive）
预期 F1：≥ 0.75（参照论文 IJCAI 2019 基线）
```

**与 NER 模型（step2d）共用 SikuRoBERTa 底座**，节省存储和推理资源。

---

## 7. FCCPSL 各 C3 类在古诗词中的典型词汇示例

（从 OWL 解析后人工抽样，用于理解各类含义）

| C3 类 | 对应情绪 | 典型词汇示例 |
|-------|----------|-------------|
| ease | 平和安适 | 归隐田园、风清气爽、花气袭人、周而复始 |
| joy | 欢乐喜悦 | 欢喜、喜悦、喜庆、大快人心 |
| praise | 称颂赞美 | 才华横溢、文采飞扬、名垂青史 |
| like | 喜爱欣赏 | 爱慕、倾心、心仪、赏心悦目 |
| faith | 坚定信念 | 铮铮铁骨、忠心耿耿、矢志不渝 |
| wish | 渴望期盼 | 望眼欲穿、翘首以待、朝思暮想 |
| peculiar | 惊奇感叹 | 叹为观止、拍案惊奇、匪夷所思 |
| sorrow | 悲伤哀痛 | 断肠、泣泪、哀愁、悲恸、伤逝 |
| miss | 思念怀念 | 相思、思念、魂牵梦萦、望断天涯 |
| fear | 恐惧忧惧 | 惶恐、忧惧、惊惶失措、胆战心惊 |
| guilt | 愧疚自责 | 愧悔、自责、负疚、抱愧在心 |
| criticize | 批判指责 | 怒斥、声讨、痛骂、横加指责 |
| anger | 愤怒不平 | 愤慨、激愤、义愤填膺、怒火中烧 |
| vexed | 烦恼苦闷 | 郁郁寡欢、愁肠百结、苦闷彷徨 |
| misgive | 忧虑疑惑 | 忧虑、彷徨、举棋不定、忐忑不安 |

---

## 8. 可追溯性记录汇总

| 步骤 | 记录文件 | 内容 |
|------|----------|------|
| 01 | `output/lexicon/lexicon_stats.txt` | FCCPSL各类词数、NTUSD补充词数、合并重叠情况 |
| 02 | `output/results/summary_report.txt` | 1075首情感分布、植物情感画像、词典覆盖率 |
| 03 | `output/evaluation/evaluation_report.txt` | FSPC三类准确率、混淆矩阵、implicit类分析 |

---

## 9. 当前最需要关注的问题

### 🔴 高优先级

1. **词典覆盖率**（`02`步完成后检查）
   - 如果 1075 首诗中 >30% 的诗覆盖率 < 2 词，词典法整体失效
   - 建议：检查低覆盖率诗词的规律，考虑加入单字情感词（NTUSD 单字条目）

2. **FSPC 手动下载**（`03`步前提）
   - FSPC_V1.0.json 需从 GitHub 下载到 `output/lexicon/`
   - 命令：`wget https://raw.githubusercontent.com/THUNLP-AIPoet/Datasets/master/FSPC/FSPC_V1.0.json -O output/lexicon/FSPC_V1.0.json`

### 🟡 中优先级

3. **C3层 15类 vs 原8维框架的取舍**
   - 对外展示建议用 C2 层 5 类（更直观）
   - 对内分析用 C3 层 15 类（更精细）
   - 未来推荐系统中考虑是否暴露情感维度给用户

4. **正文 vs 赏析的权重比例（0.6:0.4）**
   - 纯经验值，建议在 FSPC 上做敏感性分析（0.5:0.5 / 0.7:0.3 哪个更准）

---

*文档生成时间：2026-03-26*  
*对应代码版本：step2e_sentiment v1.0（01~03脚本初始版）*
