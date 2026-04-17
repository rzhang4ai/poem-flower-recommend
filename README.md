# 古诗词 × 花卉推荐系统
# Classical Chinese Poetry × Flower Recommendation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://poem-flower-recommend.streamlit.app)

> 输入心情或场合，系统从 1,075 首古典咏花诗词中为你推荐最贴切的一首。  
> Describe your mood or occasion — the system recommends the most fitting flower poem from a corpus of 1,075 classical Chinese poems.

**[🌺 Live Demo](https://poem-flower-recommend.streamlit.app)**

---

## 项目简介 / Overview

本系统以《中国历代咏花诗词鉴赏辞典》为核心语料，结合《诗学含英》意象词典，构建了一个多维度古诗推荐框架。系统融合了规则、统计与语言模型方法，对每首诗进行情感分类与意象标注，最终通过多通道匹配与 RRF 融合排序呈现推荐结果。

This system uses the *Encyclopaedia of Chinese Flower Poetry through the Ages* as its core corpus, combined with imagery from the classical lexicon *Shixue Hanying*. It builds a multi-dimensional poem recommendation framework by integrating rule-based, statistical, and language-model approaches — performing sentiment classification and imagery annotation on each poem, then ranking results via multi-channel matching and Reciprocal Rank Fusion (RRF).

涵盖 **1,075 首诗词 · 79 种花卉 · 宋代为主，兼收唐、元、明、清**。

Covers **1,075 poems · 79 flower types · primarily Song dynasty, with Tang, Yuan, Ming, and Qing**.

---

## 系统架构 / Architecture

```
用户输入（心情 / 场合 / 意象）
User input: mood / occasion / imagery
           │
           ▼
  LLM 意图解析（Gemini）
  LLM Intent Parsing
           │
     ┌─────┴───────────────────────────┐
     ▼         ▼           ▼           ▼
  BM25       情感余弦     9维语义      嵌入相似度
  意象词匹配  Sentiment   Dim Vector   Embedding
  BM25       Cosine      Similarity   Similarity
     └─────┬───────────────────────────┘
           ▼
    RRF 融合排序（动态权重归一化）
    RRF Fusion Ranking
           │
           ▼
    推荐结果 + 意象词云
    Results + Imagery Word Cloud
```

**数据管线 / Data Pipeline**

```
《中国历代咏花诗词鉴赏辞典》 ──数字化+人工核对──▶ poems_dataset_merged_done.csv
《诗学含英》 ────────────────数字化+清洗──────▶ 诗学含英_simp_clean.txt
                                    │
                          情感标注 + 意象词标注
                          Sentiment + Imagery Annotation
                                    │
                                    ▼
                    poems_structured_shangxi_wip.csv
                                    │
                                    ▼
                          推荐系统 / Recommender
```

---

## 核心资源 / Core Resources

### 数据集 / Datasets

详见 [DATA_POLICY.md](DATA_POLICY.md) 了解字段说明与使用条款。  
See [DATA_POLICY.md](DATA_POLICY.md) for field descriptions and usage terms.

| 文件 / File | 说明 / Description | 规模 |
|---|---|---|
| [`00.poems_dataset/poems_dataset_merged_done.csv`](00.poems_dataset/poems_dataset_merged_done.csv) | 数字化诗词主数据集（含人工核对） Digitized poem corpus with manual verification | 1,075 首 |
| [`05.imagery_labels/sxhy_imagery/诗学含英_simp_clean.txt`](05.imagery_labels/sxhy_imagery/诗学含英_simp_clean.txt) | 《诗学含英》意象词典数字化版本 Digitized imagery lexicon from *Shixue Hanying* | ~120 KB |
| [`07.final_labels/poems_structured_shangxi_wip.csv`](07.final_labels/poems_structured_shangxi_wip.csv) | 含情感分类与意象标注的最终结构化数据 Final structured dataset with sentiment labels and imagery annotations | 1,075 首 |

### 情感分类模型 / Sentiment Model

我们基于 [CCPoemBERT](https://huggingface.co/SIKU-BERT/ccpoem-bert) 微调了一个古诗词情感分类器（5极性），训练数据与流程仍有优化空间。**诚邀对古诗词 NLP 感兴趣的研究者联系我们，共同改进模型训练。**  
We fine-tuned a 5-polarity sentiment classifier for classical Chinese poetry based on [CCPoemBERT](https://huggingface.co/SIKU-BERT/ccpoem-bert). The training data and pipeline have known limitations and room for improvement. **We warmly welcome researchers interested in classical Chinese poetry NLP to reach out and collaborate on improving the model.**

模型文件暂不公开，待优化后将发布至 HuggingFace。  
Model weights are not publicly released at this time; they will be published to HuggingFace after further optimization.

---

## 本地运行 / Run Locally

```bash
git clone https://github.com/rzhang4ai/poem-flower-recommend.git
cd poem-flower-recommend
pip install -r requirements.txt

# 配置 Gemini API Key（可在 Google AI Studio 免费获取）
cp 06.recommend/.env.example 06.recommend/.env
# 编辑 .env，填入 GOOGLE_API_KEY

streamlit run 05.recommend/app.py
```

---

## 数据使用说明 / Data Usage

本项目的三个核心数据集（见上表）**可用于学术和非商业用途**。使用前请发送邮件至 **rzhang4ai@gmail.com** 简单说明用途，并在您的项目或论文中注明来源。  
The three core datasets listed above are available for **academic and non-commercial use**. Before using, please send a brief email to **rzhang4ai@gmail.com** describing your intended use. Please also cite this project in your work.

完整使用条款见 [DATA_POLICY.md](DATA_POLICY.md)。  
Full terms in [DATA_POLICY.md](DATA_POLICY.md).

---

## 致谢 / Acknowledgements

- **CCPoemBERT / SIKU-BERT**：情感分类模型基础，感谢 [SIKU-BERT 项目](https://huggingface.co/SIKU-BERT)。  
  Sentiment model base — thanks to the [SIKU-BERT project](https://huggingface.co/SIKU-BERT).
- **《中国历代咏花诗词鉴赏辞典》**：核心诗词语料来源。
- **《诗学含英》（清·李渔辑）**：意象词典来源，本项目完成了其数字化与结构化整理。  
  Imagery lexicon source — this project digitized and structured the text.

---

## 引用 / Citation

如果本项目的数据或代码对您的研究有所帮助，请引用：  
If this project's data or code is useful in your research, please cite:

```
Zhang, R. et al. (2026). Classical Chinese Poetry × Flower Recommendation System.
GitHub. https://github.com/rzhang4ai/poem-flower-recommend
```

---

## 联系 / Contact

**rzhang4ai@gmail.com**

欢迎就数据合作、模型改进、或相关研究领域（花卉意象、诗词情感分析、古典文学 NLP）联系交流。  
Feel free to reach out regarding data collaboration, model improvement, or related research areas (floral imagery, classical poetry sentiment analysis, literary NLP).
