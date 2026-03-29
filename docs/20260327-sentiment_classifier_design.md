# 古诗词情感分层分类器：设计文档

> 文件路径：`02.sample_label_phase2/step2e_sentiment/07_step6_final_clf.py`
> 撰写日期：2026-03-27

---

## 一、项目背景与目标

本分类器是「花卉诗词推荐系统」数据标注管道的一部分，目标是对古典汉语诗词进行**可追溯、可验证的细粒度情感标注**，以替代早期依赖规则引擎（AI直写、不可追溯）的粗放方案。

设计的核心约束：

- **可追溯性**：每一层的预测逻辑都来自公开资源（FSPC 标注数据 + FCCPSL 情感词典），而非黑盒规则。
- **灰盒可验证**：SVM 是线性模型，权重可检查；词典分数是确定性规则，可人工复现。
- **分层细粒度**：支持从粗（5 极性）到细（15 C3 情感类型）的多粒度输出，满足推荐系统不同使用场景的需求。

---

## 二、分类器架构（三层级联）

```
输入：古诗正文（字符串）
         │
         ▼
┌─────────────────────────────────────────────┐
│  Layer 1：5极性分类                          │
│  模型：端到端微调 BERT-CCPoem                 │
│  输出：Negative / Implicit Negative /        │
│        Neutral / Implicit Positive / Positive│
└──────────────────┬──────────────────────────┘
                   │  若 L1 = Neutral → L2/L3 置空
                   ▼
┌─────────────────────────────────────────────┐
│  Layer 2：C2 粗粒度分类（3类）               │
│  模型：Linear SVM                            │
│  特征：微调BERT CLS (512维) +                │
│        FCCPSL 词典 IDF 得分 (15维) = 527维   │
│  输出：positive / neg_sorrow / neg_anger     │
│  约束：必须与 L1 极性方向一致                 │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Layer 3：C3 细粒度分类（10类）              │
│  模型：Linear SVM（同 L2 特征）              │
│  输出：praise / like / faith / joy / ease /  │
│        sorrow / miss / criticize /           │
│        vexed / misgive                       │
│  约束：必须在 L2 允许的子集内（硬约束）       │
└─────────────────────────────────────────────┘
```

### 一致性约束规则

| L1 极性 | L2 允许值 | L3 允许值 |
|--------|----------|----------|
| Positive / Implicit Positive | positive | praise, like, faith, joy, ease, wish |
| Negative / Implicit Negative | neg_sorrow, neg_anger | sorrow, miss / criticize, vexed, misgive |
| Neutral | n/a | n/a |

若 SVM 预测结果违反约束，则在约束允许的类别子集内取概率最高的类别，保证三层输出逻辑自洽。

---

## 三、使用的数据与资源

### 3.1 FSPC（细粒度情感诗词语料库）

- **来源**：清华大学 NLP 实验室，[GitHub](https://github.com/THUNLP-AIPoet/Datasets/tree/master/FSPC)
- **内容**：5000 首古典汉语诗词，每首人工标注 5 类极性标签
  - Negative / Implicit Negative / Neutral / Implicit Positive / Positive
- **用途**：
  - 微调 BERT-CCPoem（L1 模型训练数据）
  - 提供伪标签逻辑验证的基准极性
  - L1 SVM 基线的训练/测试集

### 3.2 FCCPSL（细粒度古典汉语诗词情感词典）

- **来源**：Weiiiing 整理，[GitHub](https://github.com/Weiiiing/poetry-sentiment-lexicon)
- **内容**：约 14,369 个情感词条，标注至 15 个 C3 细粒度类别（anger, criticize, ease, faith, fear, guilt, joy, like, misgive, miss, peculiar, praise, sorrow, vexed, wish）及对应的 DUTIR C1/C2 上位分类
- **用途**：
  - 远程监督伪标签生成（黄金数据集构建）
  - SVM 特征工程（15 维 IDF 加权词典得分）

### 3.3 BERT-CCPoem

- **来源**：清华大学 NLP 实验室，[HuggingFace](https://huggingface.co/THUNLP/bert-ccpoem)
- **内容**：在大规模古典汉语诗词语料上预训练的 BERT 模型（hidden_size=512）
- **用途**：
  - L1 微调基座（`ccpoem_sentiment_ft`，接5类分类头后在 FSPC 上微调）
  - L2/L3 SVM 的语义特征来源（`[CLS]` 向量）

### 3.4 黄金数据集（Golden Dataset）

- **构建方式**：远程监督（Distant Supervision）
  1. 用 FCCPSL 词条对 FSPC 5000 首诗进行子串匹配
  2. 否定词处理（不/无/莫/非/未/休/勿 前置则作废该匹配）
  3. IDF 加权计分：得分 = Σ (命中次数 × 1/log₂(|该类词数|+1))
  4. 纯净性过滤：最高分类别唯一且得分 > 0 才接受
  5. 逻辑验证：伪标签与 FSPC 原极性对照，冲突则丢弃
- **规模**：1896 首（占 FSPC 的 37.9%）
- **词长参数**：MIN_WORD_LEN=2，MAX_WORD_LEN=2，保留单字白名单（愁悲哀泪怨苦痛喜乐欢悦思念忆）

---

## 四、训练方法与实验流程

整个管道由 7 个有序脚本构成：

| 脚本 | 功能 |
|------|------|
| `07_step1_pseudo_labeling.py` | 远程监督生成黄金数据集 |
| `07_step3_split.py` | 固定分层切分（FSPC 80/20，黄金集 80/20，seed=42） |
| `07_step4a_svm_5pol.py` | 基础 BERT CLS + SVM 5极性（基线） |
| `07_step4b_bert_5pol_eval.py` | 微调 BERT 直接推断 5 极性（对比） |
| `07_step4c_svm_5pol_ftcls.py` | 微调 BERT CLS + SVM 5 极性（补充对比） |
| `07_step5b_custom_c2_svm.py` | C2 语义重构方案对比（路线三） |
| `07_step5c_feature_ablation.py` | 特征消融实验（4方案 × 2任务） |
| `07_step6_final_clf.py` | **最终分层分类器推断** |

### 关键决策点

**L1 模型选择：微调 BERT 胜出**

| 方法 | Macro-F1 | 备注 |
|------|---------|------|
| SVM（基础 BERT CLS） | 0.3579 | 无 FSPC 训练信息 |
| SVM（微调 BERT CLS） | 0.5659 | 使用了 FSPC 特征 |
| 端到端微调 BERT | **0.6597** | 有数据泄露风险，真实值略低 |

**L2/L3 特征选择：微调 CLS + 词典得分（F4）全面最优**

| 特征方案 | 维度 | C2 Macro-F1 | C3 Macro-F1 |
|---------|-----|------------|------------|
| F1 基础 CLS | 512 | 0.5690 | 0.2252 |
| F2 微调 CLS | 512 | 0.6361 | 0.2872 |
| F3 基础 CLS + 词典 | 527 | 0.8909 | 0.8798 |
| **F4 微调 CLS + 词典** | **527** | **0.9345** | **0.8963** |

词典分数是绝对主力（对 C3 单独贡献 +0.65），微调 CLS 在其基础上再贡献约 +0.02~0.04。

**C2 标签体系重构（路线三）**

原始 DUTIR 4类（pleasure/favour/surprise/sadness/disgust）语义边界模糊，平均 Macro-F1 仅 0.44。重构为 3 类语义更清晰的方案后：

| C2 方案 | Macro-F1 |
|--------|---------|
| 旧 DUTIR 4类 | 0.4360 |
| 语义重构 4类（pos_praise / pos_joy / neg_sorrow / neg_anger） | 0.4321 |
| **合并正向 3类（positive / neg_sorrow / neg_anger）** | **0.5699** |

---

## 五、性能指标汇总

> 以下指标均在固定测试集（seed=42 的 stratified 20% 切分）上评估。
> C2/C3 使用伪标签作为代理指标（pseudo-label accuracy），并非人工标注 ground truth。

| 层级 | 任务 | Macro-F1 | 备注 |
|------|-----|---------|------|
| L1 | 5极性 | 0.6597 | 微调BERT，含数据泄露 |
| L2 | C2（3类粗粒度） | 0.9345 | 伪标签代理指标，偏乐观 |
| L3 | C3（10类细粒度） | 0.8963 | 伪标签代理指标，偏乐观 |

---

## 六、潜在风险与局限性

### 6.1 数据泄露（L1）

微调 BERT 在 FSPC **全量 5000 首**上训练，而测试集来自同一数据集的 20% 切分，严格意义上属于近似对比（approximate comparison）而非盲测。实际泛化性能可能比报告值低 0.03~0.08。

### 6.2 伪标签循环验证（L2/L3 最严重）

L2/L3 的 SVM 使用 FCCPSL 词典得分作为特征，而训练标签（黄金数据集的伪标签）也是由同一词典匹配生成的。**词典得分对自身生成的标签预测准确率天然偏高**，导致 F3/F4 方案的 0.89~0.93 指标存在显著虚高。

**实际使用时 L2/L3 的泛化精度会明显低于报告值**，特别是对词典覆盖率低的诗词（大量生僻用词、隐喻类诗词）。

### 6.3 词典覆盖率有限

FCCPSL 词典按照 MIN_WORD_LEN=2 / MAX_WORD_LEN=2 过滤后，对于以单字意象为主的简短古诗（如五言绝句）覆盖率较低，此时 L2/L3 的词典分数维度近似全零，退化为纯 CLS 特征，性能接近 F2（Macro-F1 约 0.63/0.29）。

### 6.4 黄金数据集覆盖率有限

黄金数据集仅包含 1896/5000 首（37.9%）FSPC 诗词，且偏向情感词汇明确、密度较高的诗作。对于情感含蓄、以意境取胜的古诗（如禅意诗、山水诗），伪标签本身质量就较低，进而影响 SVM 的训练质量。

### 6.5 C3 类别不均衡

黄金数据集中，10 个有效 C3 类别的样本量从 8（miss，思念）到 88（praise，称颂）差距悬殊。样本量少的类别（miss, misgive, ease, faith）即使在测试集上 F1 较高，也可能因过拟合或测试样本极少（8条）导致结果不稳定。

### 6.6 历史领域偏差

BERT-CCPoem 在古典诗词语料上预训练，FCCPSL 词典也以古典汉语词汇为主，两者对**现代汉语诗歌**的迁移能力未经验证，不建议直接用于现代诗情感分类。

### 6.7 缺乏人工标注的 C2/C3 测试集

目前 C2/C3 没有独立的人工标注测试集，无法做真正意义上的精度评估。如需严格验证，建议人工标注约 200 条 C3 样本作为外部测试集。

---

## 七、如何开放给他人使用

### 7.1 GitHub 授权方式

**推荐授权协议（根据你的意图选择）：**

| 场景 | 推荐 License |
|------|------------|
| 学术引用，允许任意使用 | MIT License |
| 允许使用但要求同等开放（Copyleft） | Apache 2.0 |
| 仅限学术研究，禁止商业用途 | CC BY-NC 4.0（配合说明文档） |
| 私有授权，逐人审批 | 不添加 License，仓库设 Private，手动邀请 Collaborator |

> 注意：FSPC 和 BERT-CCPoem 均来自清华大学，FCCPSL 来自第三方开源项目，**发布时需在 README 中明确标注这些依赖的来源和各自的原始许可证**，避免侵权。

**操作步骤：**

1. 在仓库根目录新增 `LICENSE` 文件（GitHub 可一键生成）
2. 在 `README.md` 中列出第三方资源来源与版权声明
3. 若需逐人审批，将仓库设为 Private → Settings → Collaborators → 发邀请链接

### 7.2 将分类器封装为 API

推荐两种方案，按复杂度排列：

---

#### 方案A（最简）：FastAPI + uvicorn 本地/云端部署

**新建 `api/app.py`：**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "02.sample_label_phase2/step2e_sentiment"))
from 07_step6_final_clf import HierarchicalSentimentClassifier  # noqa

app = FastAPI(title="古诗词情感分类 API")
clf = HierarchicalSentimentClassifier()   # 启动时加载，一次性

class PoemRequest(BaseModel):
    text: str

@app.post("/classify")
def classify(req: PoemRequest):
    result = clf.predict_one(req.text)
    return {
        "l1_polarity":    result["l1_polarity"],
        "l1_polarity_zh": result["l1_polarity_zh"],
        "l2_c2":          result["l2_c2"],
        "l2_c2_zh":       result["l2_c2_zh"],
        "l3_c3":          result["l3_c3"],
        "l3_c3_zh":       result["l3_c3_zh"],
    }
```

**安装依赖并启动：**

```bash
pip install fastapi uvicorn
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**调用示例：**

```bash
curl -X POST http://localhost:8000/classify \
     -H "Content-Type: application/json" \
     -d '{"text": "春风又绿江南岸明月何时照我还"}'
```

---

#### 方案B（推荐分发）：HuggingFace Spaces + Gradio

适合让他人无需安装环境即可在线试用：

1. 在 [HuggingFace Spaces](https://huggingface.co/spaces) 新建 Space（选 Gradio SDK）
2. 上传模型文件（`ccpoem_sentiment_ft` + 两个 `.pkl`）和推断代码
3. 新建 `app.py`（Gradio 界面），核心代码：

```python
import gradio as gr
from classifier import HierarchicalSentimentClassifier

clf = HierarchicalSentimentClassifier()

def classify(text):
    r = clf.predict_one(text)
    return r["l1_polarity_zh"], r["l2_c2_zh"], r["l3_c3_zh"]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="输入古诗正文"),
    outputs=[
        gr.Textbox(label="L1 极性"),
        gr.Textbox(label="L2 粗粒度情感"),
        gr.Textbox(label="L3 细粒度情感"),
    ],
    title="古诗词情感分层分类器",
)
demo.launch()
```

HuggingFace Spaces 免费公开托管，天然支持授权（设为 Private Space 可限制访问者）。

---

#### 模型文件注意事项

发布前请确认：

| 文件 | 是否可直接上传 GitHub |
|-----|------------------|
| `svm_c2_ablation_best.pkl` | ✓（约 1.6MB） |
| `svm_c3_ablation_best.pkl` | ✓（约 4.4MB） |
| `models/ccpoem_sentiment_ft/` | ✗ 体积过大（建议托管在 HuggingFace Hub，GitHub 仅存引用路径） |
| `output/svm_models/ft_cls_features.npy` | ✗ 无需发布（推断时实时计算） |

建议在 `.gitignore` 中排除大文件，在 README 中说明如何下载模型权重。

---

## 八、推荐引用格式

如他人使用本分类器，建议引用以下原始资源：

```
1. FSPC: Zhipeng Gao et al. (2020). A Large-Scale Chinese Short-Text Conversation Dataset.
   https://github.com/THUNLP-AIPoet/Datasets/tree/master/FSPC

2. BERT-CCPoem: THUNLP-AIPoet.
   https://huggingface.co/THUNLP/bert-ccpoem

3. FCCPSL: Weiiiing.
   https://github.com/Weiiiing/poetry-sentiment-lexicon
```
