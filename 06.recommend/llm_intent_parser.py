"""
方向 A：用 LLM（Gemini，fallback Doubao）将用户现代汉语输入解析为结构化推荐参数。

解析结果包含：
  scene_key       : 场景键（对应 scene_presets.json 的 key）
  emotion_label   : 情感基调（对应 emotion_lexicon.json 的 key）
  imagery_hints   : 可用于 BM25 的古典意象词列表
  classical_query : 改写为古典风格的搜索短语（直接喂给 BM25）
  confidence      : 解析置信度 0~1

优先级：Gemini（GOOGLE_API_KEY）> Doubao（ARK_API_KEY）> 规则 fallback。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── 系统 Prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是古典诗词推荐系统的意图解析器。将用户现代汉语输入解析为结构化 JSON 推荐参数。

只输出合法 JSON，不要任何说明、注释或代码块标记。格式：
{
  "scene_key": "送别友人" | "表达敬意" | "爱情思念" | "节日庆贺" | "个人欣赏" | null,
  "emotion_label": "积极赞美" | "温暖感谢" | "思念忧郁" | "平静淡然" | "悲凉沉郁" | null,
  "imagery_hints": ["意象1", "意象2", ...],
  "classical_query": "5-10字古典搜索短语",
  "confidence": 0.0~1.0
}

场景选择依据：
  送别友人 → 离别/分别/毕业/出国/远行/饯行/朋友离开
  表达敬意 → 赠给师长/前辈/长辈，称颂品格/高洁/节操/风骨/致敬/退休
  爱情思念 → 爱情/相思/恋人/思念/心上人/暗恋
  节日庆贺 → 节日/生日/祝贺/庆典/过年/周年纪念
  个人欣赏 → 自赏/装饰/收藏/书房/家居/自己喜欢

情感选择依据：
  积极赞美 → 高兴/赞美/欣喜/开心/称赞/欣赏
  温暖感谢 → 感激/感谢/温情/感动/暖心
  思念忧郁 → 思念/惆怅/忧愁/不舍/怀念/感伤
  平静淡然 → 淡然/超脱/平静/清静/豁达
  悲凉沉郁 → 悲伤/沉重/哀痛/压抑/凄凉

imagery_hints 必须是能在古诗词中出现的词语（如「月」「梅」「清风」「孤鸿」「霜」），非现代口语。
classical_query 应能有效检索古诗词意象（如「梅香暗度寒」「月下思故人」）。
"""

# ── 情感标签 → emotion_lexicon key 映射 ──────────────────────────────────
_EMO_TO_LEX: dict[str, str] = {
    "积极赞美": "赞美",
    "温暖感谢": "敬意",
    "思念忧郁": "思念",
    "平静淡然": "平静",
    "悲凉沉郁": "哀伤",
}

# ── 规则 fallback（不依赖 LLM）─────────────────────────────────────────────
_SCENE_RULES: list[tuple[str, list[str]]] = [
    ("送别友人",  ["送别", "离别", "分别", "饯别", "饯行", "毕业", "远行", "出国", "告别"]),
    ("表达敬意",  ["老师", "恩师", "导师", "长辈", "前辈", "师长", "敬意", "品格", "高洁", "退休"]),
    ("爱情思念",  ["爱人", "恋人", "情人", "思念", "相思", "想念", "怀念", "心上人"]),
    ("节日庆贺",  ["节日", "中秋", "春节", "重阳", "生日", "祝贺", "庆贺", "周年"]),
    ("个人欣赏",  ["欣赏", "自赏", "装饰", "摆设", "书房", "家居", "自用", "喜欢"]),
]
_EMO_RULES: list[tuple[str, list[str]]] = [
    ("赞美",  ["赞美", "称颂", "敬佩", "钦佩", "赞叹", "欣赏"]),
    ("敬意",  ["感谢", "感激", "感恩", "致谢", "温情"]),
    ("思念",  ["思念", "想念", "怀念", "不舍", "惆怅", "眷恋"]),
    ("平静",  ["平静", "淡然", "超脱", "豁达", "清静"]),
    ("哀伤",  ["悲伤", "悲凉", "哀痛", "压抑", "凄凉"]),
]


@dataclass
class IntentResult:
    scene_key:       str | None       = None
    emotion_lex_key: str | None       = None   # emotion_lexicon.json 的 key
    imagery_hints:   list[str]        = field(default_factory=list)
    classical_query: str              = ""
    confidence:      float            = 0.5
    from_llm:        bool             = False   # True = LLM 解析；False = 规则 fallback


def _fallback(query: str) -> IntentResult:
    """规则匹配 fallback，不需要 LLM。"""
    q = str(query or "")
    scene = None
    for sk, kws in _SCENE_RULES:
        if any(kw in q for kw in kws):
            scene = sk
            break
    emo = None
    for ek, kws in _EMO_RULES:
        if any(kw in q for kw in kws):
            emo = ek
            break
    return IntentResult(
        scene_key=scene,
        emotion_lex_key=emo,
        imagery_hints=[],
        classical_query="",
        confidence=0.4,
        from_llm=False,
    )


def _call_llm(query: str) -> str | None:
    """
    优先调用 Gemini，失败后尝试 Doubao，均失败返回 None。
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]

    # 优先：Gemini
    try:
        import gemini_client
        if gemini_client.is_available():
            return gemini_client.chat(messages, temperature=0.1, max_tokens=400)
    except Exception as e:
        logger.debug("Gemini 调用失败 (%s)，尝试 Doubao", e)

    # 备选：Doubao / Ark
    try:
        import llm_client
        if llm_client.is_available():
            return llm_client.chat(messages, temperature=0.1, max_tokens=400)
    except Exception as e:
        logger.debug("Doubao 调用失败 (%s)", e)

    return None


def parse(query: str) -> IntentResult:
    """
    调用 LLM 解析用户意图；失败时退回规则 fallback。

    优先级：Gemini（GOOGLE_API_KEY）> Doubao（ARK_API_KEY）> 规则 fallback。
    """
    raw = _call_llm(query)
    if raw is None:
        logger.debug("所有 LLM 均不可用，使用规则 fallback")
        return _fallback(query)

    try:
        # 去掉可能的 markdown 代码块标记
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        data: dict[str, Any] = json.loads(raw)
    except Exception as e:
        logger.warning("LLM 返回内容解析失败（%s），退回规则 fallback", e)
        return _fallback(query)

    emo_label = data.get("emotion_label")
    emo_lex = _EMO_TO_LEX.get(emo_label or "", None)  # type: ignore[arg-type]

    return IntentResult(
        scene_key=data.get("scene_key"),
        emotion_lex_key=emo_lex,
        imagery_hints=data.get("imagery_hints", []),
        classical_query=data.get("classical_query", ""),
        confidence=float(data.get("confidence", 0.7)),
        from_llm=True,
    )
