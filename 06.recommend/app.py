"""
古诗花卉推荐 · 多轮对话式界面

运行（项目根目录）：
  source flower_env/bin/activate
  streamlit run 05.recommend/app.py

LLM 配置（写入 05.recommend/.env 或在终端 export）：
  GOOGLE_API_KEY=your-key
  GEMINI_CHAT_MODEL=gemini-3-flash-preview
  GEMINI_EMBED_MODEL=gemini-embedding-2-preview

嵌入（一次性离线，可选）：
  python 05.recommend/embed_shangxi_gemini.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import streamlit as st

from recommend import load_data_and_index, recommend

# ── 常量 ────────────────────────────────────────────────────────────────────

# 5 极性情感卡片：直接对应分类器标签
# prob_l1 向量顺序：[negative, implicit_negative, neutral, implicit_positive, positive]
POLARITY_OPTIONS: list[tuple[str, str, str]] = [
    ("positive",          "很棒",   "positive.png"),
    ("implicit_positive", "还 ok",  "implicit_positive.png"),
    ("neutral",           "一般般", "neutral.png"),
    ("implicit_negative", "不舒服", "implicit_negative.png"),
    ("negative",          "不好",   "negative.png"),
]
POLARITY_VECS: dict[str, list[float]] = {
    "positive":          [0.01, 0.02, 0.02, 0.10, 0.85],
    "implicit_positive": [0.02, 0.05, 0.12, 0.74, 0.07],
    "neutral":           [0.05, 0.10, 0.70, 0.10, 0.05],
    "implicit_negative": [0.05, 0.74, 0.12, 0.06, 0.03],
    "negative":          [0.75, 0.15, 0.05, 0.03, 0.02],
}
_ICON_DIR = HERE / "assets" / "emotions"
_WC_DIR   = HERE.parent / "04.visualization" / "output" / "wordcloud"

# 送人场景 → scene_key
GIFT_SCENES: set[str] = {"送别友人", "表达敬意", "爱情思念", "节日庆贺"}
SELF_SCENES: set[str] = {"个人欣赏"}

# 关系快捷按钮 → (唯一id, 显示标签, scene_key)
# scene_key == "__self__" 表示切换到自用流程
RELATION_BUTTONS: list[tuple[str, str, str]] = [
    ("领导",   "🏢 领导",  "表达敬意"),
    ("师长",   "🎓 师长",  "表达敬意"),
    ("朋友",   "🤝 朋友",  "送别友人"),
    ("爱人",   "💌 爱人",  "爱情思念"),
    ("晚辈",   "🌱 晚辈",  "表达敬意"),
    ("节日",   "🎉 节日",  "节日庆贺"),
    ("自己",   "🌸 送自己", "__self__"),
]

_GIFT_KWS = [
    # 动词短语
    "送给", "赠给", "赠送", "送人", "要送", "想送", "准备送", "打算送",
    "买花", "送花", "送束", "送朵",
    # 关系人
    "老师", "导师", "教授", "领导", "上司", "老板",
    "朋友", "同学", "同事",
    "恋人", "爱人", "伴侣", "情人", "男朋友", "女朋友",
    "父母", "爸爸", "妈妈",
    # 场合
    "礼物", "生日", "毕业", "退休", "出国", "纪念日",
]
_SELF_KWS  = ["自己", "我想", "我的", "此刻", "心情", "感觉", "今晚",
              "最近", "一首", "我在", "我最近", "我现在"]

_WELCOME = (
    "你好！描述一下此刻的心情、场合，或想在诗里看到什么——"
    "我来帮你找最贴切的花诗。"
)

# ── 数据加载 ─────────────────────────────────────────────────────────────────

@st.cache_data
def _load():
    return load_data_and_index()


# ── 意图解析 ─────────────────────────────────────────────────────────────────

def _parse_intent(query: str):
    try:
        import llm_intent_parser
        return llm_intent_parser.parse(query)
    except Exception:
        return None


def _classify(intent, query: str) -> str:
    """
    返回 'gift' | 'self' | 'ambiguous'。
    规则关键词优先（更可靠），LLM scene_key 仅在关键词无法区分时作为参考。
    """
    has_gift = any(kw in query for kw in _GIFT_KWS)
    has_self = any(kw in query for kw in _SELF_KWS)
    if has_gift and not has_self:
        return "gift"
    if has_self and not has_gift:
        return "self"
    # 关键词两者都有或都没有 → 参考 LLM
    if intent and intent.scene_key in GIFT_SCENES:
        return "gift"
    if intent and intent.scene_key in SELF_SCENES:
        return "self"
    return "ambiguous"


# 用户文本中明确提到的关系词 → 才认为关系已知（不依赖 LLM 推断）
_EXPLICIT_RELATION_KWS = [
    "老师", "导师", "教授", "领导", "上司", "老板",
    "朋友", "同学", "同事",
    "恋人", "爱人", "伴侣", "情人", "男友", "女友", "男朋友", "女朋友",
    "父母", "爸爸", "妈妈", "爷爷", "奶奶", "长辈",
    "晚辈", "学生", "孩子", "儿子", "女儿",
]

def _gift_has_relation(query: str) -> bool:
    """仅当用户文本中明确出现关系词时，才认为关系已知；LLM 推断不算。"""
    return any(kw in query for kw in _EXPLICIT_RELATION_KWS)


# 用户文本中明确的情绪词 → 才跳过情感卡片
_EXPLICIT_EMO_KWS = [
    "开心", "快乐", "高兴", "愉快", "欢喜", "喜悦",
    "忧郁", "忧愁", "悲伤", "悲", "愁", "难过", "伤心",
    "平静", "淡然", "沉静",
    "沉重", "压抑", "痛苦",
    "思念", "想念", "念",
    "感动", "温暖", "幸福",
    "不舒服", "难受",
    "烦", "焦虑", "迷茫", "彷徨",
    "释然", "满足", "踏实",
    "激动", "兴奋",
    "很棒", "还ok", "一般般",
]

def _has_emotion_signal(query: str) -> bool:
    """仅检查用户原文是否有明确情感词；不依赖 LLM 推断，避免误判。"""
    return any(kw in query for kw in _EXPLICIT_EMO_KWS)


# ── 推荐调用 ─────────────────────────────────────────────────────────────────

def _do_recommend(df, dim_kw, emo, scenes,
                  query: str, intent,
                  polarity_key: str | None = None,
                  scene_key_override: str | None = None) -> tuple[list, dict]:
    emo_vec = POLARITY_VECS.get(polarity_key) if polarity_key else None
    effective_scene = scene_key_override or (intent.scene_key if intent else None)
    return recommend(
        df, dim_kw, emo, scenes,
        query,
        intent=intent,
        scene_key=effective_scene,
        emotion_vec_override=emo_vec,
        top_k=3,
        max_per_flower=2,
    )


# ── 渲染组件 ─────────────────────────────────────────────────────────────────

def _render_polarity_cards() -> None:
    """5 极性卡片：图标 + 按钮，点击立即 rerun 触发流程推进。"""
    cols = st.columns(5)
    for col, (key, label, icon_file) in zip(cols, POLARITY_OPTIONS):
        with col:
            icon_path = _ICON_DIR / icon_file
            if icon_path.exists():
                st.image(str(icon_path), use_container_width=True)
            is_active = st.session_state.get("pending_polarity") == key
            if st.button(label, key=f"pol_{key}",
                         type="primary" if is_active else "secondary",
                         use_container_width=True):
                st.session_state.pending_polarity = key
                st.rerun()          # 立即触发 rerun，让 pending 被消费


def _render_relation_buttons() -> None:
    """关系快捷按钮（两行），点击立即 rerun。用唯一 id 作按钮 key。"""
    row1 = RELATION_BUTTONS[:4]   # 领导 师长 朋友 爱人
    row2 = RELATION_BUTTONS[4:]   # 晚辈 节日 送自己
    for row in (row1, row2):
        cols = st.columns(len(row))
        for col, (uid, label, scene_key) in zip(cols, row):
            with col:
                is_active = st.session_state.get("pending_relation") == uid
                if st.button(label, key=f"rel_{uid}",
                             type="primary" if is_active else "secondary",
                             use_container_width=True):
                    # 用 uid 暂存，区分同 scene_key 的不同关系
                    st.session_state.pending_relation = uid
                    st.rerun()


def _render_clarify_buttons() -> None:
    """场合模糊时的澄清按钮：送人 / 自己欣赏。"""
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🎁 送给别人", key="clarify_gift", use_container_width=True):
            st.session_state.pending_clarify = "gift"
            st.rerun()
    with c2:
        if st.button("🌸 自己欣赏", key="clarify_self", use_container_width=True):
            st.session_state.pending_clarify = "self"
            st.rerun()


def _render_recommendations(results: list[dict], signals: dict) -> None:
    if not results:
        st.warning("暂无匹配结果，请补充更多描述。")
        return

    # ── 整体匹配信号摘要 ────────────────────────────────────────────────
    tags: list[str] = []
    if signals.get("from_llm"):
        if signals.get("detected_scene"):
            tags.append(f"场景：{signals['detected_scene']}")
        if signals.get("emo_hit") and signals["emo_hit"] != "（精调指定）":
            pol_label = next((l for k, l, _ in POLARITY_OPTIONS
                               if k == signals["emo_hit"]), signals["emo_hit"])
            tags.append(f"情感：{pol_label}")
        if signals.get("classical_query"):
            tags.append(f"古典短语：「{signals['classical_query']}」")
    if signals.get("qtok"):
        tags.append(f"关键词：{' · '.join(signals['qtok'][:5])}")
    weights = signals.get("weights", {})
    active_w = [f"{k}={round(v * 100)}%" for k, v in weights.items() if v > 0.01]
    if active_w:
        tags.append(f"权重：{', '.join(active_w)}")
    if tags:
        badge = "🤖" if signals.get("from_llm") else "📋"
        st.caption(f"{badge} {'　'.join(tags)}")

    # ── 每首诗：双列卡片 ─────────────────────────────────────────────────
    for i, r in enumerate(results, 1):
        flower  = r.get("花名", "")
        wc_path = _WC_DIR / f"wordcloud_{flower}.png"

        st.markdown("---")

        # 标题行
        st.markdown(
            f"**{i}. 《{r.get('诗名', '')}》**"
            f"　{r.get('作者', '')}（{r.get('朝代', '')}）"
            f"　🌸 *{flower}*"
        )

        # 双列：左正文，右词云
        col_text, col_wc = st.columns([3, 2], gap="medium")

        with col_text:
            full_text = r.get("全文", "") or r.get("正文摘录", "")
            if full_text:
                # 保留原始换行格式
                lines = [l.strip() for l in full_text.splitlines() if l.strip()]
                st.markdown("\n\n".join(lines))
            # 判断逻辑以小号字呈现
            note = r.get("说明", "")
            if note:
                st.caption(note)

        with col_wc:
            if wc_path.exists():
                st.image(str(wc_path), use_container_width=True,
                         caption=f"{flower} 意象词云")
            else:
                st.caption(f"（暂无「{flower}」词云）")


# ── 侧边栏 ───────────────────────────────────────────────────────────────────

def _sidebar() -> None:
    with st.sidebar:
        st.markdown("## ⚙️ 配置")

        gemini_ok = False
        doubao_ok = False
        try:
            import gemini_client
            gemini_ok = gemini_client.is_available()
        except Exception:
            pass
        if not gemini_ok:
            try:
                import llm_client
                doubao_ok = llm_client.is_available()
            except Exception:
                pass

        if gemini_ok:
            st.success("🟢 LLM 已就绪（Gemini）")
        elif doubao_ok:
            st.success("🟢 LLM 已就绪（Doubao fallback）")
        else:
            st.error("🔴 LLM 未配置（规则降级模式）")
            with st.expander("如何配置？"):
                st.markdown(
                    "在 `05.recommend/.env` 写入：\n"
                    "```\n"
                    "GOOGLE_API_KEY=your-key\n"
                    "GEMINI_CHAT_MODEL=gemini-3-flash-preview\n"
                    "GEMINI_EMBED_MODEL=gemini-embedding-2-preview\n"
                    "```\n"
                    "在 [Google AI Studio](https://aistudio.google.com/apikey) 免费获取 Key，"
                    "然后重启 Streamlit。"
                )

        st.divider()

        emb_path = HERE / "output" / "poems_embeddings.npy"
        emb_ids  = HERE / "output" / "poems_embed_ids.json"
        if emb_path.exists() and emb_ids.exists():
            try:
                import numpy as _np
                _n = len(json.loads(emb_ids.read_text(encoding="utf-8")))
                _d = _np.load(str(emb_path)).shape[1]
                st.success(f"🟢 嵌入已就绪（{_n} 首 × {_d} 维）")
            except Exception:
                st.success("🟢 嵌入向量已就绪")
        else:
            st.warning("🟡 嵌入未生成（嵌入通道权重=0）")
            with st.expander("如何生成？"):
                st.markdown(
                    "一次性离线步骤，中断可续跑：\n"
                    "```bash\n"
                    "source flower_env/bin/activate\n"
                    "python 05.recommend/embed_shangxi_gemini.py\n"
                    "# 续跑：\n"
                    "python 05.recommend/embed_shangxi_gemini.py --resume\n"
                    "```\n"
                    "完成后重启 Streamlit 即可。"
                )

        st.divider()
        st.markdown("**数据** · 1075 首诗 · 79 种花")
        st.markdown("**通道** · BM25 + 情感余弦 + 9维dim + 嵌入")
        st.markdown("**来源** · CCPoem-BERT · 诗学含英 · SikuRoBERTa")

        if st.button("🔄 重置对话", use_container_width=True):
            for k in ["history", "phase", "intent", "last_query",
                      "accumulated_query", "pending_polarity",
                      "pending_relation", "pending_clarify", "polarity_key"]:
                st.session_state.pop(k, None)
            st.rerun()


# ── 主逻辑 ───────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="诗花推荐", layout="centered", page_icon="🌺")
    _sidebar()
    st.title("🌺 古诗词 × 花卉")

    df, dim_kw, emo, scenes = _load()

    # ── session state 初始化 ─────────────────────────────────────────────
    defaults: dict[str, Any] = {
        "history":           [],   # [{role, type, content, results?, signals?}]
        # 阶段：start | awaiting_clarify | awaiting_gift_who |
        #        awaiting_emotion | awaiting_vibe | done
        "phase":             "start",
        "intent":            None,
        "last_query":        "",   # 最近一次用户原始输入
        "accumulated_query": "",   # 历轮用户输入的拼接，供推荐使用
        "polarity_key":      None, # 用户选定的极性
        "pending_polarity":  None, # 卡片点击暂存（触发 rerun 后消费）
        "pending_relation":  None, # 关系按钮暂存
        "pending_clarify":   None, # 澄清按钮暂存
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── 欢迎语（仅首次）──────────────────────────────────────────────────
    if not st.session_state.history:
        with st.chat_message("assistant"):
            st.markdown(_WELCOME)

    # ── 渲染历史（静态）──────────────────────────────────────────────────
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "recommendations":
                st.markdown(msg.get("content", ""))
                _render_recommendations(msg["results"], msg["signals"])
            else:
                st.markdown(msg["content"])

    # ═══════════════════════════════════════════════════════════════════════
    # 消费「按钮点击」事件（各阶段的 pending_* 暂存）
    # ═══════════════════════════════════════════════════════════════════════

    # ── 1. 澄清按钮（ambiguous → gift/self）────────────────────────────
    if st.session_state.phase == "awaiting_clarify" \
            and st.session_state.pending_clarify:
        choice = st.session_state.pending_clarify
        st.session_state.pending_clarify = None

        if choice == "gift":
            _append_history("assistant", "好的！是送给谁的呢？")
            st.session_state.phase = "awaiting_gift_who"
        else:
            _append_history("assistant", "好的！你现在的感觉更偏向哪个？")
            st.session_state.phase = "awaiting_emotion"
        st.rerun()   # 刷新 history 显示 + 让新阶段的按钮出现

    # ── 2. 关系按钮（awaiting_gift_who → recommend 或切到自用流程）──────
    if st.session_state.phase == "awaiting_gift_who" \
            and st.session_state.pending_relation:
        uid = st.session_state.pending_relation
        st.session_state.pending_relation = None

        # 从 RELATION_BUTTONS 查标签和 scene_key
        btn_entry = next((b for b in RELATION_BUTTONS if b[0] == uid), None)
        rel_label  = btn_entry[1] if btn_entry else uid
        scene_key  = btn_entry[2] if btn_entry else uid

        if scene_key == "__self__":
            # 用户反悔，切换到自用流程
            _append_history("user", "（想送给自己）")
            _append_history("assistant", "好的！那先说说你现在的感觉更偏向哪个？")
            st.session_state.phase = "awaiting_emotion"
            st.rerun()
        else:
            _append_history("user", f"（选了「{rel_label}」）")
            with st.spinner("为你找诗…"):
                results, signals = _do_recommend(
                    df, dim_kw, emo, scenes,
                    st.session_state.accumulated_query,
                    st.session_state.intent,
                    scene_key_override=scene_key,
                )
            reply = f"这是为「{rel_label}」找到的花诗："
            _append_history("assistant", reply, results=results, signals=signals)
            st.session_state.phase = "done"
            st.rerun()

    # ── 3. 情感卡片（awaiting_emotion → awaiting_vibe）─────────────────
    if st.session_state.phase == "awaiting_emotion" \
            and st.session_state.pending_polarity:
        polarity = st.session_state.pending_polarity
        st.session_state.pending_polarity = None
        st.session_state.polarity_key = polarity

        pol_label = next((l for k, l, _ in POLARITY_OPTIONS
                          if k == polarity), polarity)
        _append_history("user", f"（选了「{pol_label}」）")
        _append_history("assistant",
                        "你希望进入什么氛围？比如轻松，舒适，花园，江南，海边…")
        st.session_state.phase = "awaiting_vibe"
        st.rerun()

    # ── 4. 当前阶段的持久交互元素（问题文字已在 history 里，只补充按钮/卡片）
    elif st.session_state.phase == "awaiting_emotion":
        # history 已有问题文字；此处只渲染卡片，不重复文字
        _render_polarity_cards()

    elif st.session_state.phase == "awaiting_clarify":
        _render_clarify_buttons()

    elif st.session_state.phase == "awaiting_gift_who":
        _render_relation_buttons()

    # ═══════════════════════════════════════════════════════════════════════
    # 聊天输入框（所有阶段均可输入，不同阶段不同处理）
    # ═══════════════════════════════════════════════════════════════════════
    hint_map = {
        "start":             "描述心情、场合，或想在诗里看到什么…",
        "awaiting_clarify":  "或者直接描述更多细节…",
        "awaiting_gift_who": "也可以直接描述，比如「老师退休了」…",
        "awaiting_emotion":  "或者直接用文字描述感受…",
        "awaiting_vibe":     "输入氛围/意象（或直接回车跳过）…",
        "done":              "继续描述，或告诉我想怎么调整…",
    }
    hint = hint_map.get(st.session_state.phase, "描述心情或场合…")

    if user_input := st.chat_input(hint):
        st.session_state.pending_polarity = None
        st.session_state.pending_relation = None
        st.session_state.pending_clarify  = None
        st.session_state.last_query = user_input

        _append_history("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        # 累积查询：把每轮用户输入拼起来，让推荐上下文更完整
        st.session_state.accumulated_query = (
            (st.session_state.accumulated_query + " " + user_input).strip()
        )

        phase = st.session_state.phase

        # ── 送人-关系阶段：用户自由补充描述后直接推荐 ──────────────────
        if phase == "awaiting_gift_who":
            with st.spinner("理解中…"):
                intent = _parse_intent(st.session_state.accumulated_query)
            st.session_state.intent = intent
            with st.chat_message("assistant"):
                with st.spinner("为你找诗…"):
                    results, signals = _do_recommend(
                        df, dim_kw, emo, scenes,
                        st.session_state.accumulated_query, intent,
                    )
                reply = "根据你的描述，找到以下花诗："
                st.markdown(reply)
                _render_recommendations(results, signals)
            _append_history("assistant", reply, results=results, signals=signals)
            st.session_state.phase = "done"

        # ── 氛围输入：接收后直接推荐 ────────────────────────────────────
        elif phase == "awaiting_vibe":
            with st.chat_message("assistant"):
                with st.spinner("为你找诗…"):
                    results, signals = _do_recommend(
                        df, dim_kw, emo, scenes,
                        st.session_state.accumulated_query,
                        st.session_state.intent,
                        polarity_key=st.session_state.polarity_key,
                    )
                reply = "根据你的心情和氛围，找到以下花诗："
                st.markdown(reply)
                _render_recommendations(results, signals)
            _append_history("assistant", reply, results=results, signals=signals)
            st.session_state.phase = "done"

        # ── done 阶段继续输入：重新解析意图，刷新推荐 ───────────────────
        elif phase == "done":
            with st.spinner("理解中…"):
                intent = _parse_intent(st.session_state.accumulated_query)
            st.session_state.intent = intent
            with st.chat_message("assistant"):
                with st.spinner("为你更新推荐…"):
                    results, signals = _do_recommend(
                        df, dim_kw, emo, scenes,
                        st.session_state.accumulated_query,
                        intent,
                        polarity_key=st.session_state.polarity_key,
                    )
                reply = "根据你补充的描述，更新推荐如下："
                st.markdown(reply)
                _render_recommendations(results, signals)
            _append_history("assistant", reply, results=results, signals=signals)

        # ── 其他阶段（start / awaiting_clarify / gift_who / emotion）──────
        else:
            with st.spinner("理解中…"):
                intent = _parse_intent(st.session_state.accumulated_query)
            st.session_state.intent = intent

            category = _classify(intent, st.session_state.accumulated_query)

            if category == "ambiguous":
                ask = "想送给别人，还是自己欣赏？"
                _append_history("assistant", ask)
                st.session_state.phase = "awaiting_clarify"
                with st.chat_message("assistant"):
                    st.markdown(ask)
                    _render_clarify_buttons()

            elif category == "gift":
                # 只有用户文本里明确出现关系词才跳过追问
                if _gift_has_relation(st.session_state.accumulated_query):
                    with st.chat_message("assistant"):
                        with st.spinner("为你找诗…"):
                            results, signals = _do_recommend(
                                df, dim_kw, emo, scenes,
                                st.session_state.accumulated_query, intent,
                            )
                        reply = "为你找到以下花诗："
                        st.markdown(reply)
                        _render_recommendations(results, signals)
                    _append_history("assistant", reply, results=results, signals=signals)
                    st.session_state.phase = "done"
                else:
                    # 关系不明确 → 必须追问送给谁
                    ask = "是送给谁的呢？"
                    _append_history("assistant", ask)
                    st.session_state.phase = "awaiting_gift_who"
                    with st.chat_message("assistant"):
                        st.markdown(ask)
                        _render_relation_buttons()

            else:  # self
                # 只有用户原文中有明确情绪词才跳过情感卡片
                if _has_emotion_signal(st.session_state.accumulated_query):
                    # 有情感信号 → 直接问氛围
                    ask = "你希望进入什么氛围？比如轻松，舒适，花园，江南，海边…"
                    _append_history("assistant", ask)
                    st.session_state.phase = "awaiting_vibe"
                    with st.chat_message("assistant"):
                        st.markdown(ask)
                else:
                    # 无情感信号 → 必须展示情感卡片
                    ask = "你现在的感觉更偏向哪个？"
                    _append_history("assistant", ask)
                    st.session_state.phase = "awaiting_emotion"
                    with st.chat_message("assistant"):
                        st.markdown(ask)
                        _render_polarity_cards()


# ── 工具 ─────────────────────────────────────────────────────────────────────

def _append_history(role: str, content: str,
                    results: list | None = None,
                    signals: dict | None = None) -> None:
    msg: dict[str, Any] = {"role": role, "content": content}
    if results is not None:
        msg["type"]    = "recommendations"
        msg["results"] = results
        msg["signals"] = signals or {}
    else:
        msg["type"] = "text"
    st.session_state.history.append(msg)


if __name__ == "__main__":
    main()
