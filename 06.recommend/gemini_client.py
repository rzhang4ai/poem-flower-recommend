"""
Google Gemini API 客户端封装（google-genai SDK）。

环境变量（可写入 05.recommend/.env 或在终端 export）：
  GOOGLE_API_KEY      必填，Google AI Studio API Key
                      获取地址：https://aistudio.google.com/apikey
  GEMINI_CHAT_MODEL   聊天模型 ID，默认 gemini-3-flash-preview
  GEMINI_EMBED_MODEL  嵌入模型 ID，默认 gemini-embedding-2-preview

重要说明：
  - 带「思考」的 Gemini（如 gemini-2.5-flash/pro）会把内部推理计入
    max_output_tokens，导致正文被截断。本客户端统一设置
    thinking_config=ThinkingConfig(thinking_budget=0) 规避。
  - 嵌入使用 task_type 区分文档（RETRIEVAL_DOCUMENT）与查询（RETRIEVAL_QUERY），
    参考：https://ai.google.dev/gemini-api/docs/embeddings
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

_DEFAULT_CHAT_MODEL  = "gemini-3-flash-preview"
_DEFAULT_EMBED_MODEL = "gemini-embedding-2-preview"
_DEFAULT_MAX_TOKENS  = 1024


def _load_dotenv() -> None:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()


def _get_secret(key: str) -> str:
    """按优先级读取密钥：st.secrets → 环境变量（含 .env）。"""
    # Streamlit Cloud secrets（share.streamlit.io 后台配置）
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, "")


def _api_key() -> str:
    return _get_secret("GOOGLE_API_KEY") or _get_secret("GEMINI_API_KEY")


def is_available() -> bool:
    """检查 API Key 是否已配置（不发起网络请求）。"""
    _load_dotenv()
    return bool(_api_key())


@lru_cache(maxsize=1)
def _client():
    """惰性初始化 Gemini 客户端。"""
    try:
        from google import genai
    except ImportError as e:
        raise ImportError("请安装 google-genai：pip install google-genai") from e
    key = _api_key()
    if not key:
        raise EnvironmentError(
            "未找到 GOOGLE_API_KEY。请在 05.recommend/.env 设置，或 export GOOGLE_API_KEY=xxx"
        )
    return genai.Client(api_key=key)


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> str:
    """
    调用 Gemini 聊天接口（OpenAI message list → Gemini 格式自动转换）。

    messages 格式：[{"role": "system"|"user"|"assistant", "content": "..."}]

    thinking_budget=0 已强制禁用，避免 thinking token 占用 max_output_tokens。
    """
    from google.genai import types

    model = model or os.environ.get("GEMINI_CHAT_MODEL", _DEFAULT_CHAT_MODEL)

    # 分离 system prompt
    system_parts: list[str] = []
    history: list[dict] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            history.append({"role": "user", "parts": [{"text": content}]})
        elif role in ("assistant", "model"):
            history.append({"role": "model", "parts": [{"text": content}]})

    system_instruction = "\n\n".join(system_parts) if system_parts else None

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    if system_instruction:
        config.system_instruction = system_instruction

    try:
        resp = _client().models.generate_content(
            model=model,
            contents=history if len(history) > 1 else history[0]["parts"][0]["text"],
            config=config,
        )
        return resp.text or ""
    except Exception as e:
        logger.warning("Gemini chat API 调用失败: %s", e)
        raise


def embed(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    model: str | None = None,
    output_dimensionality: int | None = None,
) -> list[list[float]]:
    """
    调用 Gemini 嵌入接口。

    Parameters
    ----------
    texts               : 待嵌入文本列表
    task_type           : 任务类型
                          - "RETRIEVAL_DOCUMENT"  离线嵌入诗词文档
                          - "RETRIEVAL_QUERY"     在线嵌入用户查询
                          - "SEMANTIC_SIMILARITY" 相似度计算
    output_dimensionality : 输出维度（768/1536/3072），默认 None=3072
                            使用非 3072 维时需自行 L2 归一化（3072 维已自动归一化）

    Returns
    -------
    每条文本对应的浮点向量列表
    """
    from google.genai import types

    model = model or os.environ.get("GEMINI_EMBED_MODEL", _DEFAULT_EMBED_MODEL)

    config = types.EmbedContentConfig(task_type=task_type)
    if output_dimensionality:
        config.output_dimensionality = output_dimensionality

    # 单次最多 100 条（API 限制）
    results: list[list[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = _client().models.embed_content(
                model=model,
                contents=batch,
                config=config,
            )
            results.extend([e.values for e in resp.embeddings])
        except Exception as e:
            logger.warning("Gemini embed API 调用失败 (batch %d): %s", i // batch_size, e)
            raise
    return results
