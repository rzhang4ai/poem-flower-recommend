"""
豆包（Doubao / Ark）API 客户端封装。
Doubao 平台兼容 OpenAI 协议，直接使用 openai 包驱动。

环境变量（可写入 05.recommend/.env 或在终端 export）：
  ARK_API_KEY       必填，Ark 平台 API Key
  ARK_CHAT_MODEL    聊天模型 ID，默认 doubao-pro-32k
  ARK_EMBED_MODEL   嵌入模型 ID，默认 doubao-embedding-large-text
  ARK_BASE_URL      可选，默认 https://ark.volces.com/api/v3/
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """读取同目录 .env 文件（若存在）注入环境变量；不依赖 python-dotenv。"""
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
    """按优先级读取密钥：st.secrets → 环境变量。"""
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, "")


@lru_cache(maxsize=1)
def _client():
    """惰性初始化 OpenAI 客户端（指向 Doubao/Ark 端点）。"""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("请安装 openai 包：pip install openai") from e

    api_key = _get_secret("ARK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "未找到 ARK_API_KEY。请在 st.secrets 或 .env 中配置 ARK_API_KEY"
        )
    base_url = _get_secret("ARK_BASE_URL") or "https://ark.volces.com/api/v3/"
    return OpenAI(api_key=api_key, base_url=base_url)


def is_available() -> bool:
    """检查 API Key 是否已配置（不发起网络请求）。"""
    _load_dotenv()
    return bool(os.environ.get("ARK_API_KEY", ""))


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 600,
    timeout: float = 20.0,
) -> str:
    """
    调用 Doubao 聊天补全接口。

    Parameters
    ----------
    messages  : OpenAI 格式的消息列表
    model     : 模型 ID（默认读 ARK_CHAT_MODEL 环境变量）
    timeout   : 请求超时秒数

    Returns
    -------
    模型返回的文本内容
    """
    model = model or os.environ.get("ARK_CHAT_MODEL", "doubao-pro-32k")
    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("Doubao chat API 调用失败: %s", e)
        raise


def embed(
    texts: list[str],
    model: str | None = None,
    timeout: float = 30.0,
) -> list[list[float]]:
    """
    调用 Doubao 嵌入接口，返回每条文本的浮点向量列表。

    Parameters
    ----------
    texts   : 待嵌入的文本列表（单次建议 ≤ 32 条）
    model   : 嵌入模型 ID（默认读 ARK_EMBED_MODEL 环境变量）
    """
    model = model or os.environ.get("ARK_EMBED_MODEL", "doubao-embedding-large-text")
    try:
        resp = _client().embeddings.create(
            model=model,
            input=texts,
            timeout=timeout,
        )
        # 按 index 排序保证顺序
        items = sorted(resp.data, key=lambda x: x.index)
        return [item.embedding for item in items]
    except Exception as e:
        logger.warning("Doubao embed API 调用失败: %s", e)
        raise
