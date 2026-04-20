# -*- coding: utf-8 -*-
# =====================
# 
# 
# Author: liumin.423
# Date:   2025/7/8
# =====================
import json
import os
from typing import List, Any, Optional, Tuple

from dotenv import load_dotenv
from litellm import acompletion

from genie_tool.util.log_util import timer, AsyncTimer
from genie_tool.util.sensitive_detection import SensitiveWordsReplace

load_dotenv()


def _resolve_litellm_model_and_auth(model: str, kwargs: dict) -> Tuple[str, dict]:
    """
    LiteLLM 需要显式 provider。使用 OpenAI 兼容端点（如阿里云 DashScope
    compatible-mode）时，裸模型名如 qwen3.6-plus 会报
    LLM Provider NOT provided，需写成 openai/qwen3.6-plus 并带上 api_base/api_key。
    已含 provider/ 的模型名（如 deepseek/deepseek-chat、openai/gpt-4o）保持不变。
    """
    api_base = kwargs.get("api_base") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_base or not api_key:
        api_base = api_base or os.getenv("DEEPSEEK_API_BASE")
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")

    extra = {}
    if api_base:
        extra["api_base"] = api_base
    if api_key:
        extra["api_key"] = api_key

    if extra.get("api_base") and extra.get("api_key") and "/" not in model:
        model = f"openai/{model}"

    return model, extra


@timer(key="enter")
async def ask_llm(
        messages: str | List[Any],
        model: str,
        temperature: float = None,
        top_p: float = None,
        stream: bool = False,

        # 自定义字段
        only_content: bool = False,     # 只返回内容

        extra_headers: Optional[dict] = None,
        **kwargs,
):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if os.getenv("SENSITIVE_WORD_REPLACE", "false") == "true":
        for message in messages:
            if isinstance(message.get("content"), str):
                message["content"] = SensitiveWordsReplace.replace(message["content"])
            else:
                message["content"] = json.loads(
                    SensitiveWordsReplace.replace(json.dumps(message["content"], ensure_ascii=False)))
    model, auth = _resolve_litellm_model_and_auth(model, kwargs)
    response = await acompletion(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
        extra_headers=extra_headers,
        **auth,
        **kwargs
    )
    async with AsyncTimer(key=f"exec ask_llm"):
        if stream:
            async for chunk in response:
                if only_content:
                    if chunk.choices and chunk.choices[0] and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                else:
                    yield chunk
        else:
            yield response.choices[0].message.content if only_content else response


if __name__ == "__main__":
    pass
