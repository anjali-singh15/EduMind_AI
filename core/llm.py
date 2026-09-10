import time
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI


DEFAULT_MODELS = ("gemini-3.6-flash",)


def make_llm(api_key: str, temperature: float = 0.3, model: Optional[str] = None) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model or DEFAULT_MODELS[0],
        google_api_key=api_key,
        temperature=temperature,
    )


def invoke_with_retry(api_key: str, messages, temperature: float = 0.3, attempts: int = 3) -> str:
    last_error = None
    for model in DEFAULT_MODELS:
        llm = make_llm(api_key, temperature, model)
        for attempt in range(attempts):
            try:
                response = llm.invoke(messages)
                content = response.content
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict):
                        return first.get("text", str(first))
                    return str(first)
                return str(content)
            except Exception as exc:
                last_error = exc
                text = str(exc).lower()
                if "503" in text and attempt < attempts - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                break
    raise last_error
