from openai import OpenAI
import json
from retry import retry
import os
from ProcedureMem.runtime_config import load_environment


load_environment()

_UNSET = object()

TEMPERATURE = 0.2
TOP_P = 1
MAX_TOKENS = 4096


def _required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_client(api_key=None, api_base_url=_UNSET):
    resolved_api_key = api_key or _required_env("OPENAI_API_KEY")
    kwargs = {"api_key": resolved_api_key}
    if api_base_url is _UNSET:
        api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    else:
        api_base = api_base_url
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)

def get_response(messages, model=None, api_key=None, api_base_url=_UNSET):
    client = _get_client(api_key=api_key, api_base_url=api_base_url)
    response = client.chat.completions.create(
        model=model or _required_env("MODEL_NAME"),
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS
    )
    if not hasattr(response, "error"):
        return response.choices[0].message.content
    return response.error.message

@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def get_llm_response(
    messages,
    is_string=False,
    model=None,
    api_key=None,
    api_base_url=_UNSET,
):
    ans = get_response(
        messages,
        model=model,
        api_key=api_key,
        api_base_url=api_base_url,
    )
    if is_string:
        return ans
    else:
        cleaned_text = ans.strip("`json\n").strip("`\n").strip("```\n")
        ans = json.loads(cleaned_text)
        return ans

from langchain_openai import OpenAIEmbeddings

def get_embedding_model():
    api_key = os.getenv("EMBEDDING_MODEL_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing EMBEDDING_MODEL_KEY (or OPENAI_API_KEY fallback)"
        )
    api_base = os.getenv("EMBEDDING_MODEL_BASE_URL") or os.getenv("OPENAI_API_BASE")
    embedding = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
        openai_api_key=api_key,
        openai_api_base=api_base,
        max_retries=10
    )
    return embedding

if __name__ == "__main__":
    message = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    print(get_response(message))
    # test embedding
    embedding = get_embedding_model()
    print(embedding.embed_query("Hello, how are you?"))
