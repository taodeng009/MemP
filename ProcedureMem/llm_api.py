from openai import OpenAI
import json
import math
from retry import retry
import os
from ProcedureMem.runtime_config import load_environment


load_environment()

_UNSET = object()

DEFAULT_MEMORY_BUILD_TEMPERATURE = 0.0
DEFAULT_MEMORY_BUILD_SEED = 42
DEFAULT_MEMORY_BUILD_TOP_K = 1
TOP_P = 1
MAX_TOKENS = 4096


def _required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _optional_bool_env(name):
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, found {value!r}")


def resolve_memory_build_temperature(value=None):
    raw_value = value
    if raw_value is None:
        raw_value = os.getenv("MEMORY_BUILD_TEMPERATURE")
    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        return DEFAULT_MEMORY_BUILD_TEMPERATURE
    try:
        temperature = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "MEMORY_BUILD_TEMPERATURE must be a non-negative finite number, "
            f"found {raw_value!r}"
        ) from exc
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError(
            "MEMORY_BUILD_TEMPERATURE must be a non-negative finite number, "
            f"found {raw_value!r}"
        )
    return temperature


def resolve_memory_build_seed(value=None):
    raw_value = value
    if raw_value is None:
        raw_value = os.getenv("MEMORY_BUILD_SEED")
    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        return DEFAULT_MEMORY_BUILD_SEED
    if isinstance(raw_value, bool):
        raise ValueError(
            f"MEMORY_BUILD_SEED must be an integer, found {raw_value!r}"
        )
    try:
        seed = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"MEMORY_BUILD_SEED must be an integer, found {raw_value!r}"
        ) from exc
    if isinstance(raw_value, float) and raw_value != seed:
        raise ValueError(
            f"MEMORY_BUILD_SEED must be an integer, found {raw_value!r}"
        )
    if isinstance(raw_value, str) and raw_value.strip() != str(seed):
        raise ValueError(
            f"MEMORY_BUILD_SEED must be an integer, found {raw_value!r}"
        )
    return seed


def resolve_memory_build_top_k(value=None):
    raw_value = value
    if raw_value is None:
        raw_value = os.getenv("MEMORY_BUILD_TOP_K")
    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        return DEFAULT_MEMORY_BUILD_TOP_K
    if isinstance(raw_value, bool):
        raise ValueError(
            f"MEMORY_BUILD_TOP_K must be a positive integer, found {raw_value!r}"
        )
    try:
        top_k = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"MEMORY_BUILD_TOP_K must be a positive integer, found {raw_value!r}"
        ) from exc
    if (
        top_k < 1
        or (isinstance(raw_value, float) and raw_value != top_k)
        or (isinstance(raw_value, str) and raw_value.strip() != str(top_k))
    ):
        raise ValueError(
            f"MEMORY_BUILD_TOP_K must be a positive integer, found {raw_value!r}"
        )
    return top_k


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

def get_response(
    messages,
    model=None,
    api_key=None,
    api_base_url=_UNSET,
    temperature=None,
    seed=None,
    top_k=None,
):
    client = _get_client(api_key=api_key, api_base_url=api_base_url)
    request = {
        "model": model or _required_env("MODEL_NAME"),
        "messages": messages,
        "temperature": resolve_memory_build_temperature(temperature),
        "seed": resolve_memory_build_seed(seed),
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }
    extra_body = {"top_k": resolve_memory_build_top_k(top_k)}
    enable_thinking = _optional_bool_env("MEMORY_BUILD_ENABLE_THINKING")
    if enable_thinking is not None:
        extra_body["enable_thinking"] = enable_thinking
    request["extra_body"] = extra_body
    response = client.chat.completions.create(**request)
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
    temperature=None,
    seed=None,
    top_k=None,
):
    ans = get_response(
        messages,
        model=model,
        api_key=api_key,
        api_base_url=api_base_url,
        temperature=temperature,
        seed=seed,
        top_k=top_k,
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
        max_retries=10,
        # Non-OpenAI endpoints must tokenize with their own model vocabulary.
        check_embedding_ctx_length=False,
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
