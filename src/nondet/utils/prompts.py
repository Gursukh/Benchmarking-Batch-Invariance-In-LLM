"""Load prompt sets; a failed download raises."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Prompt:
    id: str
    text: str
    source: str
    reference: str | None = None


# name -> (HF repo, id field); all use the test split with problem/answer fields
_DATASETS = {
    "math500": ("HuggingFaceH4/MATH-500", "unique_id"),
    "aime2025": ("math-ai/aime25", "id"),
}


def _load(name: str) -> list[Prompt]:
    from datasets import load_dataset

    try:
        repo, id_field = _DATASETS[name]
    except KeyError:
        raise ValueError(
            f"unknown prompt set {name!r}; available: {', '.join(_DATASETS)}"
        )
    ds = load_dataset(repo, split="test")
    return [
        Prompt(str(row.get(id_field, i)), row["problem"], name, str(row.get("answer", "")))
        for i, row in enumerate(ds)
    ]


def prompt_set(name: str = "math500", n: int | None = 100, seed: int = 0) -> list[Prompt]:
    """Up to n prompts (all if n is None), shuffled by seed."""
    items = _load(name)
    random.Random(seed).shuffle(items)
    return items if n is None else items[:n]


# only Qwen templates read enable_thinking; others ignore it
DEFAULT_CHAT_TEMPLATE_KWARGS: dict = {"enable_thinking": False}


def apply_chat(tokenizer, text: str, chat_template_kwargs: dict | None = None) -> str:
    """Wrap text in the model's chat template (no-op if none); thinking off by default."""
    if not getattr(tokenizer, "chat_template", None):
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_kwargs or DEFAULT_CHAT_TEMPLATE_KWARGS),
    )


# leading newline+spaces trip the SentencePiece/Metaspace decoder; a good tokenizer round-trips it
_RT_PROBE = "\n a b"


def _roundtrips(tok) -> bool:
    try:
        ids = tok.encode(_RT_PROBE, add_special_tokens=False)
        return tok.decode(ids) == _RT_PROBE
    except Exception:
        return False


def load_tokenizer(model_id: str):
    """Load a tokenizer that decodes correctly. Some repos declare a SentencePiece/
    Metaspace class that drops spaces/newlines; if the default fails the round-trip
    probe, fall back to the generic fast backend that honors tokenizer.json."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tok = AutoTokenizer.from_pretrained(model_id)
    if _roundtrips(tok):
        return tok
    try:
        fixed = PreTrainedTokenizerFast.from_pretrained(model_id)
    except Exception:
        return tok
    return fixed if _roundtrips(fixed) else tok


def vllm_tokenizer(model_id: str) -> str:
    """Tokenizer path for vLLM's LLM(tokenizer=...). Returns model_id if it round-trips,
    else writes a corrected tokenizer under the HF cache and returns that dir."""
    import os

    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if _roundtrips(AutoTokenizer.from_pretrained(model_id)):
        return model_id
    out = os.path.join(
        os.path.expanduser("~/.cache/huggingface/fixed_tokenizers"),
        model_id.replace("/", "__"),
    )
    if not os.path.isfile(os.path.join(out, "tokenizer.json")):
        os.makedirs(out, exist_ok=True)
        PreTrainedTokenizerFast.from_pretrained(model_id).save_pretrained(out)
    return out
