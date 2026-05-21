"""Engine contract used by both runners: setup, generate, teardown."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


@dataclass
class Sample:
    """One generated completion. logprobs[i] is the chosen token's logprob at step i."""

    text: str
    token_ids: list[int]
    logprobs: list[float]
    n_prompt_tokens: int
    n_output_tokens: int
    finish_reason: str
    stop_reason: str


class Engine(ABC):
    """A model with a particular batch-invariance setting."""

    label: str = ""
    name: str

    @property
    def key(self) -> str:
        """Stable class-name id used in filenames and the engine CSV column."""
        return type(self).__name__

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        n: int,
        sampling: dict | None = None,
    ) -> list[list[Sample]]:
        """Return n samples per prompt, indexed [prompt_idx][sample_idx]."""

    @abstractmethod
    def teardown(self) -> None: ...


# Class attrs point at these; __init__ always copies them so the shared
# dicts are never mutated in place.
DEFAULT_SAMPLING: dict = {
    "temperature": 0,
    "max_tokens": 2048,
    "logprobs": 1,
}
DEFAULT_VLLM_KWARGS: dict = {
    "enable_prefix_caching": False,
}
DEFAULT_CHAT_TEMPLATE_KWARGS: dict = {
    "enable_thinking": False,
}


class VLLMBase(Engine):
    """Common vLLM setup. Subclass to swap models or batch-invariance modes."""

    label = "Default"

    hf_id: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    vllm_kwargs: dict = DEFAULT_VLLM_KWARGS
    chat_template_kwargs: dict = DEFAULT_CHAT_TEMPLATE_KWARGS
    default_sampling: dict = DEFAULT_SAMPLING

    def __init__(
        self,
        name: str | None = None,
        vllm_kwargs: dict | None = None,
        sampling: dict | None = None,
    ) -> None:
        self._llm: LLM | None = None
        self._tokenizer = None
        self.vllm_kwargs = {**type(self).vllm_kwargs, **(vllm_kwargs or {})}
        # model / dtype / max_model_len may arrive via vllm_kwargs; pull them
        # onto the instance so the in-process LLM and the engine spec agree.
        self.hf_id = self.vllm_kwargs.pop("model", type(self).hf_id)
        self.dtype = self.vllm_kwargs.pop("dtype", type(self).dtype)
        self.max_model_len = self.vllm_kwargs.pop(
            "max_model_len", type(self).max_model_len
        )
        self.chat_template_kwargs = dict(type(self).chat_template_kwargs)
        self.name = name or self.key
        self.default_sampling = {**type(self).default_sampling, **(sampling or {})}

    def setup(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        llm_kwargs = {
            "model": self.hf_id,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            **self.vllm_kwargs,
            **self._extra_vllm_kwargs(),
        }
        self._llm = LLM(**llm_kwargs)

    @property
    def llm(self) -> LLM:
        """The underlying vllm.LLM. Raises if setup() hasn't been called."""
        if self._llm is None:
            raise RuntimeError(
                f"engine {self.name!r} has no LLM; call setup() before .llm"
            )
        return self._llm

    @property
    def tokenizer(self):
        """The underlying tokenizer. Raises if setup() hasn't been called."""
        if self._tokenizer is None:
            raise RuntimeError(
                f"engine {self.name!r} has no tokenizer; call setup() before .tokenizer"
            )
        return self._tokenizer

    def _extra_vllm_kwargs(self) -> dict:
        """Extra LLM() kwargs computed at setup. Base adds none."""
        return {}

    def _apply_chat_template(self, prompt: str) -> str:
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

    def generate(
        self,
        prompts: list[str],
        n: int,
        sampling: dict | None = None,
    ) -> list[list[Sample]]:
        assert self._llm is not None, "call setup() first"
        params = SamplingParams(n=n, **{**self.default_sampling, **(sampling or {})})
        chat_prompts = [self._apply_chat_template(p) for p in prompts]
        outputs = self._llm.generate(chat_prompts, params, use_tqdm=False)

        result: list[list[Sample]] = []
        for req in outputs:
            n_prompt_tokens = (
                len(req.prompt_token_ids) if req.prompt_token_ids is not None else 0
            )
            samples: list[Sample] = []
            for comp in req.outputs:
                token_ids = list(comp.token_ids)
                # vLLM returns logprobs as a list of {token_id: Logprob} dicts.
                # Pull the chosen token's logprob per step, NaN if missing.
                if comp.logprobs is not None:
                    logprobs = [
                        float(step_lp[tok].logprob)
                        if step_lp is not None and tok in step_lp
                        else float("nan")
                        for tok, step_lp in zip(token_ids, comp.logprobs)
                    ]
                else:
                    logprobs = [float("nan")] * len(token_ids)
                samples.append(
                    Sample(
                        text=comp.text,
                        token_ids=token_ids,
                        logprobs=logprobs,
                        n_prompt_tokens=n_prompt_tokens,
                        n_output_tokens=len(token_ids),
                        finish_reason=str(comp.finish_reason or ""),
                        stop_reason=(
                            str(comp.stop_reason)
                            if comp.stop_reason is not None
                            else ""
                        ),
                    )
                )
            result.append(samples)
        return result

    def teardown(self) -> None:
        del self._llm
        self._llm = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
