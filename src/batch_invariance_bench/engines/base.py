"""Engine contract and the vLLM engines used by both runners.

A mode needing env vars before build (TM, FXPR) overrides _env(); FXPR registers
its kernels in _on_setup(). VLLMBase does the env save/restore.
"""

from __future__ import annotations

import gc
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _apply_env(updates: Mapping[str, str]) -> dict[str, str | None]:
    """Set env vars, return old values for _restore_env()."""
    prev: dict[str, str | None] = {}
    for key, value in updates.items():
        prev[key] = os.environ.get(key)
        os.environ[key] = value
    return prev


def _restore_env(prev: Mapping[str, str | None]) -> None:
    """Undo _apply_env()."""
    for key, old in prev.items():
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


@dataclass
class Sample:
    """One completion; logprobs[i] is the chosen token's logprob at step i."""

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
        """Class-name id used in filenames and the engine CSV column."""
        return type(self).__name__

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        n: int,
        sampling: dict | None = None,
        use_tqdm: bool = False,
    ) -> list[list[Sample]]:
        """n samples per prompt, indexed [prompt_idx][sample_idx]."""

    @abstractmethod
    def teardown(self) -> None: ...


class VLLMBase(Engine):
    """Common vLLM setup. Subclass to swap models or modes."""

    label = "Default"

    hf_id: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    # __init__ copies these so the shared dicts are never mutated.
    vllm_kwargs: dict = {"enable_prefix_caching": False}
    chat_template_kwargs: dict = {"enable_thinking": False}
    default_sampling: dict = {"temperature": 0, "max_tokens": 2048, "logprobs": 1}

    def __init__(
        self,
        name: str | None = None,
        vllm_kwargs: dict | None = None,
        sampling: dict | None = None,
    ) -> None:
        self._llm: LLM | None = None
        self._tokenizer = None
        self._prev_env: dict[str, str | None] = {}
        self.vllm_kwargs = {**type(self).vllm_kwargs, **(vllm_kwargs or {})}
        # model/dtype/max_model_len may arrive via vllm_kwargs; pull onto self.
        self.hf_id = self.vllm_kwargs.pop("model", type(self).hf_id)
        self.dtype = self.vllm_kwargs.pop("dtype", type(self).dtype)
        self.max_model_len = self.vllm_kwargs.pop(
            "max_model_len", type(self).max_model_len
        )
        self.chat_template_kwargs = dict(type(self).chat_template_kwargs)
        self.name = name or self.key
        self.default_sampling = {**type(self).default_sampling, **(sampling or {})}

    def _env(self) -> dict[str, str]:
        """Env vars to set before build. Base sets none."""
        return {}

    def _on_setup(self) -> None:
        """Hook after env is applied, before LLM() is built. Base does nothing."""

    def _extra_vllm_kwargs(self) -> dict:
        """Extra LLM() kwargs at setup. Base adds none."""
        return {}

    def setup(self) -> None:
        self._prev_env = _apply_env(self._env())
        self._on_setup()
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
        """The vllm.LLM; raises before setup()."""
        if self._llm is None:
            raise RuntimeError(
                f"engine {self.name!r} has no LLM; call setup() before .llm"
            )
        return self._llm

    @property
    def tokenizer(self):
        """The tokenizer; raises before setup()."""
        if self._tokenizer is None:
            raise RuntimeError(
                f"engine {self.name!r} has no tokenizer; call setup() before .tokenizer"
            )
        return self._tokenizer

    def _apply_chat_template(self, prompt: str) -> str:
        if not getattr(self._tokenizer, "chat_template", None):
            return prompt
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
        use_tqdm: bool = False,
    ) -> list[list[Sample]]:
        assert self._llm is not None, "call setup() first"
        params = SamplingParams(n=n, **{**self.default_sampling, **(sampling or {})})
        chat_prompts = [self._apply_chat_template(p) for p in prompts]
        outputs = self._llm.generate(chat_prompts, params, use_tqdm=use_tqdm)

        result: list[list[Sample]] = []
        for req in outputs:
            n_prompt_tokens = (
                len(req.prompt_token_ids) if req.prompt_token_ids is not None else 0
            )
            samples: list[Sample] = []
            for comp in req.outputs:
                token_ids = list(comp.token_ids)
                # logprobs come as per-step {token_id: Logprob}; pull the chosen
                # token's logprob, NaN if missing.
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
        try:
            del self._llm
            self._llm = None
            self._tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            _restore_env(self._prev_env)
            self._prev_env = {}


class VLLMTMBatchInvariant(VLLMBase):
    """vLLM with Thinking Machines batch-invariant ops.

    https://github.com/thinking-machines-lab/batch_invariant_ops
    """

    label = "TM"

    def _env(self) -> dict[str, str]:
        # vLLM reads this at build time.
        return {"VLLM_BATCH_INVARIANT": "1"}


# Tells "argument omitted" apart from an explicit None (None means disable).
_UNSET: Any = object()


class VLLMFxpr(VLLMBase):
    """vLLM with fxpr_vllm fixed-point reduction kernels."""

    label = "FXPR"

    quantization: str | None = "fixedpoint"
    attention_backend: str | None = "CUSTOM"
    fxp_int_bits: int = 32
    fxp_frac_bits: int = 16

    def __init__(
        self,
        name: str | None = None,
        *,
        quantization: str | None = _UNSET,
        attention_backend: str | None = _UNSET,
        fxp_int_bits: int | None = None,
        fxp_frac_bits: int | None = None,
        vllm_kwargs: dict | None = None,
        sampling: dict | None = None,
    ) -> None:
        super().__init__(name=name, vllm_kwargs=vllm_kwargs, sampling=sampling)
        if quantization is not _UNSET:
            self.quantization = quantization
        if attention_backend is not _UNSET:
            self.attention_backend = attention_backend
        if fxp_int_bits is not None:
            self.fxp_int_bits = fxp_int_bits
        if fxp_frac_bits is not None:
            self.fxp_frac_bits = fxp_frac_bits

    def _env(self) -> dict[str, str]:
        # Read by fxpr at kernel registration.
        return {
            "FXPR_INT_BITS": str(self.fxp_int_bits),
            "FXPR_FRAC_BITS": str(self.fxp_frac_bits),
        }

    def _on_setup(self) -> None:
        # Env (set by setup) must land before register() reads it.
        from fxpr_vllm.register import register

        register()

    def _extra_vllm_kwargs(self) -> dict:
        extra: dict = {}
        if self.quantization is not None:
            extra["quantization"] = self.quantization
        if self.attention_backend is not None:
            extra["attention_backend"] = self.attention_backend
        return extra
