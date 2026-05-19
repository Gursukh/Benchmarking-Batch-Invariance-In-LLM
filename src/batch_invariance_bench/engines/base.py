"""Engine classes.

Engine is the contract the correctness harness uses: setup, generate, teardown.
VLLMBase is the shared vLLM implementation that the batch-invariance modes
subclass. VLLMBase.server_spec() turns an engine into a ServerSpec, which the
perf harness uses to launch a matching `vllm serve` process.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class ServerSpec:
    """Everything needed to launch a `vllm serve` process for one engine.

    Built only by VLLMBase.server_spec() and read by perf.server.VLLMServer.
    `env` already has VLLM_PLUGINS set from `plugins`.
    """

    key: str
    model_id: str
    dtype: str
    max_model_len: int
    env: dict[str, str] = field(default_factory=dict)
    cli: tuple[str, ...] = field(default_factory=tuple)
    plugins: tuple[str, ...] = field(default_factory=tuple)


class Engine(ABC):
    """A model with a particular batch-invariance setting."""

    # Short label like "Default" or "TM"; subclasses set it.
    label: str = ""

    name: str

    @property
    def key(self) -> str:
        """Stable engine id used in filenames and the `engine` CSV column.

        One class maps to one key.
        """
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


# Default config. Class attrs point at these; __init__ always copies them,
# so the shared dicts are never changed in place.
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
        # Per-instance overrides beat the class defaults. Always a fresh dict.
        self.vllm_kwargs = {**type(self).vllm_kwargs, **(vllm_kwargs or {})}
        # model / dtype / max_model_len can come in via vllm_kwargs; pull them
        # onto the instance so the in-process LLM and the server spec agree.
        # vLLM's "model" kwarg is our hf_id.
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
        # dtype / max_model_len were already resolved in __init__.
        # _extra_vllm_kwargs() lets a subclass add kwargs without storing them.
        llm_kwargs = {
            "model": self.hf_id,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            **self.vllm_kwargs,
            **self._extra_vllm_kwargs(),
        }
        self._llm = LLM(**llm_kwargs)

    def _extra_vllm_kwargs(self) -> dict:
        """Extra vllm.LLM() kwargs, computed at setup. Base engine adds none."""
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
                # vLLM gives logprobs as a list of {token_id: Logprob} dicts.
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

    def _server_env(self) -> dict[str, str]:
        """Env vars (besides VLLM_PLUGINS) the server needs. Base engine: none."""
        return {}

    def _server_cli(self) -> list[str]:
        """Extra `vllm serve` CLI flags for this engine."""
        return []

    def _server_plugins(self) -> list[str]:
        """vLLM plugin names this engine needs in the server.

        Becomes the VLLM_PLUGINS allowlist. Empty means no plugins.
        """
        return []

    def server_spec(self) -> ServerSpec:
        """Build the `vllm serve` launch spec for this engine.

        VLLM_PLUGINS is always set (empty when there are no plugins) so plugin
        loading is explicit rather than ambient.
        """
        plugins = tuple(self._server_plugins())
        env = {"VLLM_PLUGINS": ",".join(plugins), **self._server_env()}
        return ServerSpec(
            key=self.key,
            model_id=self.hf_id,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            env=env,
            cli=tuple(self._server_cli()),
            plugins=plugins,
        )

    def teardown(self) -> None:
        del self._llm
        self._llm = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
