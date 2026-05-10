from __future__ import annotations

import gc

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from batch_invariance_bench.engine import Engine, Sample


DEFAULT_SAMPLING: dict = {
    "temperature": 0,
    "max_tokens": 2048,
    "logprobs": 1,
}


class VLLMBase(Engine):
    """Common vLLM setup. Subclass to swap models or batch-invariance modes."""

    hf_id: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    vllm_kwargs: dict = {
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
    }

    chat_template_kwargs: dict = {"enable_thinking": False}

    default_sampling: dict = DEFAULT_SAMPLING

    def __init__(self, name: str | None = None) -> None:
        self.name = name or f"{type(self).__name__}::{self.hf_id}"
        self._llm: LLM | None = None
        self._tokenizer = None

    def setup(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self._llm = LLM(
            model=self.hf_id,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            **self.vllm_kwargs,
        )

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
            n_prompt_tokens = len(req.prompt_token_ids) if req.prompt_token_ids is not None else 0
            samples: list[Sample] = []
            for comp in req.outputs:
                token_ids = list(comp.token_ids)
                # vLLM gives us logprobs as list[dict[token_id -> Logprob]]; pull
                # out the chosen token's logprob at each step. NaN if missing.
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
                        stop_reason=str(comp.stop_reason) if comp.stop_reason is not None else "",
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
