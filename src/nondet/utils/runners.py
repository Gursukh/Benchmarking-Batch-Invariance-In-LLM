"""Op-class map and the HF/vLLM runners the experiments share.

Backends (transformers, vllm) are imported lazily in load(), so importing this
module and the op-class helpers stays cheap.
"""

from __future__ import annotations

import gc
import re

import torch

# --- op map

OP_CLASSES = ("gemm", "rmsnorm", "softmax", "attention")

_GEMM_LEAVES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
}
_RMSNORM_LEAVES = {
    "input_layernorm",
    "post_attention_layernorm",
    "q_norm",
    "k_norm",
    "norm",
}
_ATTENTION_LEAVES = {"self_attn"}

_LAYER_RE = re.compile(r"\blayers\.(\d+)\b")


def _leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def classify(name: str) -> str | None:
    leaf = _leaf(name)
    if leaf in _GEMM_LEAVES:
        return "gemm"
    if leaf in _RMSNORM_LEAVES:
        return "rmsnorm"
    if leaf in _ATTENTION_LEAVES:
        return "attention"
    return None


def build_module_map(model) -> dict[str, str]:
    """{module_name: op_class} for every submodule we can classify."""
    out = {}
    for name, _ in model.named_modules():
        cls = classify(name)
        if cls is not None:
            out[name] = cls
    return out


def layer_index_of(name: str) -> int | None:
    """Decoder-layer index from a module name, or None for top-level modules."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def inventory(model) -> list[dict]:
    """One record per submodule; unclassified modules get an empty op_class."""
    rows = []
    for name, mod in model.named_modules():
        if name == "":
            continue  # the top-level model wrapper itself
        rows.append(
            {
                "module_name": name,
                "leaf": _leaf(name),
                "module_type": type(mod).__name__,
                "op_class": classify(name) or "",
                "layer_idx": layer_index_of(name),
                # recurse=False so a leaf's params aren't also counted on its parents
                "n_params": sum(p.numel() for p in mod.parameters(recurse=False)),
            }
        )
    return rows


# --- HF runner

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class HFRunner:
    def __init__(
        self,
        hf_id: str,
        dtype: str = "bfloat16",
        attn_impl: str = "eager",
        device: str | None = None,
    ):
        self.hf_id = hf_id
        self.dtype = _DTYPES.get(dtype, torch.bfloat16)
        self.attn_impl = attn_impl
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load(self) -> "HFRunner":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.hf_id, dtype=self.dtype, attn_implementation=self.attn_impl
            )
            .to(self.device)
            .eval()
        )
        return self

    def module_map(self) -> dict[str, str]:
        return build_module_map(self.model)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens: int = 256) -> dict:
        """Greedy-decode one prompt: text, token_ids, per-step logprobs, final next-token dist."""
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(self.device)
        attn = enc.attention_mask.to(self.device)
        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        seq = gen.sequences[0]
        gen_ids = seq[input_ids.shape[1] :]
        step_logprobs = [
            float(torch.log_softmax(score[0].float(), dim=-1)[int(tid)].item())
            for score, tid in zip(gen.scores, gen_ids)
        ]
        out = self.model(input_ids=seq.unsqueeze(0), use_cache=False)
        final = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
        return {
            "text": self.tokenizer.decode(gen_ids, skip_special_tokens=True),
            "token_ids": [int(t) for t in gen_ids.tolist()],
            "step_logprobs": step_logprobs,
            "final_logprob_dist": final.cpu().tolist(),
        }

    def free(self):
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- vLLM runner


class VLLMRunner:
    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        quantization: str | None = None,
        enforce_eager: bool = True,
        seed: int = 0,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        max_logprobs: int | None = None,
        tokenizer: str | None = None,
        **extra,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.quantization = quantization
        self.enforce_eager = enforce_eager
        self.seed = seed
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_logprobs = max_logprobs
        # explicit tokenizer path for vLLM; None uses model_id
        self.tokenizer = tokenizer
        self.extra = extra
        self.llm = None

    def load(self) -> "VLLMRunner":
        from vllm import LLM

        kwargs = dict(
            model=self.model_id,
            dtype=self.dtype,
            enforce_eager=self.enforce_eager,
            seed=self.seed,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=False,
            tensor_parallel_size=1,
        )
        if self.quantization:
            kwargs["quantization"] = self.quantization
        if self.max_logprobs is not None:
            kwargs["max_logprobs"] = self.max_logprobs
        if self.tokenizer is not None:
            kwargs["tokenizer"] = self.tokenizer
        kwargs.update(self.extra)
        self.llm = LLM(**kwargs)
        return self

    def vocab_size(self) -> int:
        try:
            return self.llm.llm_engine.model_config.get_vocab_size()
        except Exception:
            return len(self.llm.get_tokenizer())

    def _dense_logprobs(self, logprob_dict, vocab: int) -> torch.Tensor:
        vec = torch.full((vocab,), float("-inf"))
        for tid, lp in logprob_dict.items():
            val = lp.logprob if hasattr(lp, "logprob") else float(lp)
            if 0 <= tid < vocab:
                vec[tid] = val
        return vec

    def target_next_token_logprobs(self, prompts: list[str]) -> torch.Tensor:
        """Full-vocab next-token log-softmax for prompt 0 (rest are batch padding)."""
        from vllm import SamplingParams

        vocab = self.vocab_size()
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=vocab)
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        return self._dense_logprobs(outs[0].outputs[0].logprobs[0], vocab)

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        logprobs: int | None = None,
        temperature: float = 0.0,
    ):
        """Generate; returns a dict per prompt with text, token_ids, step_logprobs."""
        from vllm import SamplingParams

        sp = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, logprobs=logprobs
        )
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        results = []
        for o in outs:
            comp = o.outputs[0]
            results.append(
                {
                    "text": comp.text,
                    "token_ids": list(comp.token_ids),
                    "step_logprobs": comp.logprobs,  # None unless logprobs requested
                    "finish_reason": comp.finish_reason,  # "stop" / "length"
                }
            )
        return results

    def teardown(self):
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        del self.llm
        self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
