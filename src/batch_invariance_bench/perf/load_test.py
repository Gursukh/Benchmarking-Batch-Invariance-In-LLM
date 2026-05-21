"""In-process load generator for offline inference.

Submits all requests at once, then drives engine.step() until they finish.
We avoid LLM.generate() because vLLM v1 leaves the metrics empty. TTFT is
reported as prefill only and as submit to first token; ITL is the pooled
inter token gaps measured at step boundaries.
"""

from __future__ import annotations

import random
import statistics
import time
from typing import Sequence

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


_VOCAB_CACHE: dict[str, int] = {}
_CORPUS_CACHE: list[str] | None = None


def _nearest_rank(values: Sequence[float], q: float) -> float:
    vals = [v for v in values if v == v]
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(len(s) - 1, int(q * (len(s) - 1) + 0.5))
    return s[idx]


def _stat(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p95": None}
    return {
        "mean": statistics.fmean(values),
        "p50": _nearest_rank(values, 0.50),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
    }


def _vocab_size(model_id: str) -> int:
    if model_id not in _VOCAB_CACHE:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        _VOCAB_CACHE[model_id] = int(tok.vocab_size)
    return _VOCAB_CACHE[model_id]


def _load_corpus() -> list[str]:
    global _CORPUS_CACHE
    if _CORPUS_CACHE is not None:
        return _CORPUS_CACHE
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    _CORPUS_CACHE = [str(row["problem"]) for row in ds.select(range(min(200, len(ds))))]
    return _CORPUS_CACHE


def _encode_prompt(
    model_id: str,
    target_len: int,
    rng: random.Random,
    use_random_tokens: bool,
) -> list[int]:
    if use_random_tokens:
        vocab = _vocab_size(model_id)
        return [rng.randrange(vocab) for _ in range(target_len)]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    corpus = _load_corpus()
    text = corpus[rng.randrange(len(corpus))]
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        vocab = _vocab_size(model_id)
        return [rng.randrange(vocab) for _ in range(target_len)]
    if len(ids) >= target_len:
        start = rng.randrange(len(ids) - target_len + 1)
        return ids[start : start + target_len]
    out: list[int] = []
    while len(out) < target_len:
        out.extend(ids)
    return out[:target_len]


def _build_sampling_params(sampling_params: dict, out_len: int) -> SamplingParams:
    return SamplingParams(max_tokens=out_len, **sampling_params)


def warmup(
    llm: LLM,
    model_id: str,
    n: int,
    concurrency: int,
    sampling_params: dict | None = None,
    seed: int = 0,
) -> None:
    """Warmup batch at the target concurrency so kernels are hot."""
    n = max(n, 2 * concurrency)
    if n <= 0:
        return
    if sampling_params is None:
        sampling_params = {"ignore_eos": True, "temperature": 0}
    vocab = _vocab_size(model_id)
    rng = random.Random(seed)
    prompts = [
        TokensPrompt(prompt_token_ids=[rng.randrange(vocab) for _ in range(16)])
        for _ in range(n)
    ]
    params = [_build_sampling_params(sampling_params, 8) for _ in range(n)]
    llm.generate(prompts, params, use_tqdm=False)


def run_load_test(
    *,
    llm: LLM,
    model_id: str,
    concurrency: int,
    max_requests: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    sampling_params: dict | None = None,
    seed: int = 0,
    use_random_tokens: bool = False,
    target_duration_s: float | None = None,
) -> dict:
    """Submit up to max_requests and drive engine.step() to completion.

    Returns a dict for the perf row schema. With target_duration_s set, stop
    adding requests once the time budget passes; in flight ones still finish.
    """
    if sampling_params is None:
        sampling_params = {"ignore_eos": True, "temperature": 0}

    rng = random.Random(seed)
    items: list[tuple[str, TokensPrompt, SamplingParams, int]] = []
    for i in range(max_requests):
        in_len = max(1, int(rng.gauss(mean_input_tokens, stddev_input_tokens)))
        out_len = max(1, int(rng.gauss(mean_output_tokens, stddev_output_tokens)))
        token_ids = _encode_prompt(model_id, in_len, rng, use_random_tokens)
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        params = _build_sampling_params(sampling_params, out_len)
        items.append((f"req-{i}", prompt, params, i))

    engine = llm.llm_engine
    t_arrival: dict[str, float] = {}
    t_first_token: dict[str, float] = {}
    t_finish: dict[str, float] = {}
    n_output: dict[str, int] = {}
    arrival_order: dict[str, int] = {}
    itl_gaps: list[float] = []
    last_step_emit_t: dict[str, float] = {}
    submitted: set[str] = set()

    t0 = time.perf_counter()
    for rid, prompt, params, idx in items:
        if target_duration_s is not None and time.perf_counter() - t0 > target_duration_s:
            break
        engine.add_request(rid, prompt, params)
        t_arrival[rid] = time.perf_counter()
        arrival_order[rid] = idx
        submitted.add(rid)

    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        now = time.perf_counter()
        for out in step_outputs:
            rid = out.request_id
            if out.outputs:
                new_total = len(out.outputs[0].token_ids)
                prev_total = n_output.get(rid, 0)
                delta = new_total - prev_total
                if delta > 0:
                    if rid not in t_first_token:
                        t_first_token[rid] = now
                    if rid in last_step_emit_t:
                        itl_gaps.append(now - last_step_emit_t[rid])
                        itl_gaps.extend([0.0] * (delta - 1))
                    else:
                        itl_gaps.extend([0.0] * (delta - 1))
                    last_step_emit_t[rid] = now
                    n_output[rid] = new_total
            if out.finished:
                t_finish[rid] = now

    wall_s = time.perf_counter() - t0

    ttfts_prefill: list[float] = []
    ttfts_submit: list[float] = []
    e2es: list[float] = []
    req_tps: list[float] = []
    n_errors = 0
    n_completed = 0
    total_out = 0
    first_err: str | None = None

    for rid, _, _, _ in items:
        if rid not in submitted:
            continue
        if rid not in t_finish:
            n_errors += 1
            if first_err is None:
                first_err = f"request {rid} did not finish"
            continue
        n_out = n_output.get(rid, 0)
        if n_out == 0:
            n_errors += 1
            if first_err is None:
                first_err = f"request {rid} produced zero tokens"
            continue
        n_completed += 1
        total_out += n_out
        e2e = t_finish[rid] - t_arrival[rid]
        e2es.append(e2e)
        if rid in t_first_token:
            stft = t_first_token[rid] - t_arrival[rid]
            ttfts_submit.append(stft)
            if arrival_order[rid] < concurrency:
                ttfts_prefill.append(stft)
        if e2e > 0:
            req_tps.append(n_out / e2e)

    if n_completed == 0:
        raise RuntimeError(
            f"every offline-inference request failed; first error: {first_err}"
        )

    ttft = _stat(ttfts_prefill)
    stft = _stat(ttfts_submit)
    itl = _stat(itl_gaps)
    e2e = _stat(e2es)
    n_total = n_completed + n_errors

    return {
        "num_completed": n_completed,
        "num_errors": n_errors,
        "error_rate": (n_errors / n_total) if n_total else 0.0,
        "ttft_mean_s": ttft["mean"],
        "ttft_p50_s": ttft["p50"],
        "ttft_p90_s": ttft["p90"],
        "ttft_p95_s": ttft["p95"],
        "submit_to_first_token_mean_s": stft["mean"],
        "submit_to_first_token_p50_s": stft["p50"],
        "submit_to_first_token_p90_s": stft["p90"],
        "submit_to_first_token_p95_s": stft["p95"],
        "itl_mean_s": itl["mean"],
        "itl_p50_s": itl["p50"],
        "itl_p90_s": itl["p90"],
        "itl_p95_s": itl["p95"],
        "e2e_mean_s": e2e["mean"],
        "e2e_p50_s": e2e["p50"],
        "e2e_p90_s": e2e["p90"],
        "e2e_p95_s": e2e["p95"],
        "req_output_throughput_mean": statistics.fmean(req_tps) if req_tps else None,
        "overall_output_throughput": (total_out / wall_s) if wall_s > 0 else None,
    }
