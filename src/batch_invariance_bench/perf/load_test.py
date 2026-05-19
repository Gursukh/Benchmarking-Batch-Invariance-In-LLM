"""In-house closed-loop load generator.

Replaces llmperf (unmaintained, Ray-based, Python <3.11 only). Drives a vLLM
OpenAI server over streaming /v1/completions, keeping `concurrency` requests in
flight, and returns TTFT, ITL, E2E and throughput in the CSV row schema.

ITL stats pool every inter-token gap of every request, the usual definition.
Percentiles of per-request mean ITL would hide decode-stall tails.

Prompts are random token ids of a fixed length from a fixed seed, so every
engine sees the same work.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import random
import statistics
import time

import httpx

from batch_invariance_bench.common.sampling import nearest_rank


# vocab size per model id, so the tokenizer is not reloaded each cell.
_VOCAB_CACHE: dict[str, int] = {}


def _vocab_size(model_id: str) -> int:
    if model_id not in _VOCAB_CACHE:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        _VOCAB_CACHE[model_id] = int(tok.vocab_size)
    return _VOCAB_CACHE[model_id]


def _stat(values: list[float]) -> dict[str, float | None]:
    """mean, p50, p90 and p95 of values. All None if values is empty."""
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p95": None}
    return {
        "mean": statistics.fmean(values),
        "p50": nearest_rank(values, 0.50),
        "p90": nearest_rank(values, 0.90),
        "p95": nearest_rank(values, 0.95),
    }


def _one_request(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict,
) -> dict:
    """Run one streaming completion and time it.

    Never raises; a failure goes in the returned dict's `error` field.
    itl_gaps_s holds every inter-token gap for the caller to pool.
    """
    t_start = time.perf_counter()
    token_times: list[float] = []
    error: str | None = None
    try:
        with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", "replace")
                raise RuntimeError(f"HTTP {resp.status_code}: {body[:300]}")
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                # vLLM streams one token per chunk for /v1/completions.
                if choices and choices[0].get("text"):
                    token_times.append(time.perf_counter())
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    t_end = time.perf_counter()

    n_out = len(token_times)
    ttft = (token_times[0] - t_start) if token_times else None
    e2e = (t_end - t_start) if error is None else None
    # every gap between consecutive streamed tokens
    itl_gaps = [token_times[i + 1] - token_times[i] for i in range(n_out - 1)]
    return {
        "error": error,
        "ttft_s": ttft,
        "itl_gaps_s": itl_gaps,
        "e2e_s": e2e,
        "n_output_tokens": n_out,
        "req_output_throughput": (
            (n_out / e2e) if (error is None and n_out and e2e and e2e > 0) else None
        ),
    }


def run_load_test(
    *,
    model_id: str,
    base_url: str,
    api_key: str,
    concurrency: int,
    max_requests: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    timeout_s: float,
    sampling_params: dict | None = None,
    seed: int = 0,
) -> dict:
    """Closed-loop load test.

    Keeps `concurrency` streaming requests in flight until `max_requests`
    finish. Returns a summary dict in the perf-row schema, writes no files.
    """
    if sampling_params is None:
        sampling_params = {"ignore_eos": True, "temperature": 0}

    vocab = _vocab_size(model_id)
    rng = random.Random(seed)
    url = f"{base_url}/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Build every request up front from the fixed seed.
    specs: list[dict] = []
    for _ in range(max_requests):
        in_len = max(1, int(rng.gauss(mean_input_tokens, stddev_input_tokens)))
        out_len = max(1, int(rng.gauss(mean_output_tokens, stddev_output_tokens)))
        specs.append(
            {
                "model": model_id,
                **sampling_params,
                "prompt": [rng.randrange(vocab) for _ in range(in_len)],
                "max_tokens": out_len,
                "stream": True,
            }
        )

    limits = httpx.Limits(
        max_connections=concurrency + 8,
        max_keepalive_connections=concurrency + 8,
    )
    t0 = time.perf_counter()
    with httpx.Client(limits=limits, timeout=httpx.Timeout(timeout_s)) as client:

        def _run(payload: dict) -> dict:
            return _one_request(client, url, headers, payload)

        with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            results = list(ex.map(_run, specs))
    wall_s = time.perf_counter() - t0

    ok = [r for r in results if r["error"] is None]
    n_errors = len(results) - len(ok)
    ttft = _stat([r["ttft_s"] for r in ok if r["ttft_s"] is not None])
    # pool every inter-token gap from every completed request
    itl = _stat([gap for r in ok for gap in r["itl_gaps_s"]])
    e2e = _stat([r["e2e_s"] for r in ok if r["e2e_s"] is not None])
    req_tps = [
        r["req_output_throughput"] for r in ok if r["req_output_throughput"] is not None
    ]
    total_out = sum(r["n_output_tokens"] for r in ok)

    if results and not ok:
        first_err = next(r["error"] for r in results if r["error"])
        raise RuntimeError(f"every load-test request failed; first error: {first_err}")

    return {
        "num_completed": len(ok),
        "num_errors": n_errors,
        "error_rate": (n_errors / len(results)) if results else 0.0,
        "ttft_mean_s": ttft["mean"],
        "ttft_p50_s": ttft["p50"],
        "ttft_p90_s": ttft["p90"],
        "ttft_p95_s": ttft["p95"],
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
