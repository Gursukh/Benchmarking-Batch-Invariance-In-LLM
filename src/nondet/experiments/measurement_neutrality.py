"""Per-run decode dump: text, token_ids, per-step logprobs, final next-token dist,
for every (run, instrumentation) on each platform.

The instrument axis shows measurement neutrality: adding observability must not change
the output. On HF that is read-only forward hooks (see _register_hf_hooks); vLLM has no
hooks, so the logprobs request is toggled instead. Neutrality holds if text/token_ids
match across the on/off pairs and both engines.
"""

from __future__ import annotations

import json

from nondet.utils import prompts, results

FILENAME = "measurement_neutrality.csv"

# hardcoded sweep: full AIME 2025, bf16, both platforms and instrumentations
MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-30B-A3B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]
DTYPE = "bfloat16"
PROMPT_SET = "aime2025"
MAX_TOKENS = 256
RUNS = (1, 2)
INSTRUMENT = (False, True)  # bare, then instrumented

ROW_FIELDS = [
    "platform",
    "prompt_id",
    "run",
    "instrument",  # was instrumentation active
    "text",
    "token_ids",  # JSON list of token ids
    "log_probs",  # JSON list of per-step chosen-token logprobs
    "final_logprob_dist",  # JSON list, full-vocab next-token logprobs
]


def _row(env, platform, prompt_id, run, instrument, text, token_ids, log_probs, final_dist):
    return {
        **env,
        "platform": platform,
        "prompt_id": prompt_id,
        "run": run,
        "instrument": instrument,
        "text": text,
        "token_ids": json.dumps(token_ids),
        "log_probs": json.dumps(log_probs),
        "final_logprob_dist": json.dumps(final_dist),
    }


def _register_hf_hooks(runner):
    """Read-only forward hook per classified submodule (detach and stash the activation).
    Returns the handles to remove afterward."""
    model = runner.model
    store, handles = {}, []
    for name in runner.module_map():
        module = model.get_submodule(name)

        def hook(_mod, _inp, out, _name=name):
            t = out[0] if isinstance(out, (tuple, list)) else out
            store[_name] = t.detach()[0]

        handles.append(module.register_forward_hook(hook))
    return handles


def hf_rows(model_id, dtype, prompt_list, env):
    from nondet.utils.runners import HFRunner

    runner = HFRunner(model_id, dtype=dtype).load()
    rows = []
    for p in prompt_list:
        text = prompts.apply_chat(runner.tokenizer, p.text)
        for run in RUNS:
            for instrument in INSTRUMENT:
                handles = _register_hf_hooks(runner) if instrument else []
                try:
                    g = runner.generate(text, max_new_tokens=MAX_TOKENS)
                finally:
                    for h in handles:
                        h.remove()
                rows.append(
                    _row(
                        env, "hf", p.id, run, instrument,
                        g["text"], g["token_ids"], g["step_logprobs"],
                        g["final_logprob_dist"],
                    )
                )
    runner.free()
    return rows


def _chosen_logprob(step_dict, tid):
    lp = step_dict.get(tid) if step_dict else None
    if lp is None:
        return float("nan")
    return lp.logprob if hasattr(lp, "logprob") else float(lp)


def vllm_rows(model_id, dtype, prompt_list, env):
    from nondet.utils.runners import VLLMRunner

    runner = VLLMRunner(model_id, dtype=dtype, max_logprobs=200_000).load()
    tok = runner.llm.get_tokenizer()
    rows = []
    for p in prompt_list:
        text = prompts.apply_chat(tok, p.text)
        for run in RUNS:
            for instrument in INSTRUMENT:
                # no forward hooks on vLLM; toggle the logprobs request instead
                g = runner.generate(
                    [text], max_tokens=MAX_TOKENS,
                    logprobs=1 if instrument else None,
                )[0]
                tok_ids = g["token_ids"]
                if instrument:
                    steps = g["step_logprobs"] or []
                    log_probs = [_chosen_logprob(sd, tid) for sd, tid in zip(steps, tok_ids)]
                    final = runner.target_next_token_logprobs([text + g["text"]]).tolist()
                else:
                    log_probs = []
                    final = []
                rows.append(
                    _row(env, "vllm", p.id, run, instrument, g["text"], tok_ids, log_probs, final)
                )
    runner.teardown()
    return rows


def _error_row(env, platform, exc):
    return {
        **env,
        "platform": platform,
        "prompt_id": "-",
        "run": "-",
        "instrument": "-",
        "text": f"error:{type(exc).__name__}: {exc}",
        "token_ids": "",
        "log_probs": "",
        "final_logprob_dist": "",
    }


def main():
    fieldnames = results.ENV_FIELDS + ROW_FIELDS
    prompt_list = prompts.prompt_set(PROMPT_SET, n=None)  # the whole AIME 2025 set

    rows = []
    for model_id in MODELS:
        env = results.env_columns(model=model_id, dtype=DTYPE)
        try:
            rows += hf_rows(model_id, DTYPE, prompt_list, env)
        except Exception as e:
            rows.append(_error_row(env, "hf", e))
        try:
            rows += vllm_rows(model_id, DTYPE, prompt_list, env)
        except Exception as e:
            rows.append(_error_row(env, "vllm", e))

    path = results.write_rows(FILENAME, rows, fieldnames)
    print(f"[measurement_neutrality] wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
