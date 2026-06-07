"""Output-level divergence under batching: decode each prompt at index 0 of a batch
of B, padded with random-token prompts.

Raw data only (no metrics); one CSV row per (model, prompt, batch_size), with tokens
and top-k logprobs packed into JSON cells. One padding set per B, seeded on B so it is
identical across model runs.
"""

from __future__ import annotations

import json
import random
import time

from nondet.utils import prompts, results

EXP_NAME = "baseline_divergence"
FIELDS = [
    "prompt_id",
    "source",
    "reference",
    "batch_size",
    "pad_seed",
    "thinking",
    "completion_len",
    "finish_reason",
    "completion_text",
    "tokens",  # JSON list of decoded token ids
    "logprobs",  # JSON list of per-position top-k {token_id: logprob} dicts
]


def _seed(base: int, B: int) -> int:
    return hash((base, B)) & 0xFFFFFFFF


def _random_pads(tok, n: int, pad_len: int, seed: int, vocab: int) -> list[str]:
    """n chat-wrapped padding prompts of pad_len random tokens; [] when n <= 0 (B=1)."""
    rng = random.Random(seed)
    pads = []
    for _ in range(max(n, 0)):
        # sample the real vocab so every id decodes to a token
        ids = [rng.randrange(vocab) for _ in range(pad_len)]
        text = tok.decode(ids, skip_special_tokens=True)
        pads.append(prompts.apply_chat(tok, text, {"enable_thinking": False}))
    return pads


def _topk_dict(step_dict, k: int) -> dict[int, float]:
    """Top-k {token_id: logprob} from a vLLM step dict, ordered by logprob desc."""
    items = sorted(step_dict.items(), key=lambda kv: kv[1].logprob, reverse=True)
    return {tid: lp.logprob for tid, lp in items[:k]}


def main():
    ap = results.base_parser()
    ap.add_argument("--prompts", type=int, default=30)
    ap.add_argument("--prompt-set", default="aime2025")
    ap.add_argument("--batch-sizes", default="1,4,16,64,256")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--pad-len", type=int, default=256)
    ap.add_argument("--pad-seed", type=int, default=0)
    ap.add_argument("--no-thinking", action="store_true")
    args = ap.parse_args()

    model_id = args.model
    env = results.env_from_args(args)
    thinking = not args.no_thinking
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    targets = prompts.prompt_set(args.prompt_set, n=args.prompts)

    from nondet.utils.runners import VLLMRunner

    # corrected tokenizer for repos that mis-decode; vLLM gets the matching one
    tok = prompts.load_tokenizer(model_id)
    runner = VLLMRunner(
        model_id,
        dtype=args.dtype,
        max_model_len=max(4096, args.max_tokens + args.pad_len + 1024),
        tokenizer=prompts.vllm_tokenizer(model_id),
    ).load()
    # real vocab, not the model's padded get_vocab_size() with non-decoding reserved ids
    tok_vocab = getattr(tok, "vocab_size", None) or len(tok)

    # one padding set per batch size, reused for every target prompt
    pad_seeds = {B: _seed(args.pad_seed, B) for B in batch_sizes}
    pads_by_B = {
        B: _random_pads(tok, B - 1, args.pad_len, pad_seeds[B], tok_vocab)
        for B in batch_sizes
    }

    # append each row as it is generated, so a partial run survives a crash
    path = results.experiment_path(EXP_NAME, results.gpu_model_filename(model_id))
    fieldnames = results.ENV_FIELDS + FIELDS
    n_total = len(targets) * len(batch_sizes)
    n_done = 0
    for p in targets:
        text = prompts.apply_chat(tok, p.text, {"enable_thinking": thinking})
        for B in batch_sizes:
            t0 = time.perf_counter()
            tgt = runner.generate(
                [text, *pads_by_B[B]], max_tokens=args.max_tokens, logprobs=args.topk
            )[0]
            dt = time.perf_counter() - t0
            token_ids = tgt["token_ids"]
            top_logprobs = [_topk_dict(step, args.topk) for step in tgt["step_logprobs"] or []]
            row = {
                **env,
                "timestamp": results.now(),  # row write time
                "prompt_id": p.id,
                "source": p.source,
                "reference": p.reference or "",
                "batch_size": B,
                "pad_seed": pad_seeds[B],
                "thinking": thinking,
                "completion_len": len(token_ids),
                "finish_reason": tgt["finish_reason"],
                "completion_text": tgt["text"],
                "tokens": json.dumps(token_ids),
                "logprobs": json.dumps(top_logprobs),
            }
            results.append_csv_rows(path, [row], fieldnames)
            n_done += 1
            print(
                f"[baseline_divergence] {n_done}/{n_total} "
                f"prompt={p.id} B={B} len={row['completion_len']} "
                f"finish={row['finish_reason']} {dt:.1f}s -> {path}",
                flush=True,
            )
    runner.teardown()

    print(
        f"[baseline_divergence] wrote {n_done} rows "
        f"({len(targets)} prompts x {len(batch_sizes)} batch sizes) -> {path}"
    )


if __name__ == "__main__":
    main()
