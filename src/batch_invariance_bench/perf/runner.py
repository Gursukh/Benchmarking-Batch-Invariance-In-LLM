from __future__ import annotations

import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from batch_invariance_bench.common.csvio import now, slug
from batch_invariance_bench.common.gpu import gpu_info, vllm_version
from batch_invariance_bench.engines.base import VLLMBase
from batch_invariance_bench.engines.default import VLLMDefault
from batch_invariance_bench.engines.tm_batch_invariant import VLLMTMBatchInvariant
from batch_invariance_bench.perf.load_test import run_load_test, warmup
from batch_invariance_bench.perf.schema import (
    PERF_COLUMNS,
    append_perf_row,
    perf_csv_path,
)


def perf_log_path(out_dir: Path, gpu_name: str, run_id: str) -> Path:
    return out_dir / f"{slug(gpu_name)}.{run_id}.perf.log"


@contextmanager
def _tee_fd_to_file(log_path: Path) -> Iterator[Path]:
    """Mirror stdout and stderr into log_path while still printing them.

    Redirects fds 1 and 2 so output from vLLM and its subprocesses is caught
    too. A pump thread copies each pipe to both the terminal and the log.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "ab", buffering=0)

    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)

    r_out, w_out = os.pipe()
    r_err, w_err = os.pipe()
    os.dup2(w_out, 1)
    os.dup2(w_err, 2)
    os.close(w_out)
    os.close(w_err)

    def _pump(read_fd: int, mirror_fd: int) -> None:
        try:
            while True:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    break
                try:
                    os.write(mirror_fd, chunk)
                except OSError:
                    pass
                try:
                    log_file.write(chunk)
                except OSError:
                    pass
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass

    t_out = threading.Thread(target=_pump, args=(r_out, saved_stdout_fd), daemon=True)
    t_err = threading.Thread(target=_pump, args=(r_err, saved_stderr_fd), daemon=True)
    t_out.start()
    t_err.start()

    try:
        yield log_path
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        t_out.join(timeout=5.0)
        t_err.join(timeout=5.0)
        log_file.close()


# Same work for every engine: ignore_eos forces exactly max_tokens, and
# temperature 0 keeps decoding deterministic.
DEFAULT_SAMPLING_PARAMS: dict = {"ignore_eos": True, "temperature": 0}


def default_engines() -> list[VLLMBase]:
    return [VLLMDefault(), VLLMTMBatchInvariant()]


def _setup_engine_for_cell(
    engine: VLLMBase, concurrency: int, mean_input_tokens: int
) -> dict:
    """Set per-cell vllm_kwargs and call engine.setup().

    Floors max_num_batched_tokens at 2048 so small concurrencies still get a
    healthy budget. Returns the saved kwargs so the caller can restore them.
    """
    saved = dict(engine.vllm_kwargs)
    engine.vllm_kwargs["max_num_seqs"] = concurrency
    engine.vllm_kwargs["max_num_batched_tokens"] = max(
        concurrency * mean_input_tokens, 2048
    )
    try:
        engine.setup()
    except Exception:
        engine.vllm_kwargs = saved
        raise
    return saved


def run(
    engines: Sequence[VLLMBase] | None = None,
    concurrency: Sequence[int] = (1, 4, 16, 64),
    mean_input_tokens: int = 550,
    stddev_input_tokens: int = 150,
    mean_output_tokens: int = 150,
    stddev_output_tokens: int = 0,
    max_requests: int = 500,
    requests_per_concurrency: int = 20,
    repeats: int = 3,
    warmup_requests: int = 5,
    sampling_params: dict | None = None,
    seed: int = 0,
    use_random_tokens: bool = False,
    target_duration_s: float | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Run every engine, concurrency and repeat cell against in-process vLLM.

    Each cell builds a fresh LLM at max_num_seqs=concurrency, runs one batch,
    and appends the per-request and throughput stats to the engine CSV.
    Refuses to run with more than one GPU visible since we only read device 0.
    """

    out_dir = Path(out_path) if out_path else Path("data/perf")
    out_dir.mkdir(parents=True, exist_ok=True)

    if engines is None:
        engines = default_engines()
    if sampling_params is None:
        sampling_params = dict(DEFAULT_SAMPLING_PARAMS)

    run_id = uuid.uuid4().hex[:12]
    arch, gpu_name = gpu_info()
    vllm_v = vllm_version()

    cells = [
        (engine, conc, r)
        for engine in engines
        for conc in concurrency
        for r in range(repeats)
    ]

    log_path = perf_log_path(out_dir, gpu_name, run_id)
    print(
        f"[perf] out_dir={out_dir} run_id={run_id} cells={len(cells)} "
        f"engines={[e.name for e in engines]} concurrency={list(concurrency)} "
        f"repeats={repeats}",
        flush=True,
    )
    print(f"[perf] log={log_path}", flush=True)

    with _tee_fd_to_file(log_path):
        for engine, conc, repeat_idx in cells:
            t_cell = time.perf_counter()
            csv_path = perf_csv_path(out_dir, gpu_name, run_id, engine.name)
            effective_max_requests = max(max_requests, requests_per_concurrency * conc)

            tag = f"{engine.name} | c={conc} | r={repeat_idx}"
            print(f"[{tag}] setup...", flush=True)
            error: str | None = None
            test_summary: dict = {}
            t_run = 0.0

            saved_kwargs: dict | None = None
            try:
                saved_kwargs = _setup_engine_for_cell(
                    engine, conc, mean_input_tokens
                )
                print(
                    f"[{tag}] ready ({time.perf_counter() - t_cell:.1f}s)",
                    flush=True,
                )
                warmup(
                    engine.llm,
                    engine.hf_id,
                    warmup_requests,
                    conc,
                    sampling_params=sampling_params,
                    seed=seed,
                )

                t_run_start = time.perf_counter()
                test_summary = run_load_test(
                    llm=engine.llm,
                    model_id=engine.hf_id,
                    concurrency=conc,
                    max_requests=effective_max_requests,
                    mean_input_tokens=mean_input_tokens,
                    stddev_input_tokens=stddev_input_tokens,
                    mean_output_tokens=mean_output_tokens,
                    stddev_output_tokens=stddev_output_tokens,
                    sampling_params=sampling_params,
                    seed=seed,
                    use_random_tokens=use_random_tokens,
                    target_duration_s=target_duration_s,
                )
                t_run = time.perf_counter() - t_run_start
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                print(f"[{tag}] ERROR\n{traceback.format_exc()}", flush=True)
            finally:
                try:
                    engine.teardown()
                except Exception as e:
                    print(f"[{tag}] teardown error: {e}", flush=True)
                if saved_kwargs is not None:
                    engine.vllm_kwargs = saved_kwargs
                print(f"[{tag}] teardown", flush=True)

            row = {col: "" for col in PERF_COLUMNS}
            row.update(
                {
                    "run_id": run_id,
                    "gpu_arch": arch,
                    "gpu_name": gpu_name,
                    "engine": engine.name,
                    "engine_label": engine.label,
                    "vllm_version": vllm_v,
                    "model_id": engine.hf_id,
                    "concurrency": conc,
                    "repeat_idx": repeat_idx,
                    "mean_input_tokens": mean_input_tokens,
                    "stddev_input_tokens": stddev_input_tokens,
                    "mean_output_tokens": mean_output_tokens,
                    "stddev_output_tokens": stddev_output_tokens,
                    "effective_max_requests": effective_max_requests,
                    "duration_s": round(t_run, 3),
                    "timestamp": now(),
                    "error": error or "",
                }
            )
            row.update({k: v for k, v in test_summary.items() if v is not None})
            append_perf_row(csv_path, row)

            print(
                f"[{tag}] done ({time.perf_counter() - t_cell:.1f}s cell, "
                f"{t_run:.1f}s measured)",
                flush=True,
            )

    return out_dir
