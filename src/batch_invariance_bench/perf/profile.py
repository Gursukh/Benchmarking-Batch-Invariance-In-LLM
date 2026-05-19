"""Capture a torch-profiler trace from a vLLM server.

Runs a short load test against `vllm serve` with vLLM's built-in profiler on
(--profiler-config plus the /start_profile and /stop_profile endpoints), then
ranks the CUDA kernels in the trace.

It goes through the server on purpose: running vllm.LLM in-process inside a
notebook kernel breaks, since the forked EngineCore inherits the notebook's
fake stdout. The server subprocess has a real stdout.

    from batch_invariance_bench.engines.fxpr import VLLMFxpr
    from batch_invariance_bench.perf.profile import profile_engine
    profile_engine(VLLMFxpr(), out_path="/content/profile")

VLLMFxpr profiles correctly here: its server_spec() carries the fxpr plugin, so
the server runs the real fixed-point kernels, not stock vLLM.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from batch_invariance_bench.engines.base import ServerSpec, VLLMBase
from batch_invariance_bench.perf.load_test import run_load_test
from batch_invariance_bench.perf.runner import warmup
from batch_invariance_bench.perf.server import VLLMServer


def _as_spec(engine: ServerSpec | VLLMBase) -> ServerSpec:
    """Return a ServerSpec, accepting either a ServerSpec or an engine."""
    return engine if isinstance(engine, ServerSpec) else engine.server_spec()


def _load_trace(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with path.open() as f:
        return json.load(f)


def _rank_kernels(events: list[dict]) -> list[tuple[str, float, int]]:
    """Group CUDA-kernel events by name into (name, total_us, count), sorted desc."""
    agg: dict[str, list[float]] = {}
    for ev in events:
        if ev.get("ph") != "X" or ev.get("cat") != "kernel":
            continue
        a = agg.setdefault(ev.get("name", "?"), [0.0, 0])
        a[0] += ev.get("dur", 0)
        a[1] += 1
    rows = [(n, us, int(c)) for n, (us, c) in agg.items()]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def _format_table(rows: list[tuple[str, float, int]], limit: int) -> str:
    total = sum(r[1] for r in rows) or 1.0
    out = [f"{'kernel':<74}{'total_ms':>12}{'count':>10}{'cuda%':>8}"]
    for name, us, cnt in rows[:limit]:
        out.append(
            f"{name[:74]:<74}{us / 1000:>12.3f}{cnt:>10}{100 * us / total:>7.1f}%"
        )
    return "\n".join(out)


def profile_engine(
    engine: ServerSpec | VLLMBase,
    *,
    concurrency: int = 2,
    num_requests: int = 4,
    mean_input_tokens: int = 256,
    mean_output_tokens: int = 32,
    enforce_eager: bool = True,
    row_limit: int = 30,
    port: int = 8000,
    api_key: str = "token-abc123",
    out_path: str | Path = "data/profile",
    server_timeout_s: float = 600.0,
) -> Path:
    """Profile one engine's served prefill and decode with the torch profiler.

    `engine` may be a ServerSpec or an engine (it is converted). Writes the raw
    trace and a ranked <engine>.kernels.txt under out_path/<engine>/, prints the
    top kernels, and returns that directory.

    enforce_eager=True turns off CUDA graphs and torch.compile so single kernels
    are attributable. Set it False to profile the graph-captured path.

    The workload defaults are tiny on purpose: a few dozen forward passes rank
    kernels fine, and profiling overhead makes a big load very slow.
    """
    spec = _as_spec(engine)

    trace_dir = Path(out_path) / spec.key
    trace_dir.mkdir(parents=True, exist_ok=True)

    extra_env = {**spec.env, "VLLM_RPC_TIMEOUT": "1800000"}
    extra_cli = list(spec.cli)
    if enforce_eager and "--enforce-eager" not in extra_cli:
        extra_cli.append("--enforce-eager")
    # vLLM 0.21 adds the /start_profile and /stop_profile routes via
    # --profiler-config; the old VLLM_TORCH_PROFILER_DIR env var does not.
    # The flag takes one JSON value.
    extra_cli += [
        "--profiler-config",
        json.dumps(
            {
                "profiler": "torch",
                "torch_profiler_dir": str(trace_dir.resolve()),
                # with_stack defaults True and is expensive (a Python stack per op);
                # kernel ranking does not need it.
                "torch_profiler_with_stack": False,
            }
        ),
    ]
    cfg = dataclasses.replace(spec, env=extra_env, cli=tuple(extra_cli))

    server = VLLMServer(cfg, port=port, log_path=trace_dir / "serve.log")
    root = f"http://127.0.0.1:{port}"
    try:
        server.start(timeout_s=server_timeout_s)
        warmup(server.base_url, api_key, spec.model_id, 5, concurrency)

        with httpx.Client(timeout=300.0) as c:
            r = c.post(f"{root}/start_profile")
            if r.status_code == 404:
                raise RuntimeError(
                    "/start_profile returned 404: this vLLM build did not register "
                    "the profiler routes. Check that the installed vLLM accepts "
                    "`--profiler-config` (see serve.log for the rejected flag)."
                )
            r.raise_for_status()
        run_load_test(
            model_id=spec.model_id,
            base_url=server.base_url,
            api_key=api_key,
            concurrency=concurrency,
            max_requests=num_requests,
            mean_input_tokens=mean_input_tokens,
            stddev_input_tokens=0,
            mean_output_tokens=mean_output_tokens,
            stddev_output_tokens=0,
            timeout_s=600.0,
        )
        with httpx.Client(timeout=600.0) as c:
            c.post(f"{root}/stop_profile").raise_for_status()

        # The profiler writes the trace asynchronously; wait for it to land.
        deadline = time.time() + 60.0
        while time.time() < deadline and not list(trace_dir.glob("*.json*")):
            time.sleep(1.0)
    finally:
        server.stop()

    traces = sorted(trace_dir.glob("*.json*"))
    if not traces:
        print(f"[profile] no trace written to {trace_dir}, check serve.log", flush=True)
        return trace_dir

    rows: list[tuple[str, float, int]] = []
    for t in traces:
        try:
            rows += _rank_kernels(_load_trace(t).get("traceEvents", []))
        except Exception as e:  # noqa: BLE001 - best-effort trace parse
            print(f"[profile] could not parse {t.name}: {e}", flush=True)
    # merge duplicate kernel names across trace files
    merged: dict[str, list[float]] = {}
    for name, us, cnt in rows:
        a = merged.setdefault(name, [0.0, 0])
        a[0] += us
        a[1] += cnt
    ranked = sorted(
        ((n, us, int(c)) for n, (us, c) in merged.items()),
        key=lambda r: r[1],
        reverse=True,
    )
    table = _format_table(ranked, row_limit)
    print(f"\n=== {spec.key}: top {row_limit} CUDA kernels ===\n{table}", flush=True)
    (trace_dir / f"{spec.key}.kernels.txt").write_text(table)
    print(
        f"\n[profile] trace(s) + kernels.txt under {trace_dir}\n"
        f"[profile] open the *.json.gz in chrome://tracing or ui.perfetto.dev",
        flush=True,
    )
    return trace_dir


def profile_kernel_ncu(
    engine: ServerSpec | VLLMBase,
    *,
    kernel_regex: str,
    num_requests: int = 4,
    concurrency: int = 2,
    launch_count: int = 8,
    launch_skip: int = 0,
    metric_set: str = "full",
    port: int = 8000,
    api_key: str = "token-abc123",
    out_path: str | Path = "data/profile",
    server_timeout_s: float = 3600.0,
) -> Path:
    """Deep-profile specific CUDA kernels with Nsight Compute (ncu).

    Launches `vllm serve` under ncu, which captures `launch_count` launches of
    kernels matching kernel_regex: occupancy, tensor-core use, memory vs compute
    bound, stall reasons. Writes <engine>.ncu-rep (open in ncu-ui) and
    <engine>.ncu.txt under out_path/<engine>/, and returns that directory.

    ncu replays each captured kernel many times, so keep launch_count small;
    expect several minutes. Needs ncu on PATH and GPU counter access; on
    locked-down hosts it fails with ERR_NVGPUCTRPERM.

        profile_kernel_ncu(VLLMFxpr(), kernel_regex="_gemm_kernel")
        profile_kernel_ncu(VLLMTMBatchInvariant(), kernel_regex="matmul_kernel_persistent")
    """
    if shutil.which("ncu") is None:
        raise RuntimeError(
            "ncu (Nsight Compute) not found on PATH. Install the CUDA toolkit / "
            "nsight-compute, or use profile_engine() for a coarser torch trace."
        )
    spec = _as_spec(engine)

    out_dir = Path(out_path) / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / spec.key  # ncu appends .ncu-rep

    ncu_prefix = [
        "ncu",
        "--target-processes",
        "all",  # follow into the EngineCore subprocess
        "-f",  # overwrite an existing report
        "-o",
        str(report.resolve()),
        "-k",
        f"regex:{kernel_regex}",
        "-c",
        str(launch_count),
        "-s",
        str(launch_skip),
        "--set",
        metric_set,
    ]

    # ncu plus CUDA graphs is messy; eager launches each kernel plainly.
    extra_cli = list(spec.cli)
    if "--enforce-eager" not in extra_cli:
        extra_cli.append("--enforce-eager")
    cfg = dataclasses.replace(spec, cli=tuple(extra_cli))

    server = VLLMServer(
        cfg,
        port=port,
        log_path=out_dir / "serve_ncu.log",
        command_prefix=ncu_prefix,
    )
    try:
        # ncu slows kernels heavily; vLLM's own startup profiling pass already
        # launches the target kernels, so startup can take minutes.
        server.start(timeout_s=server_timeout_s)
        # Drive a little traffic in case startup did not hit the quota.
        try:
            warmup(server.base_url, api_key, spec.model_id, num_requests, concurrency)
        except Exception as e:  # noqa: BLE001 - warmup is best-effort here
            print(f"[ncu] warmup traffic note: {e}", flush=True)
    finally:
        server.stop()

    rep_path = report.with_suffix(".ncu-rep")
    if not rep_path.exists():
        print(
            f"[ncu] no report at {rep_path}, check {out_dir / 'serve_ncu.log'} "
            f"(ERR_NVGPUCTRPERM there = no GPU counter access)",
            flush=True,
        )
        return out_dir

    # CLI dump so the report is readable without the Nsight Compute UI.
    txt = out_dir / f"{spec.key}.ncu.txt"
    try:
        dump = subprocess.run(
            ["ncu", "--import", str(rep_path), "--page", "details"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        txt.write_text(dump.stdout)
        print(dump.stdout[-4000:], flush=True)
    except Exception as e:  # noqa: BLE001 - report dump is best-effort
        print(f"[ncu] could not dump report text: {e}", flush=True)

    print(
        f"\n[ncu] report -> {rep_path}  (open in Nsight Compute / ncu-ui)\n"
        f"[ncu] text   -> {txt}\n"
        f"[ncu] in the 'GPU Speed Of Light' section compare SM (compute) vs "
        f"Memory throughput, and check tensor-pipe use, that names the bound.",
        flush=True,
    )
    return out_dir
