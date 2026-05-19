"""Server-side batch-size sampling for the perf harness."""

from __future__ import annotations

import re
from pathlib import Path

import httpx

from batch_invariance_bench.common.sampling import BackgroundSampler, nearest_rank


# vLLM exposes these as Prometheus gauges on /metrics. A line looks like
#   vllm:num_requests_running{model_name="Qwen/Qwen3-0.6B"} 12.0
# The \b stops a partial match against a longer metric name.
_GAUGE_RE = {
    "running": re.compile(r"^vllm:num_requests_running\b.*\s([0-9.eE+-]+)\s*$"),
    "waiting": re.compile(r"^vllm:num_requests_waiting\b.*\s([0-9.eE+-]+)\s*$"),
}


class BatchSampler(BackgroundSampler):
    """Polls a vLLM server's /metrics endpoint and records the batch size.

    The running batch is a per-forward-pass number mixing prefill and decode,
    so the sampled mean is a proxy for "batch size", not an exact count. If
    /metrics is unreachable the summaries are just nan.
    """

    csv_header = ("num_requests_running", "num_requests_waiting")

    def __init__(
        self,
        base_url: str,
        hz: float = 10.0,
        out_csv: Path | None = None,
    ) -> None:
        super().__init__(hz=hz, out_csv=out_csv)
        # base_url is the OpenAI API root (".../v1"); /metrics is its sibling.
        root = base_url.rstrip("/").removesuffix("/v1").rstrip("/")
        self.metrics_url = f"{root}/metrics"
        self._client: httpx.Client | None = None

    def _open(self) -> bool:
        self._client = httpx.Client(timeout=2.0)
        return True

    def _close(self) -> None:
        if self._client is not None:
            self._client.close()

    @staticmethod
    def _parse(text: str) -> tuple[float | None, float | None]:
        running = waiting = None
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if running is None and line.startswith("vllm:num_requests_running"):
                m = _GAUGE_RE["running"].match(line)
                if m:
                    running = float(m.group(1))
            elif waiting is None and line.startswith("vllm:num_requests_waiting"):
                m = _GAUGE_RE["waiting"].match(line)
                if m:
                    waiting = float(m.group(1))
        return running, waiting

    def _sample(self) -> tuple | None:
        r = self._client.get(self.metrics_url)
        if r.status_code != 200:
            return None
        running, waiting = self._parse(r.text)
        if running is None and waiting is None:
            return None
        return (
            running if running is not None else float("nan"),
            waiting if waiting is not None else float("nan"),
        )

    def _format(self, values: tuple) -> list[str]:
        return ["" if v != v else f"{v:.0f}" for v in values]

    # summary properties; each sample is (ts, running, waiting)
    @property
    def running_mean(self) -> float:
        vals = self._column(1)
        return sum(vals) / len(vals) if vals else float("nan")

    @property
    def running_p50(self) -> float:
        return nearest_rank(self._column(1), 0.50)

    @property
    def running_p90(self) -> float:
        return nearest_rank(self._column(1), 0.90)

    @property
    def running_max(self) -> float:
        vals = self._column(1)
        return max(vals) if vals else float("nan")

    @property
    def waiting_mean(self) -> float:
        vals = self._column(2)
        return sum(vals) / len(vals) if vals else float("nan")
