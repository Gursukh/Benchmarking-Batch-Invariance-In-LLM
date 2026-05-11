from __future__ import annotations

import atexit
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx

from batch_invariance_bench.perf.configs import ServerConfig


class VLLMServer:
    """vllm serve in a subprocess. start() blocks until /v1/models responds;
    stop() does SIGTERM with a SIGKILL fallback."""

    def __init__(self, cfg: ServerConfig, port: int, log_path: Path) -> None:
        self.cfg = cfg
        self.port = port
        self.log_path = log_path
        self._proc: subprocess.Popen | None = None
        self._log_fd = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def command(self) -> list[str]:
        return [
            "vllm", "serve", self.cfg.model_id,
            "--port", str(self.port),
            "--dtype", self.cfg.dtype,
            "--max-model-len", str(self.cfg.max_model_len),
            *self.cfg.extra_cli,
        ]

    def start(self, timeout_s: float = 180.0) -> None:
        if self._proc is not None:
            raise RuntimeError("server already started")

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fd = self.log_path.open("w")

        env = {**os.environ, **self.cfg.extra_env}
        self._proc = subprocess.Popen(
            self.command(),
            env=env,
            stdout=self._log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        atexit.register(self.stop)

        deadline = time.monotonic() + timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited with code {self._proc.returncode} before "
                    f"becoming ready; check {self.log_path}"
                )
            try:
                r = httpx.get(f"{self.base_url}/models", timeout=2.0)
                if r.status_code == 200:
                    return
            except Exception as e:
                last_err = e
            time.sleep(1.0)

        self.stop()
        raise TimeoutError(
            f"vllm serve did not become ready in {timeout_s:.0f}s "
            f"(last health probe: {last_err}); see {self.log_path}"
        )

    def stop(self, timeout_s: float = 30.0) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None

        if proc.poll() is None:
            try:
                # kill the whole process group so the workers go with it
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5.0)

        if self._log_fd is not None:
            self._log_fd.close()
            self._log_fd = None
