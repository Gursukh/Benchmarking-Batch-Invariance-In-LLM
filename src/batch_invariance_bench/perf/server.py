from __future__ import annotations

import atexit
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx

from batch_invariance_bench.engines.base import ServerSpec


class VLLMServer:
    """Runs `vllm serve` in a subprocess.

    start() blocks until /v1/models responds; stop() sends SIGTERM, then
    SIGKILL if needed. Built from a ServerSpec, so a served run matches the
    in-process engine's model, dtype, env, CLI and plugins.
    """

    def __init__(
        self,
        spec: ServerSpec,
        port: int,
        log_path: Path,
        command_prefix: list[str] | None = None,
    ) -> None:
        self.spec = spec
        self.port = port
        self.log_path = log_path
        # Optional wrapper (like ncu) put in front of the vllm serve command.
        self._command_prefix = list(command_prefix or [])
        self._proc: subprocess.Popen | None = None
        self._log_fd = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def pgid(self) -> int | None:
        """Process-group id of the server, or None if it is not running.

        It leads the group, so every vLLM worker shares this pgid.
        """
        if self._proc is None:
            return None
        try:
            return os.getpgid(self._proc.pid)
        except ProcessLookupError:
            return None

    def command(self) -> list[str]:
        return [
            *self._command_prefix,
            "vllm",
            "serve",
            self.spec.model_id,
            "--port",
            str(self.port),
            "--dtype",
            self.spec.dtype,
            "--max-model-len",
            str(self.spec.max_model_len),
            *self.spec.cli,
        ]

    def start(self, timeout_s: float = 180.0) -> None:
        if self._proc is not None:
            raise RuntimeError("server already started")

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fd = self.log_path.open("w")

        env = {**os.environ, **self.spec.env}
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
