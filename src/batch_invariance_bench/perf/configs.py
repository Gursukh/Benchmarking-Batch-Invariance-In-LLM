from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ServerConfig:
    """vllm serve config for one perf engine. `name` is used as the engine
    label in output filenames; keep it matched to the correctness harness."""

    name: str
    model_id: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    extra_env: dict[str, str] = field(default_factory=dict)
    extra_cli: list[str] = field(default_factory=list)


DEFAULT_ENGINES: list[ServerConfig] = [
    ServerConfig(name="VLLMDefault"),
    ServerConfig(
        name="VLLMTMBatchInvariant",
        extra_env={"VLLM_BATCH_INVARIANT": "1"},
    ),
]


ENGINE_ALIASES: dict[str, ServerConfig] = {
    "default": DEFAULT_ENGINES[0],
    "tm":      DEFAULT_ENGINES[1],
}
