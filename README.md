# Benchmarking Batch Invariance in LLMs

Tools for measuring how much an LLM's output drifts when the batch it's decoded in
changes, and what the batch-invariance fixes cost you in throughput. vLLM's default
kernels aren't batch-invariant, so the same prompt can produce different tokens
depending on what else is in the batch. This repo lets you quantify that and compare
it against batch-invariant alternatives.

It ships two things:

- **`batch_invariance_bench`** - a library for running perf and correctness sweeps
  across different engine configurations.
- **`nondet`** - a small CLI of standalone experiments on batch-induced
  non-determinism.

## Install

```bash
pip install -e .
# for the fixed-point reduction engine see https://github.com/Gursukh/Fixed-Point-Reductions-For-vLLM 
```

Needs a CUDA GPU and `vllm==0.20.2`.

## The library

Three engine variants on the same model: stock vLLM (`VLLMBase`), Thinking Machines'
batch-invariant ops (`VLLMTMBatchInvariant`), and fixed-point reductions
(`VLLMFxpr`). Point them at the perf or correctness runners.

```python
from batch_invariance_bench.perf import run, default_engines
from batch_invariance_bench.correctness import run_correctness, run_accuracy
from batch_invariance_bench.engines import VLLMBase, VLLMTMBatchInvariant
from batch_invariance_bench.tasks import MATH500, IFEval

# throughput + latency across concurrency levels
run(engines=default_engines(), concurrency=(1, 4, 16, 64))

# dump raw outputs per batch size for offline divergence analysis
run_correctness([VLLMBase(), VLLMTMBatchInvariant()], tasks=[MATH500()])

# accuracy at one concurrency, scored inline
run_accuracy(VLLMBase(), tasks=[MATH500(), IFEval()])
```

Results are written as CSVs under `results/` (and `data/perf/` for perf runs).

## The experiments

```bash
nondet                                              # list experiments
nondet baseline_divergence --model Qwen/Qwen3-0.6B
nondet module_inventory --model Qwen/Qwen3-0.6B
```

`--model` is a Hugging Face id. CSVs land in `results/Chapter3/` (override with
`NONDET_RESULTS_DIR`).

## Building a wheel

```bash
uv build --wheel    # -> dist/
```

The wheel bundles both the library and the `nondet` command.