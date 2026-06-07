"""Vendored IFEval instruction checker (Apache-2.0).

See the per-module headers for provenance. Only the instruction registry is
re-exported; the IFEval task grader builds and runs instructions from it.
"""

from batch_invariance_bench.tasks._ifeval.instructions_registry import (
    INSTRUCTION_DICT,
)

__all__ = ["INSTRUCTION_DICT"]
