# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Triton on-device realization of the framework hook contract (``Ctx``).

The DSL-agnostic field set (see ``comms.dsl.ctx.Ctx``) realized as a
Triton ``@_aggregate`` struct, so a hook takes a single ``ctx`` instead of a flat
positional arg list. Adding a field here never changes a hook signature — that is
the whole point (stable interface as the framework grows: pipeline ``slot``/
``step``, MoE ``expert_counts``, FP8 ``scales``, warp-spec role, ...).

NOTE: this Triton build requires the aggregate ``__init__`` to be a plain Python
function — ``@triton.jit`` and ``@constexpr_function`` are both rejected by
``_aggregate`` (they are ``JITCallable``).
"""

import triton.language as tl
from triton.language.core import _aggregate as aggregate


@aggregate
class Ctx:
    """Metadata a ``produce``/``consume`` hook reads for the current tile.

    The tile payload (loaded register values) is passed to ``consume`` as an
    explicit ``regs`` arg, not carried here.
    """

    ptr: tl.tensor  # in_ptr for produce, out_ptr for consume
    idx: tl.tensor  # element offsets of the current tile
    mask: tl.tensor  # bounds mask
    # Future fields (peer, shard_idx, slot, step, scales, ...) are added here
    # WITHOUT changing any hook signature.

    def __init__(self, ptr, idx, mask):
        self.ptr = ptr
        self.idx = idx
        self.mask = mask
