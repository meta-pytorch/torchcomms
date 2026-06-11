# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""CuTe realization of the hook-contract ``Ctx`` (the per-tile view a hook reads).

The CuTe twin of ``triton/ctx.py``. Adding a field here never changes a hook
signature — same intent as the Triton ``@_aggregate`` Ctx.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Ctx:
    """One tile's view for a hook.

    ``part`` is this tile's partitioned tensor (input partition for ``produce``,
    output partition for ``consume``); ``atom`` is the gmem<->rmem copy atom.
    """

    part: Any
    atom: Any
