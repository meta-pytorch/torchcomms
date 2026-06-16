# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""CuTe realization of the hook-contract ``Ctx`` (the per-tile view a hook reads).

The CuTe twin of ``triton/ctx.py``. Adding a field here never changes a hook
signature — same intent as the Triton ``@_aggregate`` Ctx.
"""

from __future__ import annotations

from typing import Any


class Ctx:
    """One tile's view for a hook.

    ``part`` is this tile's partitioned tensor (input partition for ``produce``,
    output partition for ``consume``); ``atom`` is the gmem<->rmem copy atom.
    """

    __slots__ = ("part", "atom")

    def __init__(self, part: Any, atom: Any) -> None:
        self.part = part
        self.atom = atom
