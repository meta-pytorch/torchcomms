# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe
"""Small dtype / op-name helpers shared by backend.py and the Triton kernels."""

from typing import Any

import torch

from torchcomms._comms import RedOpType


def nbytes_of(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def cast_buffer(buf: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """View a uint8 symm-mem ``buf`` as the dtype of ``like``.

    Returns a 1-D contiguous tensor; callers slice it to the element count
    they need.
    """
    if buf.dtype != torch.uint8:
        raise RuntimeError("cast_buffer expects a uint8 buffer")
    if like.element_size() == 1:
        return buf
    # Truncate to a multiple of like.element_size() so view() succeeds.
    capacity = buf.numel() - (buf.numel() % like.element_size())
    return buf[:capacity].view(like.dtype)


def reduce_op_name(op: Any) -> str:
    """Return the canonical lowercase name of a ReduceOp."""
    t = op.type
    if t == RedOpType.SUM:
        return "sum"
    if t == RedOpType.PRODUCT:
        return "prod"
    if t == RedOpType.MIN:
        return "min"
    if t == RedOpType.MAX:
        return "max"
    if t == RedOpType.AVG:
        return "avg"
    if t == RedOpType.PREMUL_SUM:
        return "premul_sum"
    if t == RedOpType.BAND:
        return "band"
    if t == RedOpType.BOR:
        return "bor"
    if t == RedOpType.BXOR:
        return "bxor"
    raise RuntimeError(f"symmem: unsupported reduce op {t}")
