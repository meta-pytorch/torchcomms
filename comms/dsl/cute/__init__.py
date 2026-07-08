# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""CuTe backend: send/recv interface stubs (reserved for the CuTe stack)."""

from .send_recv import recv, send

__all__ = ["send", "recv"]
