# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict
"""NCCLX backend for torchcomms.

Importing this module loads the ``_comms_ncclx`` extension; torchcomms
discovers the backend automatically via the ``torchcomms.backends``
entry-point group registered by this package's wheel metadata.
"""

import torchcomms  # noqa: F401  ensure libtorchcomms.so is loaded first

from torchcomms_ncclx import _comms_ncclx  # noqa: F401

__all__ = ["_comms_ncclx"]
