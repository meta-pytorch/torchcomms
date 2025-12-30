# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Pickle support functions for TorchComm objects.

These wrapper functions are needed because C++ functions registered via pybind11
become PyCapsule objects that can't be pickled directly. By wrapping them in
Python functions, we make them pickleable.
"""

from torchcomms import _comms


def _unpickle_comm(backend, device, name, options):
    """Unpickle a TorchComm object, maintaining Python object identity."""
    return _comms._unpickle_comm(backend, device, name, options)


def _unpickle_window(window_id, comm_name, buf_shape, buf_dtype, buf_device):
    """Unpickle a TorchCommWindow object, maintaining Python object identity."""
    return _comms._unpickle_window(
        window_id, comm_name, buf_shape, buf_dtype, buf_device
    )


def _unpickle_batch_sendrecv(batch_id, comm_name):
    """Unpickle a BatchSendRecv object, maintaining Python object identity."""
    return _comms._unpickle_batch_sendrecv(batch_id, comm_name)
