// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu

namespace torch::comms {

// Create a root TCPStore from MASTER_ADDR / MASTER_PORT env vars.
// Rank 0 runs the server.
c10::intrusive_ptr<c10d::Store> createTCPStore(
    std::chrono::milliseconds timeout);

// Create an independent TCPStore using `bootstrapStore` only to
// exchange connection info.  Rank 0 binds a new store on an
// OS-assigned port and broadcasts the port through the bootstrap
// store.  The caller can release the bootstrap store immediately
// after this call returns.
c10::intrusive_ptr<c10d::Store> dupTCPStore(
    const c10::intrusive_ptr<c10d::Store>& store,
    std::chrono::milliseconds timeout);

} // namespace torch::comms
