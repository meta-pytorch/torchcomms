// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

namespace torch::comms {

// Installs a CUDA caching-allocator trace hook that registers and deregisters
// allocator segments with ctran::RegCache for the RDMA transport. Registration
// is lazy (forceReg=false): segments are cached and the IB handle is
// materialized on the first searchIbRegHandle() lookup. Idempotent — only the
// first call installs the hook.
void attachRdmaMemoryHook();

} // namespace torch::comms
