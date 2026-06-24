// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>

namespace torch::comms {

extern "C" {
// Callback invoked to register (reg) or deregister (dereg) a memory segment
// with the tensor-transfer RegCache. Returns 0 on success, non-zero on failure.
using RdmaRegFn = int (*)(void* addr, size_t len);
}

// Installs a CUDA caching-allocator trace hook that forwards segment
// alloc/free events to the supplied reg/dereg callbacks. The callbacks register
// the segment with the tensor-transfer's RegCache instance. Idempotent — only
// the first call installs the hook.
//
// The hook state (callbacks + once-flag) is a per-translation-unit global, so
// each shared object linking this file owns an independent copy.
void attachRdmaMemoryHook(RdmaRegFn reg, RdmaRegFn dereg);

// Test-only: disables the callbacks installed by attachRdmaMemoryHook by
// clearing the stored reg/dereg pointers. The CUDA caching-allocator trace
// tracker has no detach API, so the tracker stays attached but its callback
// becomes a no-op. Used to isolate gtest cases that share a process.
void detachRdmaMemoryHook();

} // namespace torch::comms
