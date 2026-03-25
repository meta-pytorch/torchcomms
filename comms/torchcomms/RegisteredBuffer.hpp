// Copyright (c) Meta Platforms, Inc. and affiliates.
// RegisteredBuffer — Lightweight handle for local registered source buffers.
//
// This header is intentionally free of heavy dependencies (no ATen, no Torch)
// so that it can be included from device-side code compiled to LLVM bitcode
// (clang device-only mode) where ATen headers are unavailable.

#pragma once

#include <cstddef>
#include <cstdint>

namespace torch::comms {

// =============================================================================
// RegisteredBuffer — Handle for Local Registered Source Buffers
// =============================================================================
//
// Represents a registered local memory region for RMA put operations.
// Created on host via hostWindow.register_local_buffer().
//
// Used by both the host-side virtual interface (TorchCommWindow) and
// device-side kernel code (TorchCommDeviceWindow) without circular includes.

struct RegisteredBuffer {
  void* base_ptr{nullptr};
  size_t size{0};
  void* backend_window{
      nullptr}; // Backend-specific window handle (e.g., ncclWindow_t)
  // RDMA local key in network byte order for IBGDA puts (PipesDeviceBackend).
  // Zero for backends that do not use IBGDA (e.g., NCCLDeviceBackend).
  uint32_t lkey{0};
};

} // namespace torch::comms
