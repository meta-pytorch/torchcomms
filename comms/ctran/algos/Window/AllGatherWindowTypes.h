// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/Window/AllGatherWindowDevTypes.h"
#include "comms/ctran/window/CtranWin.h"

namespace ctran::allgatherwindow {

// Algorithm types for AllGatherWindow
enum class Algorithm {
  kDirect, // All-to-all PUT to all peers
  kPipeline, // Ring-based pipeline for better scaling
};

// Resource struct for pipeline algorithm (GPE-Kernel synchronization)
struct Resource {
  ctran::algos::GpeKernelSync* pipeSync{nullptr};

  Resource() = default;
  ~Resource() = default;
};

// Arguments passed to GPE function
struct AllGatherWindowGpeArgs {
  CtranWin* win;
  const void* sendbuff;
  size_t count;
  commDataType_t datatype;
  Resource* resource;
};

// Helper function to get pointer with offset
inline void* getPtr(void* base, size_t offset) {
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + offset);
}

inline const void* getPtr(const void* base, size_t offset) {
  return reinterpret_cast<const void*>(
      reinterpret_cast<uintptr_t>(base) + offset);
}

} // namespace ctran::allgatherwindow
