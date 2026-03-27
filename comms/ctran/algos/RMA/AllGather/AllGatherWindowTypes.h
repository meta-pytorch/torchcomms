// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/window/CtranWin.h"

namespace ctran::allgatherwindow {

// Arguments passed to GPE function (heap-allocated, ownership transferred to
// GPE function which frees it on completion)
struct AllGatherWindowGpeArgs {
  CtranWin* win;
  size_t count;
  commDataType_t datatype;
};

// Helper function to get pointer with offset
inline void* getPtr(void* base, size_t offset) {
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + offset);
}

} // namespace ctran::allgatherwindow
