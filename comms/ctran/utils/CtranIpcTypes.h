// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "comms/ctran/utils/DevMemTypeDefs.h"

namespace ctran::utils {

constexpr int kCtranIpcHandleSize = 64;

union CtranIpcHandle {
  uint64_t fd; // hold a CUmemGenericAllocationHandle for UDS fd support
  // Dummy handle to make sure the size is the same as CUDA IPC handles without
  // pulling CUDA runtime headers into control-message type definitions.
  struct {
    unsigned char data[kCtranIpcHandleSize];
  } handle;
  struct {
    unsigned char data[kCtranIpcHandleSize];
  } cudaIpcHandle;
};

struct CtranIpcSegDesc {
  CtranIpcHandle sharedHandle{};
  size_t range{0};
};

static inline void printHandle(
    std::stringstream& ss,
    const DevMemType memType,
    const int cuMemHandleType,
    const CtranIpcHandle& handle) {
  bool isCumemFabric = false;
  isCumemFabric = cuMemHandleType & 0x8;
  if (memType == DevMemType::kCumem) {
    if (isCumemFabric) {
      ss << "fabric handle: ";
      for (int j = 0; j < kCtranIpcHandleSize; ++j) {
        ss << std::hex << static_cast<int>(handle.handle.data[j]);
      }
    } else {
      ss << "posix fd handle 0x" << std::hex << handle.fd;
    }
  } else if (memType == DevMemType::kCudaMalloc) {
    ss << "ipc handle: ";
    for (int j = 0; j < kCtranIpcHandleSize; ++j) {
      ss << "0x" << std::hex << static_cast<int>(handle.handle.data[j]);
    }
  } else {
    ss << "unsupported device memory type " << devMemTypeStr(memType);
  }
}

// maxNumHandles is needed so that CtranIpcDesc objects are constant
// size for communication. It is set to 2 to reduce ControlMsg size, since we
// temporarily disable disjoint segments support for NVL peers.
// FIXME: this is a temporary hack; we should support arbitrary size via dynamic
// allocated CtranIpcSegDesc list.
#define CTRAN_IPC_INLINE_SEGMENTS 2

struct CtranIpcDesc {
  // Total number of segments including any extra segments beyond inline.
  int totalSegments{0};
  void* base{nullptr};
  size_t range{0};
  // pass local pid for peer to import sharedHandle as file descriptor
  int pid{0};
  DevMemType memType{DevMemType::kHostUnregistered};
  int cuMemHandleType{0};

  // Preallocated payload for multi-segment IPC memory.
  CtranIpcSegDesc segments[CTRAN_IPC_INLINE_SEGMENTS] = {};

  // Number of valid entries in the inline segments[] array.
  int numInlineSegments() const {
    return std::min(totalSegments, CTRAN_IPC_INLINE_SEGMENTS);
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "base: " << base << ", range: " << std::dec << range
       << ", pid: " << pid;
    ss << ", segments: [";
    for (int i = 0; i < CTRAN_IPC_INLINE_SEGMENTS; ++i) {
      if (i != 0) {
        ss << ", ";
      }
      printHandle(ss, memType, cuMemHandleType, segments[i].sharedHandle);
      ss << " range: " << std::dec << segments[i].range;
    }
    ss << "]";
    return ss.str();
  }
};

} // namespace ctran::utils
