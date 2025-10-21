// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>

#include "comms/utils/commSpecs.h"

enum DevMemType {
  kCudaMalloc = 0,
  kManaged = 1,
  kHostPinned = 2,
  kHostUnregistered = 3,
  kCumem = 4,
};

inline const char* devMemTypeStr(DevMemType memType) {
  switch (memType) {
    case kCudaMalloc:
      return "cudaMalloc";
    case kManaged:
      return "managed";
    case kHostPinned:
      return "hostPinned";
    case kHostUnregistered:
      return "hostUnregistered";
    case kCumem:
      return "cuMem";
    default:
      return "unknown";
  }
}

/**
 * Determines the memory type of a given memory address on a specific CUDA
 * device.
 *
 * This function analyzes a memory pointer to classify it into one of the
 * supported DevMemType categories (kCumem, kCudaMalloc, kHost, kManaged,
 * kUnregistered).
 *
 * @param addr The memory pointer to analyze. Must not be nullptr.
 * @param cudaDev The CUDA device associated with the memory. Must be
 * non-negative.
 * @param memType [out] Reference to store the determined memory type.
 *
 * @return commSuccess on successful memory type determination
 *         commInvalidUsage if addr is nullptr or cudaDev is negative
 *         commInternalError for unsupported cuMem handle types or other
 * internal errors
 */
commResult_t
getDevMemType(const void* addr, const int cudaDev, DevMemType& memType);
