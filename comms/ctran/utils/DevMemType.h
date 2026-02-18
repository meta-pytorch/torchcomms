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
 * @return commSuccess on successful classification
 *         commInvalidUsage if addr is nullptr or cudaDev is negative
 *         commInternalError for unsupported cuMem handle types or other
 * internal errors
 */
commResult_t
getDevMemType(const void* addr, const int cudaDev, DevMemType& memType);

/**
 * Determines the CUDA device associated with a given memory pointer.
 *
 * For device or managed memory, returns the device from
 * cudaPointerGetAttributes. For host memory (pinned or unregistered), returns
 * the current CUDA device.
 *
 * @param addr The memory pointer to analyze. Must not be nullptr.
 * @param cudaDev Output parameter for the CUDA device ID.
 * @return commSuccess on successful determination
 *         commInvalidUsage if addr is nullptr
 */
commResult_t getCudaDevFromPtr(const void* addr, int& cudaDev);
