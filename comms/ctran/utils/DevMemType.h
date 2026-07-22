// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>

#include <vector>

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
__attribute__((visibility("default"))) commResult_t
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
__attribute__((visibility("default"))) commResult_t
getCudaDevFromPtr(const void* addr, int& cudaDev);

/**
 * One physical VMM (cuMem) segment backing a virtual address range.
 *
 * `handle` is retained (cuMemRetainAllocationHandle) and OWNED BY THE CALLER --
 * each must be paired with a cuMemRelease. `base`/`size` are the segment's full
 * physical extent (from cuMemGetAddressRange), which may extend past the
 * queried range's end for the last segment.
 */
struct CuMemSegment {
  CUmemGenericAllocationHandle handle{};
  CUdeviceptr base{};
  size_t size{};
};

/**
 * Enumerates the physical VMM (cuMem) segments backing the virtual address
 * range [addr, addr + len). A range allocated by cuMem APIs may be stitched
 * together from several physical allocations; this walks them in address order.
 *
 * For each segment it retains the allocation handle
 * (cuMemRetainAllocationHandle) and records the segment's base + full physical
 * extent (cuMemGetAddressRange), then advances to the next segment boundary.
 *
 * The retained handles are OWNED BY THE CALLER: every returned
 * CuMemSegment::handle must be released with cuMemRelease. On error, any
 * handles retained before the failure are still returned in `segments` (so the
 * caller can release them); the caller should treat a non-success return as
 * fatal for the range and release whatever was collected.
 *
 * `addr` must be cuMem-allocated memory (see getDevMemType == kCumem);
 * behavior on other memory types is that of cuMemRetainAllocationHandle.
 *
 * @param addr Start of the range. Must not be nullptr.
 * @param len  Length of the range in bytes.
 * @param segments Output; appended to (not cleared). One entry per segment.
 * @return commSuccess on success
 *         commInvalidUsage if addr is nullptr
 *         a CUDA error result if a driver call fails
 */
__attribute__((visibility("default"))) commResult_t enumerateCuMemSegments(
    const void* addr,
    size_t len,
    std::vector<CuMemSegment>& segments);
