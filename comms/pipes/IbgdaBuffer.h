// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

// Allow compilation in both host (C++) and device (CUDA) contexts
#ifdef __CUDACC__
#define IBGDA_HOST_DEVICE __host__ __device__
#else
#define IBGDA_HOST_DEVICE
#endif

namespace comms::pipes {

/**
 * IbgdaLocalBuffer - Local buffer descriptor for RDMA operations
 *
 * Represents a buffer in the local GPU's memory that can be used
 * as a source for RDMA writes or destination for RDMA reads.
 * Uses lkey (local key) for memory registration.
 *
 * This struct is usable from both host and device code.
 */
struct IbgdaLocalBuffer {
  void* ptr{nullptr};
  uint32_t lkey{0};

  IBGDA_HOST_DEVICE IbgdaLocalBuffer() = default;

  IBGDA_HOST_DEVICE IbgdaLocalBuffer(void* p, uint32_t key)
      : ptr(p), lkey(key) {}

  /**
   * Create a sub-buffer at the given byte offset
   */
  IBGDA_HOST_DEVICE IbgdaLocalBuffer subBuffer(std::size_t offset) const {
    return IbgdaLocalBuffer(static_cast<char*>(ptr) + offset, lkey);
  }
};

/**
 * IbgdaRemoteBuffer - Remote buffer descriptor for RDMA operations
 *
 * Represents a buffer in a remote GPU's memory that can be accessed
 * via RDMA operations. Uses rkey (remote key) for memory registration.
 *
 * This struct is usable from both host and device code.
 */
struct IbgdaRemoteBuffer {
  void* ptr{nullptr};
  uint32_t rkey{0};

  IBGDA_HOST_DEVICE IbgdaRemoteBuffer() = default;

  IBGDA_HOST_DEVICE IbgdaRemoteBuffer(void* p, uint32_t key)
      : ptr(p), rkey(key) {}

  /**
   * Create a sub-buffer at the given byte offset
   */
  IBGDA_HOST_DEVICE IbgdaRemoteBuffer subBuffer(std::size_t offset) const {
    return IbgdaRemoteBuffer(static_cast<char*>(ptr) + offset, rkey);
  }
};

} // namespace comms::pipes
