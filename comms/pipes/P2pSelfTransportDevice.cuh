// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

/**
 * SelfTransportDevice - Local memory copy transport
 * ==================================================
 *
 * A simple transport implementation for local memory copies within the same
 * GPU. This is useful for self-copy operations where source and destination are
 * on the same device.
 *
 * IMPLEMENTATION:
 * ===============
 * - write(): Implemented using copy_chunk_vectorized with zero offsets
 * - send(): Not implemented (pure virtual from base)
 * - recv(): Not implemented (pure virtual from base)
 *
 * USAGE:
 * ======
 * This class is primarily used for local copies where P2P communication is
 * not needed (e.g., copying within the same GPU).
 *
 * Example:
 *   SelfTransportDevice transport;
 *   transport.write(dst_d, src_d, nbytes);
 */
class P2pSelfTransportDevice {
 public:
  __host__ __device__ P2pSelfTransportDevice() = default;
  __host__ __device__ ~P2pSelfTransportDevice() = default;

  /**
   * send - Not implemented for SelfTransportDevice
   *
   * Self transport is for local copies only, not for sending to peers.
   * Calling this method will trap and abort the kernel.
   */
  __device__ void send(ThreadGroup& group, void* srcbuff, std::size_t nbytes) {
#ifdef __CUDA_ARCH__
    __trap(); // Abort kernel if send is called on SelfTransportDevice
#endif
  }

  /**
   * recv - Not implemented for SelfTransportDevice
   *
   * Self transport is for local copies only, not for receiving from peers.
   * Calling this method will trap and abort the kernel.
   */
  __device__ void recv(ThreadGroup& group, void* dstbuff, std::size_t nbytes) {
#ifdef __CUDA_ARCH__
    __trap(); // Abort kernel if recv is called on SelfTransportDevice
#endif
  }

  /**
   * write - Direct local memory copy using vectorized operations
   *
   * Performs a high-performance vectorized copy from src_d to dst_d using
   * copy_chunk_vectorized. The work is distributed across ALL thread groups
   * using for_each_item_contiguous, so each group processes only its portion
   * of the data.
   *
   * The chunk size is computed dynamically as (nbytes / total_groups) to
   * ensure good parallelism, with a minimum of 16 bytes per chunk for
   * vectorized access efficiency.
   *
   * NOTE: only support no overlap copy for now
   *
   * @param group ThreadGroup for cooperative processing
   * @param dst_d Destination pointer (device memory)
   * @param src_d Source pointer (device memory)
   * @param nbytes Number of bytes to write
   */
  __device__ __forceinline__ void write(
      ThreadGroup& group,
      char* dst_d,
      const char* src_d,
      std::size_t nbytes) {
#ifdef __CUDA_ARCH__
    // Early return for no-op cases (check before overlap to handle dst == src)
    if (nbytes == 0 || dst_d == src_d) {
      return;
    }

    // Check for buffer overlap - only support non-overlapping buffers
    if (!(src_d + nbytes <= dst_d || dst_d + nbytes <= src_d)) {
      __trap(); // Abort kernel if buffers overlap
    }

    // Compute chunk size: aim for nbytes / total_groups per chunk,
    // aligned to 16 bytes (uint4 size) for efficient vectorized access
    constexpr std::size_t kAlignment = 16;
    const std::size_t targetChunkSize = nbytes / group.total_groups;
    // Round up to nearest 16-byte boundary, minimum 16 bytes
    const std::size_t chunkSize =
        ((targetChunkSize + kAlignment - 1) / kAlignment) * kAlignment;
    // Ensure minimum chunk size
    const std::size_t alignedChunkSize = chunkSize > 0 ? chunkSize : kAlignment;

    const std::size_t numChunks =
        (nbytes + alignedChunkSize - 1) / alignedChunkSize;

    // Distribute chunks across all groups using for_each_item_contiguous
    // Each group processes its assigned contiguous range of chunks
    group.for_each_item_contiguous(numChunks, [&](uint32_t chunkIdx) {
      const std::size_t chunkOffset = chunkIdx * alignedChunkSize;
      const std::size_t chunkBytes = (chunkOffset + alignedChunkSize <= nbytes)
          ? alignedChunkSize
          : nbytes - chunkOffset;

      if (chunkBytes > 0) {
        copy_chunk_vectorized<uint4>(
            dst_d + chunkOffset, // dst_base
            src_d + chunkOffset, // src_base
            chunkBytes, // chunk_bytes
            0, // dst_offset
            0, // src_offset
            group);
      }
    });
#endif
  }
};

} // namespace comms::pipes
