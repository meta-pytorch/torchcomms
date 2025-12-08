// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

// =============================================================================
// Atomic operations with system-wide visibility for NVLink cross-GPU signaling
// =============================================================================
//
// WHY .global QUALIFIER?
// ======================
// Without explicit .global, the compiler uses generic addressing which adds:
//   1. Runtime address space detection (global vs shared vs local)
//   2. Extra instructions for address translation
//   3. Potential predicated branches in generated SASS
//
// With explicit .global:
//   1. Compiler knows memory space at compile time
//   2. Direct addressing with no runtime checks
//   3. Simpler, faster instruction encoding (~2% throughput improvement)
//
// WHY .sys SCOPE?
// ===============
// The .sys (system) scope is required for cross-GPU NVLink communication:
//   - .cta  = visible only within thread block
//   - .gpu  = visible only within same GPU
//   - .sys  = visible across all GPUs + CPU
//
// For P2P NVLink, sender writes to memory that receiver reads via NVLink peer
// mapping. The .sys scope ensures the NVLink coherence protocol propagates
// writes across GPU boundaries.

__device__ __forceinline__ void store_int_release(int* addr, int value) {
  asm volatile("st.release.sys.global.s32 [%0], %1;"
               :
               : "l"(addr), "r"(value)
               : "memory");
}

__device__ __forceinline__ int load_int_acquire(int* addr) {
  int value;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
               : "=r"(value)
               : "l"(addr)
               : "memory");
  return value;
}

/**
 * ChunkState - 128-byte aligned synchronization primitive for P2P transfers
 *
 * Provides thread-safe state signaling between GPUs using acquire-release
 * semantics. Each ChunkState represents the synchronization state for one
 * chunk of data in the staging buffer.
 *
 * MEMORY LAYOUT:
 * - First 4 bytes: actual state value (int)
 * - Remaining 124 bytes: padding to prevent false sharing
 * - Total size: 128 bytes (cache line aligned)
 *
 * STATE VALUES:
 * - -1: Chunk is free/ready for sender to write
 * - stepId (0, 1, 2, ...): Chunk contains data from step stepId, ready for
 * receiver
 *
 * THREAD SAFETY:
 * All operations use acquire-release memory ordering to ensure:
 * - Sender's data writes are visible to receiver before state update
 * - Receiver's data reads complete before state reset
 */
template <typename T>
struct alignas(128) ChunkState {
  T value_;
  char padding_[128 - sizeof(T)]{};

  __host__ __device__ ChunkState(T v = static_cast<T>(-1)) : value_(v) {}

  /**
   * load - Read chunk state with acquire semantics
   *
   * Atomically loads the state value with acquire memory ordering.
   * Ensures all memory writes by the peer that happened before their
   * store() are visible to this GPU after load() returns.
   *
   * @return Current state value
   */
  __device__ __forceinline__ T load() const {
    static_assert(
        sizeof(T) == sizeof(int), "Only 32-bit types supported for atomics");
    return static_cast<T>(
        load_int_acquire(reinterpret_cast<int*>(const_cast<T*>(&value_))));
  }

  /**
   * store - Write chunk state with release semantics
   *
   * Atomically stores the state value with release memory ordering.
   * Ensures all memory writes by this GPU before store() are visible
   * to the peer after they read the new state with load().
   *
   * @param v New state value to write
   */
  __device__ __forceinline__ void store(T v) {
    static_assert(
        sizeof(T) == sizeof(int), "Only 32-bit types supported for atomics");
    store_int_release(reinterpret_cast<int*>(&value_), static_cast<int>(v));
  }
};

static_assert(
    sizeof(ChunkState<int>) == 128,
    "ChunkState<int> must be exactly 128 bytes");
static_assert(
    alignof(ChunkState<int>) == 128,
    "ChunkState<int> must be 128-byte aligned");

/**
 * LocalState - Pointers to local GPU's data and state buffers
 *
 * Holds device pointers to this GPU's own buffers used for P2P transfers.
 * - dataBuffer: Staging buffer for data transfer (size: dataBufferSize)
 * - stateBuffer: Synchronization states (size: numChunks * 128 bytes)
 */
struct LocalState {
  char* dataBuffer;
  ChunkState<int>* stateBuffer;
};

/**
 * RemoteState - Pointers to peer GPU's data and state buffers
 *
 * Holds IPC device pointers to the peer GPU's buffers for P2P access.
 * These pointers are obtained via NVLink IPC and allow direct peer memory
 * access.
 * - dataBuffer: Peer's staging buffer (accessed via NVLink)
 * - stateBuffer: Peer's synchronization states (accessed via NVLink)
 */
struct RemoteState {
  char* dataBuffer;
  ChunkState<int>* stateBuffer;
};

/**
 * P2pNvlTransportOptions - Configuration for P2P NVLink transport
 *
 * Defines the buffer sizes and chunking parameters for staged transfers.
 * - dataBufferSize: Size of staging buffer (determines max per-step transfer)
 * - chunkSize: Size of each chunk for parallel processing (must divide evenly)
 */
struct P2pNvlTransportOptions {
  std::size_t dataBufferSize;
  std::size_t chunkSize;
};

// Device-side P2P NVLink transport handle providing point-to-point operations
// like send/recv between ranks over NVLink
class P2pNvlTransportDevice {
 public:
  __host__ __device__ P2pNvlTransportDevice(
      int myRank,
      int peerRank,
      const P2pNvlTransportOptions& options,
      const LocalState& localState,
      const RemoteState& remoteState)
      : myRank_(myRank),
        peerRank_(peerRank),
        options_(options),
        localState_(localState),
        remoteState_(remoteState) {}

  /**
   * send - Transfer data to peer GPU over NVLink
   *
   * Sends 'nbytes' bytes from srcbuff to the peer GPU using staged transfer
   * with fine-grained chunk-level synchronization. All threads in the group
   * cooperate to transfer the data in parallel.
   *
   * ALGORITHM:
   * 1. Divide transfer into STEPS (dataBufferSize each)
   * 2. Divide each step into CHUNKS for parallel processing
   * 3. For each chunk:
   *    - WAIT: Spin until receiver signals buffer is free (state == -1)
   *    - COPY: Copy chunk to staging buffer using vectorized operations
   *    - SIGNAL: Set state = stepId to notify receiver data is ready
   *
   * @param group ThreadGroup for cooperative processing (all threads
   * participate)
   * @param srcbuff Source data pointer (device memory)
   * @param nbytes Number of bytes to send
   *
   * EXAMPLE:
   * ========
   * Transfer 1GB of data with 256MB staging buffer and 512KB chunks:
   *
   * Configuration:
   *   - Total data: 1GB
   *   - Staging buffer size: 256MB
   *   - Chunk size: 512KB
   *   - Launch: 8 blocks × 256 threads = 2048 threads (64 warps)
   *
   * Execution breakdown:
   *   - Total steps: ceil(1GB / 256MB) = 4 steps
   *   - Chunks per step: ceil(256MB / 512KB) = 512 chunks
   *   - Each warp processes: 512 chunks / 64 warps = 8 chunks (contiguous)
   *
   * Timeline for one step (256MB):
   *   Step 0:
   *     - Warp 0 processes chunks [0..7]     (0MB - 4MB)
   *     - Warp 1 processes chunks [8..15]    (4MB - 8MB)
   *     - ...
   *     - Warp 63 processes chunks [504..511] (252MB - 256MB)
   *   Step 1: Next 256MB (same chunk distribution)
   *   Step 2: Next 256MB
   *   Step 3: Last 256MB
   */
  __device__ __forceinline__ void
  send(ThreadGroup& group, void* srcbuff, std::size_t nbytes) {
#ifdef __CUDA_ARCH__
    char* src = reinterpret_cast<char*>(srcbuff);

    // REMOTE-WRITE PATTERN:
    // Sender writes data directly to RECEIVER's local buffer via NVLink.
    // Benefits: Receiver reads from local memory (faster read, no NVLink hop)
    // Trade-off: Sender's copy goes over NVLink
    char* sendBuffer = remoteState_.dataBuffer;
    ChunkState<int>* sendStates = remoteState_.stateBuffer;

    const std::size_t totalSteps =
        (nbytes + options_.dataBufferSize - 1) / options_.dataBufferSize;
    const std::size_t kChunkSize = options_.chunkSize;
    const std::size_t chunksPerStep =
        (options_.dataBufferSize + kChunkSize - 1) / kChunkSize;

    for (std::size_t stepId = 0; stepId < totalSteps; ++stepId) {
      const std::size_t stepOffset = stepId * options_.dataBufferSize;
      const std::size_t stepBytes =
          (stepOffset + options_.dataBufferSize <= nbytes)
          ? options_.dataBufferSize
          : nbytes - stepOffset;
      const std::size_t numChunksThisStep =
          (stepBytes + kChunkSize - 1) / kChunkSize;

      group.for_each_item_contiguous(numChunksThisStep, [&](uint32_t chunkIdx) {
        const std::size_t chunkOffset = chunkIdx * kChunkSize;
        const std::size_t chunkBytes = (chunkOffset + kChunkSize <= stepBytes)
            ? kChunkSize
            : stepBytes - chunkOffset;

        ChunkState<int>& chunkState = sendStates[chunkIdx];

        while (chunkState.load() != -1) {
        }

        copy_chunk_vectorized<uint4>(
            sendBuffer,
            src,
            chunkBytes,
            chunkOffset,
            stepOffset + chunkOffset,
            group);

        group.sync();

        if (group.is_leader()) {
          chunkState.store(static_cast<int>(stepId));
        }
      });
    }
#endif
  }

  /**
   * recv - Receive data from peer GPU over NVLink
   *
   * similar to send(), but for receiving data from the peer GPU
   */
  __device__ __forceinline__ void
  recv(ThreadGroup& group, void* dstbuff, std::size_t nbytes) {
#ifdef __CUDA_ARCH__
    char* dst = reinterpret_cast<char*>(dstbuff);

    // REMOTE-WRITE PATTERN:
    // Receiver reads from LOCAL buffer (sender wrote here via NVLink).
    // Benefits: Local memory read is faster than reading over NVLink
    char* recvBuffer = localState_.dataBuffer;
    ChunkState<int>* recvStates = localState_.stateBuffer;

    const std::size_t totalSteps =
        (nbytes + options_.dataBufferSize - 1) / options_.dataBufferSize;
    const std::size_t kChunkSize = options_.chunkSize;
    const std::size_t chunksPerStep =
        (options_.dataBufferSize + kChunkSize - 1) / kChunkSize;

    for (std::size_t stepId = 0; stepId < totalSteps; stepId++) {
      const std::size_t stepOffset = stepId * options_.dataBufferSize;
      const std::size_t stepBytes =
          (stepOffset + options_.dataBufferSize <= nbytes)
          ? options_.dataBufferSize
          : nbytes - stepOffset;
      const std::size_t numChunksThisStep =
          (stepBytes + kChunkSize - 1) / kChunkSize;

      group.for_each_item_contiguous(numChunksThisStep, [&](uint32_t chunkIdx) {
        const std::size_t chunkOffset = chunkIdx * kChunkSize;
        const std::size_t chunkBytes = (chunkOffset + kChunkSize <= stepBytes)
            ? kChunkSize
            : stepBytes - chunkOffset;

        ChunkState<int>& chunkState = recvStates[chunkIdx];

        while (chunkState.load() != static_cast<int>(stepId)) {
        }

        copy_chunk_vectorized<uint4>(
            dst,
            recvBuffer,
            chunkBytes,
            stepOffset + chunkOffset,
            chunkOffset,
            group);

        group.sync();

        if (group.is_leader()) {
          chunkState.store(-1);
        }
      });
    }
#endif
  }

 private:
  const int myRank_{-1};
  const int peerRank_{-1};
  const P2pNvlTransportOptions options_;
  LocalState localState_;
  RemoteState remoteState_;

#ifdef P2pNvlTransport_TEST_FRIENDS
  P2pNvlTransport_TEST_FRIENDS
#endif
};

} // namespace comms::pipes
