// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>
#include "comms/pipes/BarrierState.cuh"
#include "comms/pipes/ChunkState.cuh"
#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/SignalState.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes {

/**
 * LocalState - Pointers to local GPU's buffers
 *
 * With REMOTE-WRITE pattern:
 * - Sender writes to RemoteState (peer's local buffers via NVLink)
 * - Receiver reads from LocalState (own local buffers)
 *
 * This means LocalState buffers are the DESTINATION for incoming data.
 */
struct LocalState {
  char* dataBuffer;
  DeviceSpan<ChunkState> stateBuffer;
  DeviceSpan<SignalState> signalBuffer;
  DeviceSpan<BarrierState> barrierBuffer;
};

/**
 * RemoteState - Pointers to peer GPU's buffers (via NVLink peer mapping)
 *
 * With REMOTE-WRITE pattern:
 * - Sender writes directly to these buffers (peer's local memory)
 * - This allows receiver to read from local memory (faster)
 *
 * These pointers are obtained via IPC and point to peer's LocalState buffers.
 */
struct RemoteState {
  char* dataBuffer;
  DeviceSpan<ChunkState> stateBuffer;
  DeviceSpan<SignalState> signalBuffer;
  DeviceSpan<BarrierState> barrierBuffer;
};

/**
 * P2pNvlTransportOptions - Configuration for P2P NVLink transport
 *
 * Defines the buffer sizes and chunking parameters for staged transfers.
 * - dataBufferSize: Size of ONE pipeline slot (determines max per-step
 * transfer)
 * - chunkSize: Size of each chunk for parallel processing
 * - pipelineDepth: Number of buffer slots for pipelining (typically 2-8)
 *
 * Total memory allocated = pipelineDepth × dataBufferSize
 */
struct P2pNvlTransportOptions {
  std::size_t dataBufferSize;
  std::size_t chunkSize;
  std::size_t pipelineDepth;
};

/**
 * P2pNvlTransportDevice - High-Performance GPU-to-GPU Data Transfer over NVLink
 * ==============================================================================
 *
 * Provides pipelined, chunked data transfer between GPUs using NVLink with
 * fine-grained synchronization and remote-write optimization.
 *
 * REMOTE-WRITE ARCHITECTURE
 * =========================
 *
 * Key insight: Sender writes directly to RECEIVER's local memory via NVLink.
 * This allows the receiver to read from local memory (fast) rather than
 * reading over NVLink (slower).
 *
 *   GPU A (Sender)                              GPU B (Receiver)
 *   ┌─────────────────┐                         ┌─────────────────┐
 *   │  User Source    │                         │  User Dest      │
 *   │  Buffer         │                         │  Buffer         │
 *   └────────┬────────┘                         └────────▲────────┘
 *            │                                           │
 *            │                                           │
 *            │            ┌───────────────────┐          │
 *            │            │  Staging Buffer   │          │
 *            └──────────▶ │  (on GPU B)       │ ─────────┘
 *              NVLink     │  + State Buffer   │   local copy
 *              write      └───────────────────┘
 *
 * The staging buffer lives on GPU B (receiver's local memory).
 * GPU A writes to it via NVLink using IPC pointers.
 * GPU B reads from it locally (fast).
 *
 * DATA FLOW (per chunk):
 *   1. Sender waits for state == -1 (polls via NVLink)
 *   2. Sender copies data: src → staging buffer (NVLink write)
 *   3. Sender signals: state = stepId (NVLink write)
 *   4. Receiver waits for state == stepId (local poll, fast)
 *   5. Receiver copies data: staging buffer → dst (local read, fast)
 *   6. Receiver signals: state = -1 (local write, fast)
 *
 * MEMORY LAYOUT (pipelineDepth=4, chunksPerStep=2)
 * ================================================
 *
 * Each GPU allocates its own LocalState buffers. The peer gets IPC pointers
 * to these buffers (stored as RemoteState on the peer).
 *
 * Data Buffer (size = pipelineDepth × dataBufferSize):
 *┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
 *│   Stage 0        │   Stage 1        │   Stage 2        │   Stage 3        │
 *│  (dataBufferSize)│  (dataBufferSize)│  (dataBufferSize)│  (dataBufferSize)│
 *│ step 0,4,8,12... │ step 1,5,9,13... │ step 2,6,10,14...│ step 3,7,11,15...│
 *└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
 *
 * State Buffer (size = pipelineDepth × chunksPerStep × 128 bytes):
 * ┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
 * │  Stage 0 states  │  Stage 1 states  │  Stage 2 states  │  Stage 3 states  │
 * │ [chunk0][chunk1] │ [chunk0][chunk1] │ [chunk0][chunk1] │ [chunk0][chunk1] │
 * └──────────────────┴──────────────────┴──────────────────┴──────────────────┘
 *   Each [chunkN] is a 128-byte aligned ChunkState for cache line isolation.
 *
 * PIPELINING: STEP-LEVEL VIEW
 * ===========================
 *
 * With pipelineDepth=4, sender can be up to 3 steps ahead of receiver:
 *
 *   Time │ Sender (GPU A)         │ Receiver (GPU B)        │ Stage
 *   ─────┼────────────────────────┼─────────────────────────┼──────────
 *     0  │ write step 0 → B       │                         │ stage[0]
 *     1  │ write step 1 → B       │ read step 0 from local  │ stage[1]
 *     2  │ write step 2 → B       │ read step 1 from local  │ stage[2]
 *     3  │ write step 3 → B       │ read step 2 from local  │ stage[3]
 *     4  │ wait for stage[0] free │ read step 3 from local  │ (blocked)
 *     4' │ write step 4 → B       │ (freed stage[0])        │ stage[0] reused
 *     5  │ write step 5 → B       │ read step 4 from local  │ stage[1]
 *
 * PIPELINING: CHUNK-LEVEL VIEW (Fine-Grained Parallelism)
 * ========================================================
 *
 * Within each step, chunks are processed independently by different warps.
 * Each warp owns a contiguous range of chunks and makes independent progress.
 * This enables fine-grained pipelining where fast warps don't wait for slow
 * ones.
 *
 * Example: Step 0 with 8 chunks distributed across 4 warps:
 *
 *   Sender GPU A                              Receiver GPU B
 *   (4 warps, 2 chunks each)                  (4 warps, 2 chunks each)
 *
 *   Warp 0: chunks [0,1]                      Warp 0: chunks [0,1]
 *   Warp 1: chunks [2,3]                      Warp 1: chunks [2,3]
 *   Warp 2: chunks [4,5]                      Warp 2: chunks [4,5]
 *   Warp 3: chunks [6,7]                      Warp 3: chunks [6,7]
 *
 *   Time │ Sender Warps                │ Receiver Warps
 *   ─────┼─────────────────────────────┼─────────────────────────────
 *     0  │ W0: send c0, W1: send c2    │
 *        │ W2: send c4, W3: send c6    │
 *     1  │ W0: send c1, W1: send c3    │ W0: recv c0 (c0 ready)
 *        │ W2: send c5, W3: send c7    │ W2: recv c4 (c4 ready)
 *     2  │ W0: done step 0             │ W0: recv c1, W1: recv c2
 *        │ W1: done step 0             │ W2: recv c5, W3: recv c6
 *     3  │ W0: start step 1 (stage[1]) │ W0: done step 0
 *        │                             │ W1: recv c3, W3: recv c7
 *     4  │ W0: send step1 c0           │ W1,W2,W3: done step 0
 *        │ ...                         │ W0: start step 1
 *
 * Key observations:
 *   - Each chunk has independent state → no warp-to-warp synchronization
 *   - Fast warps can start next step while slow warps finish current step
 *   - Receiver warp can process a chunk as soon as sender warp signals it
 *   - Contiguous chunk assignment → good cache locality per warp
 *
 * STATE MACHINE (per chunk)
 * =========================
 *
 * State lives in RECEIVER's local memory. Both GPUs access it:
 * - Sender accesses via NVLink (remote)
 * - Receiver accesses locally (fast)
 *
 *        ┌───────────────┐
 * init → │ READY_TO_SEND │
 *        │     (-1)      │
 *        └───────┬───────┘
 *                │
 *                │ send() waits for READY_TO_SEND, copies data,
 *                │ signals ready_to_recv(stepId)
 *                ▼
 *        ┌───────────────┐
 *    ┌─▶ │ READY_TO_RECV │
 *    │   │   (stepId)    │
 *    │   └───────┬───────┘
 *    │           │
 *    │           │ recv() waits for READY_TO_RECV, copies data,
 *    │           │ signals ready_to_send()
 *    │           ▼
 *    │   ┌───────────────┐
 *    │   │ READY_TO_SEND │
 *    │   │     (-1)      │
 *    │   └───────┬───────┘
 *    │           │
 *    │           │ send() waits for READY_TO_SEND, copies data,
 *    │           │ signals ready_to_recv(stepId)
 *    │           │
 *    └───────────┘
 *
 * CHUNK DISTRIBUTION
 * ==================
 *
 * Chunks are distributed contiguously across thread groups for cache coherence:
 *
 *   512 chunks, 64 warps → 8 chunks per warp (contiguous)
 *
 *   Warp 0:  chunks [0..7]      ← contiguous memory access
 *   Warp 1:  chunks [8..15]
 *   Warp 2:  chunks [16..23]
 *   ...
 *   Warp 63: chunks [504..511]
 *
 * USAGE EXAMPLE
 * =============
 *
 *   // Host setup (once)
 *   P2pNvlTransport transport(myRank, nRanks, mpiBootstrap, config);
 *   transport.exchange();  // Exchange IPC handles
 *   auto device = transport.getP2pTransportDevice(peerRank);
 *
 *   // Kernel (sender on GPU A)
 *   __global__ void sendKernel(P2pNvlTransportDevice p2p, void* src, size_t n)
 * { auto group = make_warp_group(); p2p.send(group, src, n);  // Writes to GPU
 * B's buffers via NVLink
 *   }
 *
 *   // Kernel (receiver on GPU B)
 *   __global__ void recvKernel(P2pNvlTransportDevice p2p, void* dst, size_t n)
 * { auto group = make_warp_group(); p2p.recv(group, dst, n);  // Reads from own
 * local buffers
 *   }
 */
class P2pNvlTransportDevice {
 public:
  // Chunk index used for metadata exchange in send_one/recv_one
  static constexpr std::size_t kMetadataChunkIndex = 0;

  __host__ __device__ P2pNvlTransportDevice() = default;
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

  __host__ __device__ ~P2pNvlTransportDevice() = default;

  /**
   * send - Transfer data to peer GPU over NVLink
   *
   * Sends 'nbytes' bytes from srcbuff to the peer GPU using pipelined staged
   * transfer with fine-grained chunk-level synchronization. All threads in the
   * group cooperate to transfer the data in parallel.
   *
   * ALGORITHM:
   * ==========
   * 1. Divide transfer into STEPS (dataBufferSize bytes each)
   * 2. For each step:
   *    a. Select pipeline SLOT: slotIdx = stepId % pipelineDepth
   *    b. Calculate buffer offset: slotIdx × dataBufferSize
   *    c. Divide step into CHUNKS for parallel warp processing
   * 3. For each chunk (distributed across warps):
   *    a. WAIT: Spin until state == -1 (receiver freed the buffer)
   *    b. COPY: src[stepOffset+chunkOffset] →
   * remoteBuffer[slotOffset+chunkOffset] c. SYNC: group.sync() to ensure all
   * threads complete copy d. SIGNAL: Leader sets state = stepId (data ready for
   * receiver)
   *
   * REMOTE-WRITE PATTERN:
   * Data is written directly to receiver's local buffer via NVLink, so
   * receiver can read from local memory without NVLink latency.
   *
   * @param group ThreadGroup for cooperative processing (all threads
   * participate)
   * @param srcbuff Source data pointer (device memory)
   * @param nbytes Number of bytes to send
   *
   * EXAMPLE:
   * ========
   * Transfer 1GB with 256MB buffer, 512KB chunks, pipelineDepth=4:
   *
   *   totalSteps = ceil(1GB / 256MB) = 4 steps
   *   chunksPerStep = ceil(256MB / 512KB) = 512 chunks
   *
   *   Step 0: slot[0], offset 0MB,   stepId=0
   *   Step 1: slot[1], offset 256MB, stepId=1
   *   Step 2: slot[2], offset 512MB, stepId=2
   *   Step 3: slot[3], offset 768MB, stepId=3
   *   Step 4: slot[0], offset 0MB,   stepId=4  ← slot reused!
   *
   * OFFSET CALCULATIONS:
   * ====================
   *   pipelineIdx = stepId % pipelineDepth
   *   dataBufferOffset = pipelineIdx × dataBufferSize    (into staging buffer)
   *   stateOffset = pipelineIdx × chunksPerStep          (into state buffer)
   *   stepOffset = stepId × dataBufferSize               (into source data)
   **/
  __device__ __forceinline__ void send(
      ThreadGroup& group,
      void* srcbuff,
      std::size_t nbytes,
      uint32_t call_index = 0,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__
    char* src = reinterpret_cast<char*>(srcbuff);

    // REMOTE-WRITE PATTERN:
    // Sender writes data directly to RECEIVER's local buffer via NVLink.
    // Benefits: Receiver reads from local memory (faster read, no NVLink hop)
    // Trade-off: Sender's copy goes over NVLink
    char* sendBuffer = remoteState_.dataBuffer;
    // Extract raw pointer to avoid aliasing issues (see DeviceSpan.cuh).
    ChunkState* const sendStates = remoteState_.stateBuffer.data();

    const std::size_t totalSteps =
        (nbytes + options_.dataBufferSize - 1) / options_.dataBufferSize;
    const std::size_t kChunkSize = options_.chunkSize;
    const std::size_t chunksPerStep =
        (options_.dataBufferSize + kChunkSize - 1) / kChunkSize;

    for (std::size_t stepId = 0; stepId < totalSteps; ++stepId) {
      // Calculate pipeline slot index for this step
      const std::size_t pipelineIdx = stepId % options_.pipelineDepth;
      const std::size_t dataBufferOffset =
          pipelineIdx * options_.dataBufferSize;
      const std::size_t stateOffset = pipelineIdx * chunksPerStep;

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

        if (chunkBytes == 0) {
          return;
        }

        ChunkState& chunkState = sendStates[stateOffset + chunkIdx];

        chunkState.wait_ready_to_send(group, timeout);

        memcpy_vectorized(
            sendBuffer + dataBufferOffset + chunkOffset,
            src + stepOffset + chunkOffset,
            chunkBytes,
            group);

        chunkState.ready_to_recv(group, stepId, call_index);
      });
    }
#endif
  }

  /**
   * recv - Receive data from peer GPU over NVLink
   *
   * Receives 'nbytes' bytes into dstbuff from the peer GPU's send() call.
   * Must be called simultaneously with peer's send() for the same byte count.
   *
   * ALGORITHM:
   * ==========
   * 1. Divide transfer into STEPS (dataBufferSize bytes each)
   * 2. For each step:
   *    a. Select pipeline SLOT: slotIdx = stepId % pipelineDepth
   *    b. Calculate buffer offset: slotIdx × dataBufferSize
   *    c. Divide step into CHUNKS for parallel warp processing
   * 3. For each chunk (distributed across warps):
   *    a. WAIT: Spin until state == stepId (sender wrote data)
   *    b. COPY: localBuffer[slotOffset+chunkOffset] →
   * dst[stepOffset+chunkOffset] c. SYNC: group.sync() to ensure all threads
   * complete copy d. SIGNAL: Leader sets state = -1 (buffer free for sender to
   * reuse)
   *
   * REMOTE-WRITE PATTERN:
   * Data is read from LOCAL buffer (sender wrote here via NVLink), so
   * receiver reads from local memory without NVLink latency.
   *
   * @param group ThreadGroup for cooperative processing (all threads
   * participate)
   * @param dstbuff Destination data pointer (device memory)
   * @param nbytes Number of bytes to receive (must match sender's count)
   *
   * SYNCHRONIZATION:
   * ================
   *   Sender                          Receiver
   *   ──────                          ────────
   *   wait(state == -1)
   *   copy data ──────────────────▶   [data arrives in local buffer]
   *   state = stepId ─────────────▶   wait(state == stepId)
   *                                   copy data to dst
   *                                   state = -1 ────────▶ [sender unblocks]
   */
  __device__ __forceinline__ void recv(
      ThreadGroup& group,
      void* dstbuff,
      std::size_t nbytes,
      uint32_t call_index = 0,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__
    char* dst = reinterpret_cast<char*>(dstbuff);

    // REMOTE-WRITE PATTERN:
    // Receiver reads from LOCAL buffer (sender wrote here via NVLink).
    // Benefits: Local memory read is faster than reading over NVLink
    char* recvBuffer = localState_.dataBuffer;
    // Extract raw pointer to avoid aliasing issues (see DeviceSpan.cuh).
    ChunkState* const recvStates = localState_.stateBuffer.data();

    const std::size_t totalSteps =
        (nbytes + options_.dataBufferSize - 1) / options_.dataBufferSize;
    const std::size_t kChunkSize = options_.chunkSize;
    const std::size_t chunksPerStep =
        (options_.dataBufferSize + kChunkSize - 1) / kChunkSize;

    for (std::size_t stepId = 0; stepId < totalSteps; stepId++) {
      // Calculate pipeline slot index for this step
      const std::size_t pipelineIdx = stepId % options_.pipelineDepth;
      const std::size_t dataBufferOffset =
          pipelineIdx * options_.dataBufferSize;
      const std::size_t stateOffset = pipelineIdx * chunksPerStep;

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

        if (chunkBytes == 0) {
          return;
        }

        ChunkState& chunkState = recvStates[stateOffset + chunkIdx];

        chunkState.wait_ready_to_recv(group, stepId, call_index, timeout);

        memcpy_vectorized(
            dst + stepOffset + chunkOffset,
            recvBuffer + dataBufferOffset + chunkOffset,
            chunkBytes,
            group);

        chunkState.ready_to_send(group);
      });
    }
#endif
  }

  /**
   * send_one - Send a single chunk with metadata
   *
   * Sends a single data chunk to the peer GPU along with metadata.
   * Thread-group 0 writes metadata (nbytes, offset, has_more) to the
   * receiver's first ChunkState before the data transfer begins.
   * The metadata is communicated through ChunkState fields and becomes
   * visible to the receiver when ready_to_recv is signaled (via release-store).
   *
   * INPUTS:
   * @param group ThreadGroup for cooperative processing (all threads
   *              participate)
   * @param src Source data pointer (device memory, can be nullptr if nbytes=0)
   * @param nbytes Number of bytes to send (can be 0 for metadata-only signal)
   * @param call_index Call index for disambiguating multiple send_one calls
   *                  in the same kernel (default: 0)
   * @param offset_in_output Offset in the receiver's output buffer where this
   *                         chunk should be placed (default: 0)
   * @param has_more Whether more chunks are coming after this one
   *                 (default: false for single chunk transfers)
   */
  __device__ void send_one(
      ThreadGroup& group,
      const void* src,
      std::size_t nbytes,
      uint32_t call_index = 0,
      std::size_t offset_in_output = 0,
      bool has_more = false,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__
    ChunkState* const sendStates = remoteState_.stateBuffer.data();

    // same as send(), wait for previous recv_one() to complete
    sendStates[kMetadataChunkIndex].wait_ready_to_send(group, timeout);

    // Thread-group 0 writes metadata to receiver's chunk kMetadataChunkIndex
    // This happens before send() starts, so receiver can read it
    if (group.group_id == 0) {
      sendStates[kMetadataChunkIndex].write_metadata(
          group, nbytes, offset_in_output, has_more);
    }

    // empty data transfer, just do the signaling
    if (nbytes == 0) {
      if (group.group_id == 0) {
        sendStates[kMetadataChunkIndex].ready_to_recv(
            group, kMetadataChunkIndex, call_index);
      }
      return;
    }

    // Now call regular send() to transfer the data
    // send() will handle all the pipelining and synchronization
    send(group, const_cast<void*>(src), nbytes, call_index, timeout);
#endif
  }

  /**
   * recv_one - Receive a single chunk with metadata
   *
   * Receives a single data chunk from the peer GPU along with metadata.
   * All thread-groups first wait for ChunkState[kMetadataChunkIndex] to receive
   * metadata from sender, then call regular recv() to receive the data.
   * The metadata (nbytes, offset, has_more) is read from
   * ChunkState[kMetadataChunkIndex] which was pre-written by the sender's
   * thread-group 0.
   *
   * INPUTS:
   * @param group ThreadGroup for cooperative processing (all threads
   *              participate)
   * @param dst_base Base pointer to the destination buffer (device memory)
   *
   * OUTPUTS:
   * @param nbytes Receives number of bytes in this chunk (required)
   *
   * OPTIONAL INPUTS:
   * @param call_index Call index for disambiguating multiple recv_one calls
   *                  in the same kernel (default: 0)
   *
   * OPTIONAL OUTPUTS (pass nullptr to ignore):
   * @param offset_in_output Receives offset in dst_base where this chunk
   *                         was placed (default: nullptr)
   * @param has_more Receives whether more chunks are coming after this one
   *                 (default: nullptr)
   */
  __device__ void recv_one(
      ThreadGroup& group,
      void* dst_base,
      std::size_t* nbytes,
      uint32_t call_index = 0,
      std::size_t* offset_in_output = nullptr,
      bool* has_more = nullptr,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__
    ChunkState* const recvStates = localState_.stateBuffer.data();

    // ALL thread-groups wait for chunk kMetadataChunkIndex's ready_to_recv to
    // get metadata Step kMetadataChunkIndex is used for the metadata exchange
    recvStates[kMetadataChunkIndex].wait_ready_to_recv(
        group, kMetadataChunkIndex, call_index, timeout);

    // ALL threads read metadata from chunk kMetadataChunkIndex
    // (all threads need nbytes_val to call recv())
    std::size_t nbytes_val, offset_val;
    bool has_more_val;
    recvStates[kMetadataChunkIndex].read_metadata(
        group, nbytes_val, offset_val, has_more_val);

    // Calculate destination pointer using offset
    char* dst = reinterpret_cast<char*>(dst_base) + offset_val;

    // nbytes is always written (required output)
    *nbytes = nbytes_val;
    if (offset_in_output != nullptr) {
      *offset_in_output = offset_val;
    }
    if (has_more != nullptr) {
      *has_more = has_more_val;
    }

    // empty data transfer, just do the signaling
    if (nbytes_val == 0) {
      if (group.group_id == 0) {
        recvStates[kMetadataChunkIndex].ready_to_send(group);
      }
      return;
    }

    // Now call regular recv() to receive the data
    // recv() will handle all the pipelining and synchronization
    // and will signal ready_to_send() for ChunkState[kMetadataChunkIndex] after
    // completion
    recv(group, dst, nbytes_val, call_index, timeout);
#endif
  }

  /**
   * send_multiple - Transfer multiple chunks with varying sizes to peer GPU
   *
   * Sends multiple data chunks specified by indices from srcbuff to the peer
   * GPU. First sends complete chunk sizes array via regular send(), then
   * transfers actual data chunks using send_one() which handles metadata.
   *
   * INPUTS:
   * @param group ThreadGroup for cooperative processing
   * @param srcbuff_d Source buffer containing all chunks laid out sequentially
   * @param chunk_sizes DeviceSpan of ALL chunk sizes
   * @param chunk_indices DeviceSpan of indices of chunks to send to this peer
   *
   * EXAMPLE:
   *   srcbuff_d layout: [chunk0: 100B][chunk1: 200B][chunk2: 150B][chunk3: 50B]
   *   chunk_sizes = {100, 200, 150, 50}
   *   chunk_indices = {1, 3}
   *   -> Sends chunk1 (200B) and chunk3 (50B) to peer
   *
   * PROTOCOL:
   * Phase 1: Send complete chunk sizes array (regular send)
   * Phase 2: For each chunk, use send_one() to transfer data with metadata
   *          - has_more=true for all chunks except the last
   *          - has_more=false for the last chunk (or if no chunks)
   */
  __device__ __forceinline__ void send_multiple(
      ThreadGroup& group,
      const void* srcbuff_d,
      DeviceSpan<const std::size_t> chunk_sizes,
      DeviceSpan<const std::size_t> chunk_indices,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__
    // Extract raw pointers before loops to avoid aliasing issues
    // (see DeviceSpan.cuh "Lambda Capture and Aliasing" note)
    const std::size_t* const chunk_sizes_ptr = chunk_sizes.data();
    const std::size_t* const chunk_indices_ptr = chunk_indices.data();
    const std::size_t chunk_sizes_count = chunk_sizes.size();
    const std::size_t chunk_indices_count = chunk_indices.size();

    // Phase 1: Send complete chunk sizes array
    send(
        group,
        const_cast<void*>(reinterpret_cast<const void*>(chunk_sizes_ptr)),
        chunk_sizes_count * sizeof(std::size_t),
        0,
        timeout);

    // Phase 2: Send each data chunk with metadata
    // Special case: If chunk_indices is empty, send has_more=false signal
    if (chunk_indices_count == 0) {
      // Send empty signal with has_more=false so receiver knows to stop
      send_one(
          group,
          nullptr, // No data
          0, // nbytes = 0
          1, // callIndex = 1
          0, // offset = 0
          false, // has_more = false (no more chunks)
          timeout);
      return;
    }

    // Track cumulative offset to avoid recalculating from 0 for each chunk
    // Assumes chunk_indices are sorted in increasing order
    std::size_t cumulative_offset = 0;
    std::size_t cumulative_idx =
        0; // Offset computed for chunks [0, cumulative_idx)

    for (std::size_t i = 0; i < chunk_indices_count; i++) {
      std::size_t chunk_idx = chunk_indices_ptr[i];
      std::size_t chunk_size = chunk_sizes_ptr[chunk_idx];

      // Extend cumulative_offset to cover chunks [cumulative_idx, chunk_idx)
      while (cumulative_idx < chunk_idx) {
        cumulative_offset += chunk_sizes_ptr[cumulative_idx];
        cumulative_idx++;
      }

      // Send this chunk with metadata
      // has_more = true for all chunks except the last one
      const char* src_ptr =
          reinterpret_cast<const char*>(srcbuff_d) + cumulative_offset;
      bool has_more = (i < chunk_indices_count - 1);
      send_one(
          group,
          src_ptr,
          chunk_size,
          static_cast<uint32_t>(i + 1), // callIndex increments for each call
          cumulative_offset, // offset in output buffer
          has_more,
          timeout);
    }
#endif
  }

  /**
   * recv_multiple - Receive multiple chunks with varying sizes from peer GPU
   *
   * Receives multiple data chunks into recvbuff. First receives complete chunk
   * sizes array via regular recv(), then receives actual data chunks using
   * recv_one() which reads metadata.
   *
   * INPUTS:
   * @param group ThreadGroup for cooperative processing
   *
   * OUTPUTS:
   * @param recvbuff Destination buffer where chunks are written at their
   *                 original offsets (based on chunk_sizes)
   * @param chunk_sizes DeviceSpan populated with received chunk sizes
   *
   * EXAMPLE:
   *   Sender sends chunk1 (200B) and chunk3 (50B) from 4-chunk buffer
   *   chunk_sizes.size() = 4
   *   -> chunk_sizes receives {100, 200, 150, 50}
   *   -> recvbuff written at: offset 100 (200B), offset 450 (50B)
   *
   * PROTOCOL:
   * Phase 1: Receive complete chunk sizes array (regular recv)
   * Phase 2: Use recv_one() to receive chunks until has_more=false
   *          - If first chunk has nbytes=0 and has_more=false, no data chunks
   *          - Otherwise, keep receiving until has_more=false
   */
  __device__ void recv_multiple(
      ThreadGroup& group,
      void* recvbuff,
      DeviceSpan<std::size_t> chunk_sizes,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__
    // Extract raw pointer before use (see DeviceSpan.cuh "Lambda Capture and
    // Aliasing" note)
    std::size_t* const chunk_sizes_ptr = chunk_sizes.data();
    const std::size_t chunk_sizes_count = chunk_sizes.size();

    // Phase 1: Receive complete chunk sizes array
    uint32_t call_index = 0;
    recv(
        group,
        reinterpret_cast<void*>(chunk_sizes_ptr),
        chunk_sizes_count * sizeof(std::size_t),
        call_index,
        timeout);

    // Phase 2: Receive chunks until has_more=false
    char* dst_base = reinterpret_cast<char*>(recvbuff);

    std::size_t nbytes_val = 0;
    std::size_t offset_val = 0;
    bool has_more = false;

    // Receive first chunk
    call_index++;
    recv_one(
        group,
        dst_base,
        &nbytes_val,
        call_index,
        &offset_val,
        &has_more,
        timeout);

    // If nbytes=0 and has_more=false, this is the empty signal (no chunks)
    if (nbytes_val == 0 && !has_more) {
      return;
    }

    // Keep receiving while there are more chunks
    while (has_more) {
      call_index++;
      recv_one(
          group,
          dst_base,
          &nbytes_val,
          call_index,
          &offset_val,
          &has_more,
          timeout);
    }
#endif
  }

 public:
  // Getters for testing
  __host__ const LocalState& getLocalState() const {
    return localState_;
  }

  __host__ const RemoteState& getRemoteState() const {
    return remoteState_;
  }

  /**
   * put - Direct local memory copy using vectorized operations
   *
   * Performs a high-performance vectorized copy from src_d to dst_d using
   * memcpy_vectorized. The work is distributed across ALL thread groups
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
   *
   * @return Number of bytes written by the current thread group
   */
  __device__ __forceinline__ std::size_t
  put(ThreadGroup& group, char* dst_d, const char* src_d, std::size_t nbytes) {
#ifdef __CUDA_ARCH__
    // Early return for no-op cases
    if (nbytes == 0) {
      return 0;
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
    std::size_t totalBytesWritten = 0;
    group.for_each_item_contiguous(numChunks, [&](uint32_t chunkIdx) {
      const std::size_t chunkOffset = chunkIdx * alignedChunkSize;
      const std::size_t currentChunkBytes =
          (chunkOffset + alignedChunkSize <= nbytes) ? alignedChunkSize
                                                     : nbytes - chunkOffset;

      if (currentChunkBytes > 0) {
        memcpy_vectorized(
            dst_d + chunkOffset, // dst_base
            src_d + chunkOffset, // src_base
            currentChunkBytes, // chunk_bytes
            group);
      }
      totalBytesWritten += currentChunkBytes;
    });
    return totalBytesWritten;
#endif
    return 0;
  }

  /**
   * signal_threadgroup - Signal peer GPU via NVLink
   *
   * Sends a signal to the peer's Signal object at the specified index.
   * Only the group leader performs the signal after synchronizing all threads.
   *
   * MEMORY SEMANTICS:
   * - Uses release semantics: all prior memory operations from all threads
   *   in the group are guaranteed to be visible to the peer after the signal.
   * - Uses .sys scope for cross-GPU NVLink coherence.
   *
   * @param group ThreadGroup for cooperative processing (leader signals)
   * @param signal_id Index into the signalBuffer array
   * @param op SIGNAL_SET to store value, SIGNAL_ADD to atomically add value
   * @param value The value to set or add to peer's signal counter
   */
  __device__ __forceinline__ void signal_threadgroup(
      ThreadGroup& group,
      uint64_t signal_id,
      SignalOp op,
      uint64_t value) {
    remoteState_.signalBuffer[signal_id].signal(group, op, value);
  }

  /**
   * wait_signal_until_threadgroup - Wait for signal from peer GPU
   *
   * Waits until the local Signal object at the specified index satisfies
   * the given condition. All threads in the group poll the signal.
   *
   * MEMORY SEMANTICS:
   * - Uses acquire semantics: all subsequent memory operations are guaranteed
   *   to see the peer's writes that occurred before their signal.
   * - Uses .sys scope for cross-GPU NVLink coherence.
   *
   * @param group ThreadGroup for cooperative processing
   * @param signal_id Index into the signalBuffer array
   * @param op The comparison operation (CMP_EQ, CMP_GE, etc.)
   * @param value The value to compare against
   */
  __device__ __forceinline__ void wait_signal_until_threadgroup(
      ThreadGroup& group,
      uint64_t signal_id,
      CmpOp op,
      uint64_t value,
      const Timeout& timeout = Timeout()) {
    localState_.signalBuffer[signal_id].wait_until(group, op, value, timeout);
  }

  /**
   * barrier_sync_threadgroup - Two-sided barrier synchronization with peer GPU
   *
   * Performs a full barrier synchronization between this GPU and the peer GPU
   * over NVLink. Both sides must call this function to complete the barrier.
   *
   * Synchronization protocol:
   * 1. group.sync() - Ensure all local threads have completed prior work
   * 2. Leader signals peer - Writes to peer's barrier state via NVLink
   * 3. Leader waits for peer - Polls local barrier until peer signals
   * 4. group.sync() - Broadcast completion to all threads in the group
   *
   * This provides a full memory fence: all memory operations before the barrier
   * on both GPUs are visible to all threads after the barrier completes.
   *
   * @param group ThreadGroup for cooperative thread synchronization
   * @param barrier_id Index of the barrier to use (must be < numBarriers)
   *
   * All threads in the group must call this function (collective operation).
   * Both GPUs must call with the same barrier_id to synchronize.
   */
  __device__ __forceinline__ void barrier_sync_threadgroup(
      ThreadGroup& group,
      uint64_t barrier_id,
      const Timeout& timeout = Timeout()) {
    // Ensure all prior memory operations are complete
    group.sync();

    // Only global leader performs barrier operations to avoid races where
    // different threads read different counter values.
    if (group.is_leader()) {
      // Signal peer - write to peer's local barrier state via NVLink
      remoteState_.barrierBuffer[barrier_id].arrive();

      // Wait for peer - poll local barrier state until peer signals
      localState_.barrierBuffer[barrier_id].wait(timeout);
    }

    // Ensure all threads wait for leader to complete barrier
    group.sync();
  }

  // ===========================================================================
  // Transport Configuration Accessors
  // ===========================================================================

  /**
   * chunk_size - Get the configured chunk size for this transport.
   */
  __host__ __device__ __forceinline__ std::size_t chunk_size() const {
    return options_.chunkSize;
  }

  /**
   * data_buffer_size - Get the configured data buffer size for this transport.
   */
  __host__ __device__ __forceinline__ std::size_t data_buffer_size() const {
    return options_.dataBufferSize;
  }

  /**
   * pipeline_depth - Get the configured pipeline depth for this transport.
   */
  __host__ __device__ __forceinline__ std::size_t pipeline_depth() const {
    return options_.pipelineDepth;
  }

  __device__ __forceinline__ char* local_data_buffer() const {
    return localState_.dataBuffer;
  }

  __device__ __forceinline__ ChunkState* local_state_buffer() const {
    return localState_.stateBuffer.data();
  }

  __device__ __forceinline__ char* remote_data_buffer() const {
    return remoteState_.dataBuffer;
  }

  __device__ __forceinline__ ChunkState* remote_state_buffer() const {
    return remoteState_.stateBuffer.data();
  }

  // ===========================================================================
  // Streaming Primitives API
  // ===========================================================================
  // The streaming API provides reusable primitives for pipelined collectives.
  // RecvStream yields chunks as they become ready; SendStream provides slots
  // for sending. These can be composed to build intermediate rank logic for
  // broadcast, reduce, all-to-all, etc.
  // ===========================================================================

  /**
   * StagingChunkView - Read-only view of a ready chunk in the staging buffer
   *
   * Yielded by RecvStream::for_each_ready_chunk(). The caller can read from
   * `data` and use `offset`/`size` to determine where to copy.
   *
   * MEMORY SEMANTICS:
   * - `data` points to local staging buffer (receiver's memory)
   * - Data is visible with acquire semantics (predecessor's writes are visible)
   * - Valid only within the callback scope; auto-released after callback
   *   returns
   *
   * LIFETIME:
   * - The staging buffer slot is released when the callback returns
   * - Do NOT store `data` pointer beyond the callback scope
   */
  struct StagingChunkView {
    const char* data; // Pointer into staging buffer (read-only)
    std::size_t offset; // Offset within the full message (always chunk-aligned)
    std::size_t size; // Chunk size in bytes (may be < chunkSize for last chunk)
  };

  /**
   * SendSlotView - Writable view of a staging slot for sending
   *
   * Obtained from SendStream::slot_for() or yielded by
   * SendStream::for_each_slot(). The caller writes data to `data`, then calls
   * commit_slot() to signal the receiver.
   *
   * MEMORY SEMANTICS:
   * - `data` points to REMOTE staging buffer (receiver's memory via NVLink)
   * - Writes go over NVLink to receiver's local memory
   * - commit_slot() provides release semantics (writes visible to receiver)
   *
   * INTERNAL STATE:
   * - stepId_ and stateIndex_ are used by commit_slot() to signal the correct
   *   ChunkState
   * - These are opaque to the caller
   */
  struct SendSlotView {
    char* data; // Writable pointer into remote staging buffer
    std::size_t offset; // Offset within the full message
    std::size_t size; // Chunk size in bytes

    // Internal state for commit_slot() - do not access directly
    std::size_t stepId_; // Step ID for ready_to_recv signaling
    std::size_t stateIndex_; // Index into stateBuffer for commit_slot

    __device__ __forceinline__ SendSlotView(
        char* data,
        std::size_t offset,
        std::size_t size,
        std::size_t stepId,
        std::size_t stateIndex)
        : data(data),
          offset(offset),
          size(size),
          stepId_(stepId),
          stateIndex_(stateIndex) {}
  };

  // Forward declarations for streaming primitives
  class RecvStream;
  class SendStream;

  /**
   * RecvStream - Streaming receive iterator for pipelined transfers
   *
   * Iterates over all chunks assigned to this warp group, yielding each chunk
   * as it becomes ready. The caller processes each chunk via a callback, then
   * the staging slot is automatically released.
   *
   * MEMORY ORDERING:
   * - Acquire semantics before yielding chunk (predecessor's writes are
   *   visible)
   * - Release semantics after callback returns (slot freed for predecessor
   *   reuse)
   *
   * BUFFER ASYMMETRY:
   * - RecvStream reads from LOCAL staging buffer (receiver's memory)
   * - Predecessor wrote to this buffer via NVLink (remote write pattern)
   * - This allows fast local reads without NVLink latency
   *
   * CALLBACK CONTRACT:
   * - Callback MUST NOT exit early without completing its work
   * - Callback MUST call commit_slot() on any SendStream slots acquired
   * - If callback fails, the pipeline will hang (no automatic cleanup)
   *
   * SIGNAL ORDERING:
   * - RecvStream releases predecessor's slot AFTER callback returns
   * - This matches NCCL's postPeer ordering (signal successor, then free
   *   predecessor)
   * - Safe for unidirectional ring topologies
   */
  class RecvStream {
   public:
    /**
     * for_each_ready_chunk - Iterate over all ready chunks
     *
     * For each chunk assigned to this warp group:
     * 1. Blocks until predecessor signals chunk ready (acquire semantics)
     * 2. Yields StagingChunkView to callback
     * 3. After callback returns, releases staging slot (release semantics)
     *
     * @param group ThreadGroup for cooperative processing
     * @param func Callback invoked for each ready chunk: void(StagingChunkView)
     */
    template <typename Func>
    __device__ __forceinline__ void for_each_ready_chunk(
        ThreadGroup& group,
        Func&& func);

   private:
    friend class P2pNvlTransportDevice;

    char* dataBuffer_; // Local staging buffer (receiver reads from here)
    ChunkState* stateBuffer_; // Array of per-chunk synchronization flags
    std::size_t nbytes_; // Total bytes to receive
    std::size_t dataBufferSize_; // Size of one pipeline slot
    std::size_t chunkSize_; // Size of each chunk
    std::size_t pipelineDepth_; // Number of slots
    std::size_t chunksPerStep_; // dataBufferSize / chunkSize (chunks per slot)
    std::size_t totalSteps_; // nbytes / dataBufferSize (total iterations)
    uint32_t callIndex_;
    Timeout timeout_;

    __device__ __forceinline__ RecvStream(
        char* dataBuffer,
        ChunkState* stateBuffer,
        std::size_t nbytes,
        std::size_t dataBufferSize,
        std::size_t chunkSize,
        std::size_t pipelineDepth,
        uint32_t callIndex,
        const Timeout& timeout)
        : dataBuffer_(dataBuffer),
          stateBuffer_(stateBuffer),
          nbytes_(nbytes),
          dataBufferSize_(dataBufferSize),
          chunkSize_(chunkSize),
          pipelineDepth_(pipelineDepth),
          chunksPerStep_((dataBufferSize + chunkSize - 1) / chunkSize),
          totalSteps_((nbytes + dataBufferSize - 1) / dataBufferSize),
          callIndex_(callIndex),
          timeout_(timeout) {}
  };

  /**
   * SendStream - Streaming send iterator for pipelined transfers
   *
   * Provides two APIs for different use cases:
   *
   * 1. CALLBACK API (for_each_slot): For root rank that drives its own
   *    iteration. Iterates over all staging slots, waits for each to be free,
   *    yields to callback, then AUTO-SIGNALS receiver after callback returns.
   *
   * 2. POSITIONAL API (slot_for + commit_slot): For intermediate ranks called
   *    inside RecvStream's callback. Maps a received chunk's offset to the
   *    corresponding send staging slot. REQUIRES MANUAL commit_slot() call.
   *
   * MEMORY ORDERING:
   * - slot_for() waits with acquire semantics (successor freed the slot)
   * - commit_slot() signals with release semantics (writes visible to
   *   successor)
   *
   * BUFFER ASYMMETRY:
   * - SendStream writes to REMOTE staging buffer (receiver's memory via
   *   NVLink)
   * - This is the "remote write" pattern: receiver reads from local memory
   *   (fast)
   */
  class SendStream {
   public:
    // --- Callback API (for root rank — drives its own iteration) ---
    // AUTO-COMMITS after callback returns

    /**
     * for_each_slot - Iterate over all staging slots
     *
     * For each slot assigned to this warp group:
     * 1. Waits for slot to be free (successor consumed previous data)
     * 2. Yields SendSlotView to callback (caller writes data)
     * 3. After callback returns, AUTO-SIGNALS receiver (data ready)
     *
     * NOTE: Do NOT call commit_slot() inside this callback - it auto-commits.
     *
     * @param group ThreadGroup for cooperative processing
     * @param func Callback invoked for each slot: void(SendSlotView)
     */
    template <typename Func>
    __device__ __forceinline__ void for_each_slot(
        ThreadGroup& group,
        Func&& func);

    // --- Positional API (for intermediate rank — called inside RecvStream)
    // --- REQUIRES MANUAL commit_slot() call

    /**
     * slot_for - Get send slot corresponding to a received chunk
     *
     * Reverse-maps recv_chunk.offset to the corresponding send staging slot.
     * Waits for the slot to be free before returning.
     *
     * IMPORTANT: Assumes recv and send transports have matching configs.
     *
     * @param group ThreadGroup for cooperative processing
     * @param recv_chunk The received chunk (from RecvStream)
     * @return SendSlotView pointing to the corresponding send staging buffer
     */
    __device__ __forceinline__ SendSlotView
    slot_for(ThreadGroup& group, const StagingChunkView& recv_chunk);

    /**
     * commit_slot - Signal receiver that data is ready
     *
     * MUST be called after writing data to slot.data when using slot_for().
     * Do NOT call when using for_each_slot() (it auto-commits).
     *
     * @param group ThreadGroup for cooperative processing
     * @param slot The slot returned by slot_for()
     */
    __device__ __forceinline__ void commit_slot(
        ThreadGroup& group,
        const SendSlotView& slot);

   private:
    friend class P2pNvlTransportDevice;

    char* dataBuffer_; // Remote staging buffer (receiver's memory via NVLink)
    ChunkState* stateBuffer_; // Array of per-chunk synchronization flags
    std::size_t nbytes_; // Total bytes to receive
    std::size_t dataBufferSize_; // Size of one pipeline slot
    std::size_t chunkSize_; // Size of each chunk
    std::size_t pipelineDepth_; // Number of slots
    std::size_t chunksPerStep_; // dataBufferSize / chunkSize (chunks per slot)
    std::size_t totalSteps_; // nbytes / dataBufferSize (total iterations)
    uint32_t callIndex_;
    Timeout timeout_;

    __device__ __forceinline__ SendStream(
        char* dataBuffer,
        ChunkState* stateBuffer,
        std::size_t nbytes,
        std::size_t dataBufferSize,
        std::size_t chunkSize,
        std::size_t pipelineDepth,
        uint32_t callIndex,
        const Timeout& timeout)
        : dataBuffer_(dataBuffer),
          stateBuffer_(stateBuffer),
          nbytes_(nbytes),
          dataBufferSize_(dataBufferSize),
          chunkSize_(chunkSize),
          pipelineDepth_(pipelineDepth),
          chunksPerStep_((dataBufferSize + chunkSize - 1) / chunkSize),
          totalSteps_((nbytes + dataBufferSize - 1) / dataBufferSize),
          callIndex_(callIndex),
          timeout_(timeout) {}
  };

  /**
   * recv_stream - Create a RecvStream for streaming receive
   *
   * Creates a RecvStream that yields chunks as they become ready from the
   * predecessor. Use for intermediate ranks that need to process chunks
   * as they arrive.
   *
   * NOTE: The caller is responsible for calling timeout.start() before
   * creating the stream if timeout enforcement is desired.
   *
   * @param nbytes Total number of bytes to receive
   * @param call_index Call index for disambiguating multiple calls (default 0)
   * @param timeout Timeout configuration (default none)
   * @return RecvStream for iterating over ready chunks
   */
  __device__ __forceinline__ RecvStream recv_stream(
      std::size_t nbytes,
      uint32_t call_index = 0,
      const Timeout& timeout = Timeout()) const;

  /**
   * send_stream - Create a SendStream for streaming send
   *
   * Creates a SendStream for sending chunks to the successor. Use for root
   * rank (with for_each_slot) or intermediate ranks (with slot_for +
   * commit_slot).
   *
   * NOTE: The caller is responsible for calling timeout.start() before
   * creating the stream if timeout enforcement is desired.
   *
   * @param nbytes Total number of bytes to send
   * @param call_index Call index for disambiguating multiple calls (default 0)
   * @param timeout Timeout configuration (default none)
   * @return SendStream for iterating over or accessing send slots
   */
  __device__ __forceinline__ SendStream send_stream(
      std::size_t nbytes,
      uint32_t call_index = 0,
      const Timeout& timeout = Timeout()) const;

 private:
  const int myRank_{-1};
  const int peerRank_{-1};
  const P2pNvlTransportOptions options_;
  LocalState localState_;
  RemoteState remoteState_;
};

// =============================================================================
// Streaming Primitives Implementation
// =============================================================================

template <typename Func>
__device__ __forceinline__ void
P2pNvlTransportDevice::RecvStream::for_each_ready_chunk(
    ThreadGroup& group,
    Func&& func) {
#ifdef __CUDA_ARCH__
  // Cross-step deferred signaling optimization:
  // With pipelineDepth >= 2, we defer SIGNAL-2 (ready_to_send, which frees
  // the predecessor's staging slot) from step N until after WAIT-1 of step
  // N+1. This removes SIGNAL-2 from the critical path between steps:
  //   Before: func(N) -> SIGNAL-2(N) -> WAIT-1(N+1) -> func(N+1)
  //   After:  func(N) -> WAIT-1(N+1) -> SIGNAL-2(N) -> func(N+1)
  // Safe because the predecessor reuses slot N%pd at step N+pd, and we
  // release it at step N+1 — leaving pd-1 steps of slack.
  // Falls back to immediate release when pipelineDepth == 1.
  const bool deferRelease = (pipelineDepth_ >= 2);

  for (std::size_t stepId = 0; stepId < totalSteps_; ++stepId) {
    const std::size_t pipelineIdx = stepId % pipelineDepth_;
    const std::size_t stepOffset = stepId * dataBufferSize_;
    const std::size_t stepBytes = (stepOffset + dataBufferSize_ <= nbytes_)
        ? dataBufferSize_ // Full step
        : nbytes_ - stepOffset; // Partial (last step)
    const std::size_t numChunks = (stepBytes + chunkSize_ - 1) / chunkSize_;
    const std::size_t stateOffset = pipelineIdx * chunksPerStep_;
    const std::size_t stagingOffset = pipelineIdx * dataBufferSize_;

    // Compute previous step info for deferred release
    std::size_t prevStateOffset = 0;
    std::size_t prevNumChunks = 0;
    if (deferRelease && stepId > 0) {
      const std::size_t prevPipelineIdx = (stepId - 1) % pipelineDepth_;
      prevStateOffset = prevPipelineIdx * chunksPerStep_;
      const std::size_t prevStepOffset = (stepId - 1) * dataBufferSize_;
      const std::size_t prevStepBytes =
          (prevStepOffset + dataBufferSize_ <= nbytes_)
          ? dataBufferSize_
          : nbytes_ - prevStepOffset;
      prevNumChunks = (prevStepBytes + chunkSize_ - 1) / chunkSize_;
    }

    // Iterate over max(numChunks, prevNumChunks) to ensure all previous
    // step's chunks get released even if the current step has fewer chunks
    // (can happen for the last step when nbytes is not a multiple of
    // dataBufferSize)
    const std::size_t iterChunks = (deferRelease && stepId > 0)
        ? (numChunks > prevNumChunks ? numChunks : prevNumChunks)
        : numChunks;

    // Each warp gets assigned a contiguous range of chunks
    group.for_each_item_contiguous(iterChunks, [&](uint32_t chunkIdx) {
      const bool hasCurrentChunk = (chunkIdx < numChunks);
      const std::size_t chunkOffset = chunkIdx * chunkSize_;
      const std::size_t chunkBytes = hasCurrentChunk
          ? ((chunkOffset + chunkSize_ <= stepBytes) ? chunkSize_
                                                     : stepBytes - chunkOffset)
          : 0;

      // WAIT-1: Wait for predecessor's data (acquire semantics)
      if (hasCurrentChunk && chunkBytes > 0) {
        ChunkState& state = stateBuffer_[stateOffset + chunkIdx];
        state.wait_ready_to_recv(group, stepId, callIndex_, timeout_);
      }

      // Deferred SIGNAL-2: release previous step's staging slot AFTER
      // waiting for current step's data, overlapping with the wait
      if (deferRelease && stepId > 0 && chunkIdx < prevNumChunks) {
        stateBuffer_[prevStateOffset + chunkIdx].ready_to_send(group);
      }

      // Yield chunk to callback
      if (hasCurrentChunk && chunkBytes > 0) {
        StagingChunkView view{
            dataBuffer_ + stagingOffset + chunkOffset, // Data pointer
            stepOffset + chunkOffset, // Offset (in full message)
            chunkBytes}; // Chunk size
        func(view);
      }

      // Immediate release when deferred signaling is disabled (pd == 1)
      if (!deferRelease && hasCurrentChunk && chunkBytes > 0) {
        stateBuffer_[stateOffset + chunkIdx].ready_to_send(group);
      }
    });
  }

  // Release the last step's staging slots (deferred from the final iteration)
  if (deferRelease && totalSteps_ > 0) {
    const std::size_t lastStepId = totalSteps_ - 1;
    const std::size_t lastPipelineIdx = lastStepId % pipelineDepth_;
    const std::size_t lastStepOffset = lastStepId * dataBufferSize_;
    const std::size_t lastStepBytes =
        (lastStepOffset + dataBufferSize_ <= nbytes_)
        ? dataBufferSize_
        : nbytes_ - lastStepOffset;
    const std::size_t lastNumChunks =
        (lastStepBytes + chunkSize_ - 1) / chunkSize_;
    const std::size_t lastStateOffset = lastPipelineIdx * chunksPerStep_;

    group.for_each_item_contiguous(lastNumChunks, [&](uint32_t chunkIdx) {
      stateBuffer_[lastStateOffset + chunkIdx].ready_to_send(group);
    });
  }
#endif
}

template <typename Func>
__device__ __forceinline__ void
P2pNvlTransportDevice::SendStream::for_each_slot(
    ThreadGroup& group,
    Func&& func) {
#ifdef __CUDA_ARCH__
  for (std::size_t stepId = 0; stepId < totalSteps_; ++stepId) {
    const std::size_t pipelineIdx = stepId % pipelineDepth_;
    const std::size_t stepOffset = stepId * dataBufferSize_;
    const std::size_t stepBytes = (stepOffset + dataBufferSize_ <= nbytes_)
        ? dataBufferSize_ // Full step
        : nbytes_ - stepOffset; // Partial (last step)
    const std::size_t numChunks = (stepBytes + chunkSize_ - 1) / chunkSize_;
    const std::size_t stateOffset = pipelineIdx * chunksPerStep_;
    const std::size_t stagingOffset = pipelineIdx * dataBufferSize_;

    group.for_each_item_contiguous(numChunks, [&](uint32_t chunkIdx) {
      const std::size_t chunkOffset = chunkIdx * chunkSize_;
      const std::size_t chunkBytes = (chunkOffset + chunkSize_ <= stepBytes)
          ? chunkSize_
          : stepBytes - chunkOffset;

      if (chunkBytes == 0) {
        return;
      }

      ChunkState& state = stateBuffer_[stateOffset + chunkIdx];

      // Wait for slot to be free (acquire semantics)
      state.wait_ready_to_send(group, timeout_);

      // Yield slot to callback
      SendSlotView slot{
          dataBuffer_ + stagingOffset + chunkOffset,
          stepOffset + chunkOffset,
          chunkBytes,
          stepId,
          stateOffset + chunkIdx};
      func(slot);

      // AUTO-SIGNAL receiver (release semantics)
      state.ready_to_recv(group, stepId, callIndex_);
    });
  }
#endif
}

__device__ __forceinline__ P2pNvlTransportDevice::SendSlotView
P2pNvlTransportDevice::SendStream::slot_for(
    ThreadGroup& group,
    const StagingChunkView& recv_chunk) {
#ifdef __CUDA_ARCH__
  // Reverse-map offset → step and pipeline slot
  const std::size_t stepId = recv_chunk.offset / dataBufferSize_;
  const std::size_t pipelineIdx = stepId % pipelineDepth_;
  const std::size_t offsetInStep = recv_chunk.offset - stepId * dataBufferSize_;
  const std::size_t chunkIdx = offsetInStep / chunkSize_;
  const std::size_t stateOffset = pipelineIdx * chunksPerStep_;
  const std::size_t stagingOffset = pipelineIdx * dataBufferSize_;
  const std::size_t stateIndex = stateOffset + chunkIdx;

  ChunkState& state = stateBuffer_[stateIndex];

  // Wait for slot to be free (acquire semantics)
  state.wait_ready_to_send(group, timeout_);

  return SendSlotView{
      dataBuffer_ + stagingOffset + offsetInStep,
      recv_chunk.offset,
      recv_chunk.size,
      stepId,
      stateIndex};
#else
  return SendSlotView{nullptr, 0, 0, 0, 0};
#endif
}

__device__ __forceinline__ void P2pNvlTransportDevice::SendStream::commit_slot(
    ThreadGroup& group,
    const SendSlotView& slot) {
#ifdef __CUDA_ARCH__
  ChunkState& state = stateBuffer_[slot.stateIndex_];
  // Signal receiver: data ready (release semantics)
  state.ready_to_recv(group, slot.stepId_, callIndex_);
#endif
}

__device__ __forceinline__ P2pNvlTransportDevice::RecvStream
P2pNvlTransportDevice::recv_stream(
    std::size_t nbytes,
    uint32_t call_index,
    const Timeout& timeout) const {
#ifdef __CUDA_ARCH__
  return RecvStream{
      localState_.dataBuffer,
      localState_.stateBuffer.data(),
      nbytes,
      options_.dataBufferSize,
      options_.chunkSize,
      options_.pipelineDepth,
      call_index,
      timeout};
#else
  return RecvStream{nullptr, nullptr, 0, 0, 0, 0, 0, Timeout{}};
#endif
}

__device__ __forceinline__ P2pNvlTransportDevice::SendStream
P2pNvlTransportDevice::send_stream(
    std::size_t nbytes,
    uint32_t call_index,
    const Timeout& timeout) const {
#ifdef __CUDA_ARCH__
  return SendStream{
      remoteState_.dataBuffer,
      remoteState_.stateBuffer.data(),
      nbytes,
      options_.dataBufferSize,
      options_.chunkSize,
      options_.pipelineDepth,
      call_index,
      timeout};
#else
  return SendStream{nullptr, nullptr, 0, 0, 0, 0, 0, Timeout{}};
#endif
}

} // namespace comms::pipes
