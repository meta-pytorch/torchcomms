// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>
#include "comms/pipes/ChunkState.cuh"
#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/SignalState.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

/**
 * LocalState - Pointers to local GPU's buffers
 *
 * With REMOTE-WRITE pattern:
 * - Sender writes to RemoteState (peer's local buffers via NVLink)
 * - Receiver reads from LocalState (own local buffers)
 *
 * This means LocalState buffers are the DESTINATION for incoming data.
 * Uses DeviceSpan for safe, bounds-checked access to chunk states.
 *
 * For barrier synchronization:
 * - signalState: Local SignalState that peer signals to (via NVLink).
 *   Local PE waits on this state's signal_ counter to know peer has reached
 *   barrier. The localState_ counter tracks how many signals we expect.
 */
struct LocalState {
  char* dataBuffer;
  DeviceSpan<ChunkState> stateBuffer;
  SignalState* signalState;
};

/**
 * RemoteState - Pointers to peer GPU's buffers (via NVLink peer mapping)
 *
 * With REMOTE-WRITE pattern:
 * - Sender writes directly to these buffers (peer's local memory)
 * - This allows receiver to read from local memory (faster)
 *
 * These pointers are obtained via IPC and point to peer's LocalState buffers.
 * Uses DeviceSpan for safe, bounds-checked access to chunk states.
 *
 * For barrier synchronization:
 * - signalState: Pointer to peer's SignalState (via NVLink). Local PE signals
 *   this state to notify peer that we have reached the barrier.
 */
struct RemoteState {
  char* dataBuffer;
  DeviceSpan<ChunkState> stateBuffer;
  SignalState* signalState;
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
 *                │ signals readyToRecv(stepId)
 *                ▼
 *        ┌───────────────┐
 *    ┌─▶ │ READY_TO_RECV │
 *    │   │   (stepId)    │
 *    │   └───────┬───────┘
 *    │           │
 *    │           │ recv() waits for READY_TO_RECV, copies data,
 *    │           │ signals readyToSend()
 *    │           ▼
 *    │   ┌───────────────┐
 *    │   │ READY_TO_SEND │
 *    │   │     (-1)      │
 *    │   └───────┬───────┘
 *    │           │
 *    │           │ send() waits for READY_TO_SEND, copies data,
 *    │           │ signals readyToRecv(stepId)
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
  __device__ __forceinline__ void
  send(ThreadGroup& group, void* srcbuff, std::size_t nbytes) {
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

        chunkState.waitReadyToSend(group);

        memcpy_vectorized(
            sendBuffer + dataBufferOffset + chunkOffset,
            src + stepOffset + chunkOffset,
            chunkBytes,
            group);

        chunkState.readyToRecv(group, stepId);
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
  __device__ __forceinline__ void
  recv(ThreadGroup& group, void* dstbuff, std::size_t nbytes) {
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

        chunkState.waitReadyToRecv(group, stepId);

        memcpy_vectorized(
            dst + stepOffset + chunkOffset,
            recvBuffer + dataBufferOffset + chunkOffset,
            chunkBytes,
            group);

        chunkState.readyToSend(group);
      });
    }
#endif
  }

  // Getters for testing
  __host__ const LocalState& getLocalState() const {
    return localState_;
  }

  __host__ const RemoteState& getRemoteState() const {
    return remoteState_;
  }

  /**
   * write - Not implemented for P2pNvlTransportDevice
   *
   * P2pNvlTransportDevice is designed for remote P2P transfers over NVLink.
   * For local memory copies, use P2pSelfTransportDevice instead.
   * Calling this method will trap and abort the kernel.
   */
  __device__ __forceinline__ void write(
      ThreadGroup& group,
      char* dst_d,
      const char* src_d,
      std::size_t nbytes) {
#ifdef __CUDA_ARCH__
    __trap(); // Abort kernel if write is called on P2pNvlTransportDevice
#endif
  }

  /**
   * barrier - Pairwise barrier synchronization with peer GPU
   *
   * Ensures both this PE and the peer PE have reached this point before
   * either proceeds. Uses signal/wait pattern over NVLink for synchronization.
   *
   * ALGORITHM (Pairwise Barrier):
   * =============================
   *
   * Each barrier() call increments a monotonic counter. The protocol is:
   *
   *   1. SIGNAL: Call signal() on PEER's local SignalState (via NVLink)
   *      - Atomically increments peer's signal_ counter
   *      - This tells peer "I have reached barrier N"
   *
   *   2. WAIT: Call wait() on LOCAL SignalState
   *      - Increments local localState_ to get expected count
   *      - Polls local signal_ until it reaches expected count
   *      - This waits for peer to signal "I have reached barrier N"
   *
   *   PE 0 (myRank=0)                    PE 1 (peerRank=1)
   *   ───────────────                    ─────────────────
   *   signal_=0, localState_=0           signal_=0, localState_=0
   *       │                                  │
   *       │ signal() on peer's state ───────▶│ PE1.signal_++
   *       │                                  │
   *       │ PE0.signal_++ ◀──────────────────│ signal() on peer's state
   *       │                                  │
   *   wait() on own state               wait() on own state
   *     localState_++ → expected=1        localState_++ → expected=1
   *     poll signal_ >= 1                 poll signal_ >= 1
   *       │                                  │
   *   barrier complete                   barrier complete
   *
   * MEMORY ORDERING:
   * ================
   * - All RMA operations before barrier() are visible to peer after barrier()
   * - Uses acquire/release semantics for proper memory ordering
   * - __threadfence_system() ensures all prior writes are globally visible
   *
   * USAGE EXAMPLE:
   * ==============
   *   // GPU 0 kernel (sender)
   *   __global__ void kernel(P2pNvlTransportDevice p2p, void* src, size_t n) {
   *     auto group = make_warp_group();
   *     p2p.write(group, src1, nbytes);  // Write data to peer
   *     p2p.write(group, src2, nbytes);  // Write data to peer
   *     p2p.barrier(group);        // Notify peer we're done
   *     // Now safe to reuse src buffer
   *   }
   *
   *   // GPU 1 kernel (receiver)
   *   __global__ void kernel(P2pNvlTransportDevice p2p, void* dst, size_t n) {
   *     auto group = make_warp_group();
   *     p2p.barrier(group);        // Wait for peer to finish
   *     // Now safe to use data
   *   }
   *
   * @param group ThreadGroup for cooperative processing (leader signals/waits)
   */
  __device__ __forceinline__ void barrier_threadgroup(ThreadGroup& group) {
    // Ensure all prior memory operations are complete
    group.sync();

    // Only global leader performs barrier operations to avoid races where
    // different threads read different counter values.
    if (group.is_leader()) {
      // Memory fence here to make sure memory visible to all system
      comms::device::threadfence_system();

      // Signal peer - write to peer's local barrier state via NVLink
      remoteState_.signalState->signal();

      // Wait for peer - poll local barrier state until peer signals
      localState_.signalState->wait();
    }

    // Ensure all threads wait for leader to complete barrier
    group.sync();
  }

 private:
  const int myRank_{-1};
  const int peerRank_{-1};
  const P2pNvlTransportOptions options_;
  LocalState localState_;
  RemoteState remoteState_;
};

} // namespace comms::pipes
