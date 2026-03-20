// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstring>
#include "comms/pipes/BarrierState.cuh"
#include "comms/pipes/ChunkState.cuh"
#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/DeviceCheck.cuh"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/SignalState.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/ll128/Ll128Ops.cuh"

namespace comms::pipes {

/**
 * LocalState - Pointers to local GPU's buffers
 *
 * With REMOTE-WRITE pattern:
 * - Sender writes to RemoteState (peer's local buffers via NVLink)
 * - Receiver reads from LocalState (own local buffers)
 *
 * This means LocalState buffers are the DESTINATION for incoming data.
 *
 * Chunk state buffers (usage depends on useDualStateBuffer option):
 *
 * SINGLE STATE MODE (useDualStateBuffer=false):
 *   - Only receiverStateBuffer is used
 *   - receiverStateBuffer: State to poll if I am a receiver
 *     - Sender signals data ready via NVLink write
 *     - Receiver waits locally, then signals ready-to-send locally
 *   - senderStateBuffer: Not used (empty span)
 *
 * DUAL STATE MODE (useDualStateBuffer=true):
 *   - Both buffers are used for fully local polling
 *   - receiverStateBuffer: State to poll if I am a receiver (peer writes
 *     via NVLink to signal data ready)
 *   - senderStateBuffer: State to poll if I am a sender (peer writes
 *     via NVLink to signal ready-to-send after reading)
 */
struct LocalState {
  char* dataBuffer;
  DeviceSpan<ChunkState> receiverStateBuffer;
  DeviceSpan<ChunkState> senderStateBuffer;
  DeviceSpan<SignalState> signalBuffer;
  DeviceSpan<BarrierState> barrierBuffer;
  Ll128Packet* ll128Buffer{nullptr};
};

/**
 * RemoteState - Pointers to peer GPU's buffers (via NVLink peer mapping)
 *
 * With REMOTE-WRITE pattern:
 * - Sender writes directly to these buffers (peer's local memory)
 * - This allows receiver to read from local memory (faster)
 *
 * These pointers are obtained via IPC and point to peer's LocalState buffers.
 *
 * Chunk state buffers (usage depends on useDualStateBuffer option):
 *
 * SINGLE STATE MODE (useDualStateBuffer=false):
 *   - Only receiverStateBuffer is used (points to peer's receiverStateBuffer)
 *   - receiverStateBuffer: State to signal if I am a sender (I write via
 *     NVLink to signal data ready, or I wait via NVLink for ack)
 *   - senderStateBuffer: Not used (empty span)
 *
 * DUAL STATE MODE (useDualStateBuffer=true):
 *   - Both buffers are used for fully local polling
 *   - receiverStateBuffer: State to signal if I am a sender (I write via
 *     NVLink to signal data ready to peer's receiver)
 *   - senderStateBuffer: State to signal if I am a receiver (I write via
 *     NVLink to signal ready-to-send to peer's sender after reading)
 */
struct RemoteState {
  char* dataBuffer;
  DeviceSpan<ChunkState> receiverStateBuffer;
  DeviceSpan<ChunkState> senderStateBuffer;
  DeviceSpan<SignalState> signalBuffer;
  DeviceSpan<BarrierState> barrierBuffer;
  Ll128Packet* ll128Buffer{nullptr};
};

/**
 * P2pNvlTransportOptions - Configuration for P2P NVLink transport
 *
 * Defines the buffer sizes and chunking parameters for staged transfers.
 * - dataBufferSize: Size of ONE pipeline slot (determines max per-step
 * transfer)
 * - chunkSize: Size of each chunk for parallel processing
 * - pipelineDepth: Number of buffer slots for pipelining (typically 2-8)
 * - useDualStateBuffer: If true, use dual chunk state buffers (one on each
 *   side) for local polling on both sender and receiver. If false (default),
 *   use single chunk state buffer on receiver side only.
 *
 * Total memory allocated = pipelineDepth × dataBufferSize
 *
 * STATE BUFFER MODES:
 * ===================
 * Single State (useDualStateBuffer=false, default):
 *   - 1 ChunkState per chunk, stored on receiver side
 *   - Sender polls over NVLink (slower), receiver polls locally (faster)
 *   - Lower memory usage
 *
 * Dual State (useDualStateBuffer=true):
 *   - 2 ChunkStates per chunk: one on receiver (receiverStateBuffer for data
 *     ready signal), one on sender (senderStateBuffer for ready-to-send signal)
 *   - Both sender and receiver poll locally (faster on both sides)
 *   - Higher memory usage, better performance for high-throughput workloads
 *   - REQUIRES for_each_item_strided for chunk distribution (see below)
 *
 * DUAL STATE MODE - STRIDED CHUNK ASSIGNMENT:
 * ===========================================
 * Dual state mode MUST use for_each_item_strided to ensure each chunk is
 * always assigned to the same thread group within a kernel. This is required
 * because:
 *   - ChunkState.unready() uses a plain write with group-wise sync for
 *     efficiency (st.release.gpu is too slow)
 *   - This plain write from one group may not be visible to other groups
 *     without expensive global memory barriers
 *   - With strided assignment, chunk K is ALWAYS assigned to group
 *     (K % total_groups), so the unready write is visible to the same group
 *     after group.sync()
 */
struct P2pNvlTransportOptions {
  std::size_t dataBufferSize;
  std::size_t chunkSize;
  std::size_t pipelineDepth;
  bool useDualStateBuffer{false}; // Default to single state buffer mode
  std::size_t ll128BufferNumPackets{0}; // 0 = no chunking
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
 * STATE MACHINE (per chunk) - DUAL CHUNK STATE
 * =============================================
 *
 * With dual chunk states, ALL waits are local (no NVLink polling):
 * - Each GPU has two state buffers per peer:
 *   1. receiverStateBuffer: State to poll if I am a receiver (peer writes
 *      here to signal data ready)
 *   2. senderStateBuffer: State to poll if I am a sender (peer writes here
 *      to signal ready-to-send after reading)
 *
 * SENDER (GPU A) FLOW:
 * ====================
 * 1. Wait LOCAL: localState_.senderStateBuffer for READY_TO_SEND
 *    - Polls locally for receiver's ready-to-send signal
 * 2. Copy data to remoteState_.dataBuffer (NVLink write)
 * 3. Mark LOCAL senderState as UNREADY to prevent re-sending before
 *    receiver reads (plain write + group sync)
 * 4. Signal REMOTE: remoteState_.receiverStateBuffer = stepId
 *    (This is peer's localState_.receiverStateBuffer)
 *
 * RECEIVER (GPU B) FLOW:
 * ======================
 * 1. Wait LOCAL: localState_.receiverStateBuffer for stepId
 * 2. Copy data from localState_.dataBuffer (local read)
 * 3. Mark LOCAL receiverState as UNREADY to prevent re-reading before
 *    sender writes next (plain write + group sync)
 * 4. Signal REMOTE: remoteState_.senderStateBuffer = READY_TO_SEND
 *    (This is peer's localState_.senderStateBuffer - sender can send again)
 *
 * STATE TRANSITIONS (per pipeline slot):
 * ======================================
 *
 * localState_.senderStateBuffer (sender waits here):
 *   init: READY_TO_SEND (-1)
 *   After sender sends: UNREADY (-2) (prevents re-send before receiver reads)
 *   After receiver reads: READY_TO_SEND (-1) (sender can send again)
 *
 * localState_.receiverStateBuffer (receiver waits here):
 *   init: UNREADY (-2) (no data)
 *   After sender writes: stepId (data ready)
 *   After receiver reads: UNREADY (-2) (prevents re-read before next write)
 *
 * WHY STRIDED CHUNK ASSIGNMENT:
 * =============================
 * The UNREADY state uses a plain write + group.sync() for efficiency
 * (st.release.gpu is too slow). This plain write is only visible to
 * the same thread group after group.sync(), not to other groups.
 * By using for_each_item_strided, chunk K is ALWAYS assigned to
 * group (K % total_groups), ensuring the unready write is visible
 * to the same group in subsequent iterations.
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
  __host__ P2pNvlTransportDevice() = default;

  __host__ P2pNvlTransportDevice(
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
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__
    if (options_.dataBufferSize == 0) {
      printf(
          "P2pNvlTransportDevice::send() requires staging buffer"
          " (dataBufferSize > 0) at %s:%d\n",
          __FILE__,
          __LINE__);
      __trap();
    }
    char* src = reinterpret_cast<char*>(srcbuff);

    char* sendBuffer = remoteState_.dataBuffer;
    // Remote signal buffer: peer's receiverStateBuffer (NVLink write)
    ChunkState* const remoteReceiverStates =
        remoteState_.receiverStateBuffer.data();

    const std::size_t totalSteps =
        (nbytes + options_.dataBufferSize - 1) / options_.dataBufferSize;
    const std::size_t kChunkSize = options_.chunkSize;
    const std::size_t chunksPerStep =
        (options_.dataBufferSize + kChunkSize - 1) / kChunkSize;

    if (options_.useDualStateBuffer) {
      // =====================================================================
      // DUAL CHUNK STATE MODE
      // =====================================================================
      // Uses two ChunkState buffers per peer to enable local polling:
      //   - receiverStateBuffer: State to poll if I am a receiver (sender
      //     writes via NVLink to signal data ready)
      //   - senderStateBuffer: State to poll if I am a sender (receiver
      //     signals via NVLink when ready-to-send after reading)
      //
      // STATE MACHINE (per chunk, showing sync points):
      //   ┌──────────────────────────────────────────────────────────────────┐
      //   │ SENDER (this side)              RECEIVER (peer side)            │
      //   ├──────────────────────────────────────────────────────────────────┤
      //   │ 1. Wait LOCAL senderState       1. Wait LOCAL receiverState     │
      //   │    for READY_TO_SEND               for currentStep value        │
      //   │    (ld.acquire.sys.global)         (ld.acquire.sys.global)      │
      //   │                                                                  │
      //   │ 2. Copy data to peer buffer     2. Copy data from local buffer  │
      //   │    via NVLink                      (no NVLink needed)           │
      //   │                                                                  │
      //   │    ─── group.sync() [inside unready()] ───                      │
      //   │ 3. Mark LOCAL senderState       3. Mark LOCAL receiverState     │
      //   │    as UNREADY (plain write)        as UNREADY (plain write)     │
      //   │                                                                  │
      //   │    ─── group.sync() [inside ready_to_recv/send()] ───           │
      //   │ 4. Signal REMOTE peer via       4. Signal REMOTE sender via     │
      //   │    NVLink st.release.sys to        NVLink st.release.sys        │
      //   │    receiverState (stepId)          READY_TO_SEND to senderState │
      //   └──────────────────────────────────────────────────────────────────┘
      //
      // KEY INSIGHT: Both sender and receiver poll LOCAL memory, avoiding
      // expensive NVLink round-trips for busy-wait synchronization.
      //
      // FORMAL CORRECTNESS — WHY TWO group.sync() CALLS ARE REQUIRED:
      // =============================================================
      //
      // The two syncs (one in unready(), one in ready_to_recv/send()) serve
      // different purposes and both are necessary:
      //
      // Sync #1 (inside unready(), before plain write):
      //   Ensures all threads have finished their memcpy before the leader
      //   writes UNREADY. Without this, some threads may stuck at
      //   wait_ready_to_send().
      //
      // Sync #2 (inside ready_to_recv/send(), before release store):
      //   Ensures all threads in the group observe the UNREADY plain write
      //   before the leader does st.release.sys.global to the REMOTE state.
      //   This is critical because:
      //
      //   - unready() writes value_ = UNREADY via a plain store (not
      //     st.release.sys), so it is only guaranteed visible to threads
      //     that participate in a subsequent group.sync().
      //
      //   - Without sync #2, a non-leader thread could loop back to
      //     wait_ready_to_send() and do ld.acquire.sys.global on the LOCAL
      //     senderState. This acquire load has NO acquire-release pair with
      //     the plain write — they are on the SAME address but the write is
      //     plain, not a release store. Nor does it pair with the release
      //     store in ready_to_recv(), which writes to a DIFFERENT address
      //     (the REMOTE receiverState). So the acquire load could return
      //     the stale READY_TO_SEND value, causing the thread to re-enter
      //     memcpy before the peer has consumed the previous data.
      //
      //   - sync #2 (__syncthreads) acts as a memory fence that makes the
      //     UNREADY plain write visible to all threads in the group,
      //     preventing them from seeing stale READY_TO_SEND.
      //
      // STRIDED ASSIGNMENT: Uses for_each_item_strided to ensure
      // each chunk is always assigned to the same thread group. This is
      // required because unready() uses plain write + group.sync() (not
      // st.release.sys), which is only visible within the same group.
      // =====================================================================
      ChunkState* const localSenderStates =
          localState_.senderStateBuffer.data();

      for (std::size_t stepId = 0; stepId < totalSteps; ++stepId) {
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

        group.for_each_item_strided(numChunksThisStep, [&](uint32_t chunkIdx) {
          const std::size_t chunkOffset = chunkIdx * kChunkSize;
          const std::size_t chunkBytes = (chunkOffset + kChunkSize <= stepBytes)
              ? kChunkSize
              : stepBytes - chunkOffset;

          if (chunkBytes == 0) {
            return;
          }

          const std::size_t chunkStateIdx = stateOffset + chunkIdx;

          // Wait on LOCAL senderStateBuffer for ready-to-send signal
          // (fast local poll - receiver signals when done reading)
          ChunkState& localSenderState = localSenderStates[chunkStateIdx];

          localSenderState.wait_ready_to_send(group, timeout);

          // Copy data to peer's buffer via NVLink
          memcpy_vectorized(
              sendBuffer + dataBufferOffset + chunkOffset,
              src + stepOffset + chunkOffset,
              chunkBytes,
              group);

          // Sync #1 + plain write: barrier all threads, then leader
          // writes UNREADY to local senderState (see correctness note above)
          localSenderState.unready(group);

          // Sync #2 + release store: barrier all threads (flushes the
          // UNREADY plain write), then leader does st.release.sys.global
          // to peer's receiverState via NVLink (see correctness note above)
          ChunkState& remoteReceiverState = remoteReceiverStates[chunkStateIdx];
          remoteReceiverState.ready_to_recv(group, stepId);
        });
      }
    } else {
      // =====================================================================
      // SINGLE CHUNK STATE MODE (Original Design)
      // =====================================================================
      // Uses one ChunkState buffer per peer (simpler but more NVLink latency):
      //   - receiverStateBuffer: Both wait and signal happen here via NVLink
      //
      // STATE MACHINE (per chunk):
      //   ┌──────────────────────────────────────────────────────────────────┐
      //   │ SENDER (this side)              RECEIVER (peer side)            │
      //   ├──────────────────────────────────────────────────────────────────┤
      //   │ 1. Wait REMOTE receiverState    1. Wait LOCAL receiverState     │
      //   │    for READY_TO_SEND (-1)          for stepId value             │
      //   │    (NVLink round-trip)             (fast local poll)            │
      //   │                                                                  │
      //   │ 2. Copy data to peer buffer     2. Copy data from local buffer  │
      //   │    via NVLink                      (no NVLink needed)           │
      //   │                                                                  │
      //   │ 3. Signal peer via NVLink       3. Signal LOCAL receiverState   │
      //   │    write with stepId               with READY_TO_SEND (-1)      │
      //   └──────────────────────────────────────────────────────────────────┘
      //
      // TRADE-OFF: Simpler (no call_index tracking needed) but sender's
      // busy-wait polls remote memory via NVLink, adding latency.
      // =====================================================================

      for (std::size_t stepId = 0; stepId < totalSteps; ++stepId) {
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

        group.for_each_item_contiguous(
            numChunksThisStep, [&](uint32_t chunkIdx) {
              const std::size_t chunkOffset = chunkIdx * kChunkSize;
              const std::size_t chunkBytes =
                  (chunkOffset + kChunkSize <= stepBytes)
                  ? kChunkSize
                  : stepBytes - chunkOffset;

              if (chunkBytes == 0) {
                return;
              }

              const std::size_t chunkStateIdx = stateOffset + chunkIdx;

              // Wait on REMOTE receiverStateBuffer via NVLink (slower)
              ChunkState& remoteReceiverState =
                  remoteReceiverStates[chunkStateIdx];
              remoteReceiverState.wait_ready_to_send(group, timeout);

              // Copy data to peer's buffer via NVLink
              memcpy_vectorized(
                  sendBuffer + dataBufferOffset + chunkOffset,
                  src + stepOffset + chunkOffset,
                  chunkBytes,
                  group);

              // Signal peer's receiverStateBuffer via NVLink write
              remoteReceiverState.ready_to_recv(group, stepId);
            });
      }
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
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__
    if (options_.dataBufferSize == 0) {
      printf(
          "P2pNvlTransportDevice::recv() requires staging buffer"
          " (dataBufferSize > 0) at %s:%d\n",
          __FILE__,
          __LINE__);
      __trap();
    }
    char* dst = reinterpret_cast<char*>(dstbuff);

    char* recvBuffer = localState_.dataBuffer;
    // Local wait buffer: my state (sender writes here via NVLink)
    ChunkState* const localReceiverStates =
        localState_.receiverStateBuffer.data();

    const std::size_t totalSteps =
        (nbytes + options_.dataBufferSize - 1) / options_.dataBufferSize;
    const std::size_t kChunkSize = options_.chunkSize;
    const std::size_t chunksPerStep =
        (options_.dataBufferSize + kChunkSize - 1) / kChunkSize;

    if (options_.useDualStateBuffer) {
      // =====================================================================
      // DUAL CHUNK STATE MODE (Receiver side)
      // =====================================================================
      // See send() for detailed state machine, correctness analysis, and
      // explanation of why two group.sync() calls are required.
      //
      // Receiver steps per chunk:
      // 1. Wait LOCAL receiverState for sender's signal (ld.acquire.sys)
      // 2. Copy data from local buffer
      //    ─── group.sync() [inside unready()] ───
      // 3. Mark LOCAL receiverState as UNREADY (plain write)
      //    ─── group.sync() [inside ready_to_send()] ───
      // 4. Signal REMOTE senderState via NVLink (st.release.sys)
      //
      // STRIDED ASSIGNMENT: Uses for_each_item_strided to ensure
      // each chunk is always assigned to the same thread group. This is
      // required because unready() uses plain write + group.sync() (not
      // st.release.sys), which is only visible within the same group.
      // =====================================================================
      ChunkState* const remoteSenderStates =
          remoteState_.senderStateBuffer.data();

      for (std::size_t stepId = 0; stepId < totalSteps; stepId++) {
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

        group.for_each_item_strided(numChunksThisStep, [&](uint32_t chunkIdx) {
          const std::size_t chunkOffset = chunkIdx * kChunkSize;
          const std::size_t chunkBytes = (chunkOffset + kChunkSize <= stepBytes)
              ? kChunkSize
              : stepBytes - chunkOffset;

          if (chunkBytes == 0) {
            return;
          }

          const std::size_t chunkStateIdx = stateOffset + chunkIdx;

          // Wait on LOCAL receiverStateBuffer for sender's signal
          // (fast local poll - sender signals when data is ready)
          ChunkState& localReceiverState = localReceiverStates[chunkStateIdx];
          localReceiverState.wait_ready_to_recv(group, stepId, timeout);

          // Copy data from local buffer
          memcpy_vectorized(
              dst + stepOffset + chunkOffset,
              recvBuffer + dataBufferOffset + chunkOffset,
              chunkBytes,
              group);

          // Sync #1 + plain write: barrier all threads, then leader
          // writes UNREADY to local receiverState (see send() correctness
          // note for why two syncs are required)
          localReceiverState.unready(group);

          // Sync #2 + release store: barrier all threads (flushes the
          // UNREADY plain write), then leader does st.release.sys.global
          // READY_TO_SEND to peer's senderState via NVLink
          ChunkState& remoteSenderState = remoteSenderStates[chunkStateIdx];
          remoteSenderState.ready_to_send(group);
        });
      }
    } else {
      // =====================================================================
      // SINGLE CHUNK STATE MODE (Original Design)
      // =====================================================================
      // See send() for detailed state machine documentation.
      //
      // Receiver side:
      // 1. Wait LOCAL receiverStateBuffer for sender's signal (stepId)
      // 2. Copy data from local buffer (sender wrote via NVLink)
      // 3. Signal LOCAL receiverStateBuffer with READY_TO_SEND (-1)
      // =====================================================================

      for (std::size_t stepId = 0; stepId < totalSteps; stepId++) {
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

        group.for_each_item_contiguous(
            numChunksThisStep, [&](uint32_t chunkIdx) {
              const std::size_t chunkOffset = chunkIdx * kChunkSize;
              const std::size_t chunkBytes =
                  (chunkOffset + kChunkSize <= stepBytes)
                  ? kChunkSize
                  : stepBytes - chunkOffset;

              if (chunkBytes == 0) {
                return;
              }

              const std::size_t chunkStateIdx = stateOffset + chunkIdx;

              // Wait on LOCAL receiverStateBuffer for sender's signal (stepId)
              ChunkState& localReceiverState =
                  localReceiverStates[chunkStateIdx];
              localReceiverState.wait_ready_to_recv(group, stepId, timeout);

              // Copy data from local buffer
              memcpy_vectorized(
                  dst + stepOffset + chunkOffset,
                  recvBuffer + dataBufferOffset + chunkOffset,
                  chunkBytes,
                  group);

              // Signal LOCAL receiverStateBuffer with READY_TO_SEND
              localReceiverState.ready_to_send(group);
            });
      }
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

  __host__ __device__ size_t get_ll128_buffer_num_packets() const {
    return options_.ll128BufferNumPackets;
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
      const std::size_t chunkBytes = (chunkOffset + alignedChunkSize <= nbytes)
          ? alignedChunkSize
          : nbytes - chunkOffset;

      if (chunkBytes > 0) {
        memcpy_vectorized(
            dst_d + chunkOffset, // dst_base
            src_d + chunkOffset, // src_base
            chunkBytes, // chunk_bytes
            group);
        totalBytesWritten += chunkBytes;
      }
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
  // LL128 Protocol Operations
  // ===========================================================================

  /**
   * ll128_send — Send data to peer's LL128 buffer via NVLink.
   *
   * Packs user data into LL128 packets and volatile-stores them to the
   * peer's LL128 buffer with inline flag signaling.
   *
   * PRECONDITION: ll128BufferSize > 0 in transport config.
   *
   * @param group   ThreadGroup (auto-converted to warp scope)
   * @param src     Local source buffer (16-byte aligned)
   * @param nbytes  Total bytes (must be a multiple of 16)
   * @param timeout Timeout for flag polling
   */
  __device__ __forceinline__ void ll128_send(
      const ThreadGroup& group,
      const char* src,
      size_t nbytes,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__
    PIPES_DEVICE_CHECK(remoteState_.ll128Buffer != nullptr);
    PIPES_DEVICE_CHECK(can_use_ll128(src, nbytes));

    comms::pipes::ll128_send(
        group,
        src,
        nbytes,
        remoteState_.ll128Buffer,
        timeout,
        options_.ll128BufferNumPackets);
#endif
  }

  /**
   * ll128_recv — Receive data from local LL128 buffer.
   *
   * Polls the local LL128 buffer (written remotely by peer), reads
   * payload to output buffer, and ACKs with READY_TO_WRITE.
   *
   * PRECONDITION: ll128BufferSize > 0 in transport config.
   *
   * @param group   ThreadGroup (auto-converted to warp scope)
   * @param dst     Local output buffer (16-byte aligned)
   * @param nbytes  Total bytes (must be a multiple of 16)
   * @param timeout Timeout for flag polling
   */
  __device__ __forceinline__ void ll128_recv(
      const ThreadGroup& group,
      char* dst,
      size_t nbytes,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__
    PIPES_DEVICE_CHECK(localState_.ll128Buffer != nullptr);
    PIPES_DEVICE_CHECK(can_use_ll128(dst, nbytes));

    comms::pipes::ll128_recv(
        group,
        dst,
        nbytes,
        localState_.ll128Buffer,
        timeout,
        options_.ll128BufferNumPackets);
#endif
  }

  /**
   * ll128_forward — Receive from predecessor and forward to successor.
   *
   * Reads from this transport's local LL128 buffer (predecessor wrote here),
   * forwards to successor_transport's remote LL128 buffer, copies payload
   * to local output, and ACKs predecessor.
   *
   * PRECONDITION: ll128BufferSize > 0 in both this and successor transport.
   *
   * @param group                ThreadGroup (auto-converted to warp scope)
   * @param dst                  Local output buffer (16-byte aligned)
   * @param nbytes               Total bytes (must be a multiple of 16)
   * @param successor_transport  Transport for the successor peer
   * @param timeout              Timeout for flag polling
   */
  __device__ __forceinline__ void ll128_forward(
      const ThreadGroup& group,
      char* dst,
      size_t nbytes,
      const P2pNvlTransportDevice& successor_transport,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__
    PIPES_DEVICE_CHECK(localState_.ll128Buffer != nullptr);
    PIPES_DEVICE_CHECK(successor_transport.remoteState_.ll128Buffer != nullptr);
    PIPES_DEVICE_CHECK(can_use_ll128(dst, nbytes));

    // Use the minimum packet count of local and successor buffers.
    // 0 means uncapped (legacy path where buffer is pre-sized to fit).
    const size_t my_packets = options_.ll128BufferNumPackets;
    const size_t succ_packets =
        successor_transport.options_.ll128BufferNumPackets;
    size_t effective_packets = 0;
    if (my_packets > 0 && succ_packets > 0) {
      effective_packets =
          (my_packets < succ_packets) ? my_packets : succ_packets;
    } else if (my_packets > 0) {
      effective_packets = my_packets;
    } else {
      effective_packets = succ_packets;
    }

    comms::pipes::ll128_forward(
        group,
        dst,
        nbytes,
        localState_.ll128Buffer,
        successor_transport.remoteState_.ll128Buffer,
        timeout,
        effective_packets);
#endif
  }

 private:
  const int myRank_{-1};
  const int peerRank_{-1};
  const P2pNvlTransportOptions options_;
  LocalState localState_;
  RemoteState remoteState_;
};

} // namespace comms::pipes
