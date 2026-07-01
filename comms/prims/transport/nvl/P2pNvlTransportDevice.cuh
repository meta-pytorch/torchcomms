// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <cuda.h>

#include <cuda_runtime.h>
#include <cstddef>
#include <cstring>
#include "comms/prims/core/BarrierState.cuh"
#include "comms/prims/core/ChunkState.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/DeviceMacros.cuh"
#include "comms/prims/core/MemcpyCopyOp.cuh"
#include "comms/prims/core/SignalState.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/amd/HipHostCompat.h"
#include "comms/prims/transport/ll/LlOps.cuh"
#include "comms/prims/transport/ll128/Ll128Ops.cuh"
#include "comms/prims/transport/nvl/NvlChannelState.cuh"

namespace comms::prims {

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
  LlLine* llBuffer{nullptr};
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
  LlLine* llBuffer{nullptr};
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
  std::size_t dataBufferSize{0};
  std::size_t chunkSize{0};
  std::size_t pipelineDepth{0};
  bool useDualStateBuffer{false}; // Default to single state buffer mode
  std::size_t ll128BufferNumPackets{0}; // 0 = no chunking
  std::size_t llBufferNumLines{0}; // 0 = no chunking

  // ---- Tile (per-channel) protocol fields. Populated by the host transport
  // from MultiPeerNvlTransportConfig. Used by send/recv/forward (the tile
  // path).
  //
  // Slot-major staging layout: within each pipeline slot (size =
  // dataBufferSize) each channel owns a fixed slice of size per_channel_slot.
  // Channel c at pipeline slot s reads/writes
  //   staging_base + s * dataBufferSize + c * per_channel_slot
  //
  //   per_channel_slot = MultiPeerNvlTransportConfig.perChannelSize
  //   dataBufferSize = max_num_channels * per_channel_slot
  //   max_num_channels = MultiPeerNvlTransportConfig.max_num_channels
  //
  // max_num_channels must equal the array length of the per-peer
  // NvlChannelState arrays passed into the device transport.
  std::size_t per_channel_slot{0};
  int max_num_channels{0};
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
 * { auto group = make_warp_group(); p2p.send_group(group, src, n);  // Writes
 * to GPU B's buffers via NVLink
 *   }
 *
 *   // Kernel (receiver on GPU B)
 *   __global__ void recvKernel(P2pNvlTransportDevice p2p, void* dst, size_t n)
 * { auto group = make_warp_group(); p2p.recv_group(group, dst, n);  // Reads
 * from own local buffers
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
      const RemoteState& remoteState,
      NvlChannelState* localChannels = nullptr,
      NvlChannelState* remoteChannels = nullptr)
      : myRank_(myRank),
        peerRank_(peerRank),
        options_(options),
        localState_(localState),
        remoteState_(remoteState),
        local_channels_(localChannels),
        remote_channels_(remoteChannels) {}

  __host__ __device__ ~P2pNvlTransportDevice() = default;

  // Per-block pipeline window in bytes. Returns the largest call size that
  // can be pipelined within the staging ring without triggering backpressure
  // *under the assumption* that the caller's totalGroups blocks share the
  // pipeline-slot equally — i.e. the historical pre-channels arithmetic.
  // Kept unchanged in D1 so existing callers see the same window size as
  // before. With the new channel design, calls larger than
  // (per_channel_slot * safeDepth) will internally wrap the pipeline ring
  // and pay backpressure-wait cost; revisit this default in a follow-up.
  __host__ __device__ std::size_t pipeline_window(int totalGroups) const {
    const std::size_t perBlockSlotSize =
        (options_.dataBufferSize / totalGroups) & ~15ULL;
    const std::size_t safeDepth =
        options_.pipelineDepth > 1 ? options_.pipelineDepth - 1 : 1;
    return perBlockSlotSize * safeDepth;
  }

  /**
   * send_group - Cooperative transfer to peer GPU over NVLink
   *
   * Sends 'nbytes' bytes from srcbuff to the peer GPU using pipelined staged
   * transfer with fine-grained chunk-level synchronization. Multiple groups
   * collaborate to transfer the data in parallel — work is distributed across
   * all calling groups via for_each_item_contiguous/strided.
   *
   * All calling groups must pass the same src/nbytes. Unlike send(),
   * which has each group independently send its own partition of data, this
   * version has all groups cooperate on the entire buffer.
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
  __device__ __forceinline__ void send_group(
      ThreadGroup& group,
      void* srcbuff,
      std::size_t nbytes,
      const Timeout& timeout = Timeout()) {
#if PIPES_IS_DEVICE_COMPILE
    if (options_.dataBufferSize == 0) {
      printf(
          "P2pNvlTransportDevice::send_group() requires staging buffer"
          " (dataBufferSize > 0) at %s:%d\n",
          __FILE__,
          __LINE__);
      PIPES_DEVICE_TRAP();
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
   * recv_group - Receive data from peer GPU over NVLink
   *
   * Receives 'nbytes' bytes into dstbuff from the peer GPU's send_group()
   * call. Must be called simultaneously with peer's send_group() for the same
   * byte count.
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
  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void recv_group(
      ThreadGroup& group,
      void* dstbuff,
      std::size_t nbytes,
      [[maybe_unused]] const Timeout& timeout = Timeout(),
      [[maybe_unused]] Args... args) {
#if PIPES_IS_DEVICE_COMPILE
    if (options_.dataBufferSize == 0) {
      printf(
          "P2pNvlTransportDevice::recv_group() requires staging buffer"
          " (dataBufferSize > 0) at %s:%d\n",
          __FILE__,
          __LINE__);
      PIPES_DEVICE_TRAP();
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
      // See send_group() for detailed state machine, correctness analysis, and
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

          CopyOp::recv(
              dst + stepOffset + chunkOffset,
              recvBuffer + dataBufferOffset + chunkOffset,
              chunkBytes,
              group,
              stepOffset + chunkOffset,
              args...);

          // Sync #1 + plain write: barrier all threads, then leader
          // writes UNREADY to local receiverState (see send_group()
          // correctness note for why two syncs are required)
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
      // See send_group() for detailed state machine documentation.
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

              CopyOp::recv(
                  dst + stepOffset + chunkOffset,
                  recvBuffer + dataBufferOffset + chunkOffset,
                  chunkBytes,
                  group,
                  stepOffset + chunkOffset,
                  args...);

              // Signal LOCAL receiverStateBuffer with READY_TO_SEND
              localReceiverState.ready_to_send(group);
            });
      }
    }
#endif
  }

  /**
   * forward_group - Fused receive-and-forward
   *
   * Reads data from this transport's predecessor staging buffer and writes
   * to two destinations simultaneously: the local user buffer (dstbuff) and
   * the successor's remote staging buffer. This halves the read bandwidth
   * vs sequential recv_group + send_group.
   *
   * PRECONDITIONS:
   * - `this` transport is connected to the predecessor (data arrives in
   *   this->localState_.dataBuffer)
   * - `successor` transport is connected to the next rank (data forwarded
   *   to successor.remoteState_.dataBuffer)
   * - Both transports must have matching options (dataBufferSize, chunkSize,
   *   pipelineDepth, useDualStateBuffer)
   *
   * @param group ThreadGroup for cooperative processing
   * @param dstbuff Local user buffer to copy data into
   * @param nbytes Number of bytes to forward
   * @param successor Transport to the next rank in the ring
   * @param timeout Timeout for polling operations
   */
  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void forward_group(
      ThreadGroup& group,
      void* dstbuff,
      std::size_t nbytes,
      P2pNvlTransportDevice& successor,
      [[maybe_unused]] const Timeout& timeout = Timeout(),
      [[maybe_unused]] Args... args) {
#if PIPES_IS_DEVICE_COMPILE
    if (options_.dataBufferSize == 0) {
      printf(
          "P2pNvlTransportDevice::forward_group() requires staging buffer"
          " (dataBufferSize > 0) at %s:%d\n",
          __FILE__,
          __LINE__);
      PIPES_DEVICE_TRAP();
    }
    char* dst = reinterpret_cast<char*>(dstbuff);

    // Predecessor's staging buffer (local read)
    char* recvBuffer = localState_.dataBuffer;
    ChunkState* const localReceiverStates =
        localState_.receiverStateBuffer.data();

    // Successor's staging buffer (NVLink write)
    char* successorSendBuffer = successor.remoteState_.dataBuffer;
    ChunkState* const successorRemoteReceiverStates =
        successor.remoteState_.receiverStateBuffer.data();

    const std::size_t totalSteps =
        (nbytes + options_.dataBufferSize - 1) / options_.dataBufferSize;
    const std::size_t kChunkSize = options_.chunkSize;
    const std::size_t chunksPerStep =
        (options_.dataBufferSize + kChunkSize - 1) / kChunkSize;

    if (options_.useDualStateBuffer) {
      // =================================================================
      // DUAL CHUNK STATE MODE (forward = fused recv + send)
      // =================================================================
      // Per chunk, we need four signals:
      // 1. localReceiverState.unready()        — prevents predecessor re-send
      // 2. successorLocalSenderState.unready()  — prevents re-forward
      // 3. remoteSenderState.ready_to_send()    — tells predecessor buf free
      // 4. successorRemoteReceiverState.ready_to_recv() — tells successor
      //    data ready
      //
      // Ordering: do both unready() calls first (each has group.sync() +
      // plain write), then both release-store signals (each has
      // group.sync() + st.release.sys.global).
      // =================================================================
      ChunkState* const remoteSenderStates =
          remoteState_.senderStateBuffer.data();
      ChunkState* const successorLocalSenderStates =
          successor.localState_.senderStateBuffer.data();

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

          // 1. Wait for predecessor data to be ready
          ChunkState& localReceiverState = localReceiverStates[chunkStateIdx];
          localReceiverState.wait_ready_to_recv(group, stepId, timeout);

          // 2. Wait for successor's staging buffer to be free
          ChunkState& successorLocalSenderState =
              successorLocalSenderStates[chunkStateIdx];
          successorLocalSenderState.wait_ready_to_send(group, timeout);

          CopyOp::forward(
              dst ? dst + stepOffset + chunkOffset : nullptr,
              successorSendBuffer + dataBufferOffset + chunkOffset,
              recvBuffer + dataBufferOffset + chunkOffset,
              chunkBytes,
              group,
              stepOffset + chunkOffset,
              args...);

          // 4. Both unready() calls (plain writes with group.sync())
          localReceiverState.unready(group);
          successorLocalSenderState.unready(group);

          // 5. Both release-store signals (NVLink writes)
          ChunkState& remoteSenderState = remoteSenderStates[chunkStateIdx];
          remoteSenderState.ready_to_send(group);

          ChunkState& successorRemoteReceiverState =
              successorRemoteReceiverStates[chunkStateIdx];
          successorRemoteReceiverState.ready_to_recv(group, stepId);
        });
      }
    } else {
      // =================================================================
      // SINGLE CHUNK STATE MODE (fallback)
      // =================================================================
      ChunkState* const successorRemoteReceiverStatesOnly =
          successor.remoteState_.receiverStateBuffer.data();

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

              // Wait for predecessor data
              ChunkState& localReceiverState =
                  localReceiverStates[chunkStateIdx];
              localReceiverState.wait_ready_to_recv(group, stepId, timeout);

              // Wait for successor staging to be free (NVLink poll)
              ChunkState& successorRemoteReceiverState =
                  successorRemoteReceiverStatesOnly[chunkStateIdx];
              successorRemoteReceiverState.wait_ready_to_send(group, timeout);

              CopyOp::forward(
                  dst ? dst + stepOffset + chunkOffset : nullptr,
                  successorSendBuffer + dataBufferOffset + chunkOffset,
                  recvBuffer + dataBufferOffset + chunkOffset,
                  chunkBytes,
                  group,
                  stepOffset + chunkOffset,
                  args...);

              // ACK predecessor (buffer free)
              localReceiverState.ready_to_send(group);

              // Signal successor (data ready)
              successorRemoteReceiverState.ready_to_recv(group, stepId);
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
   * put_group - Cooperative local memory copy using vectorized operations
   *
   * Performs a high-performance vectorized copy from src_d to dst_d.
   * Multiple groups collaborate on the same src/dst/nbytes — work is
   * distributed across all calling groups via for_each_item_contiguous
   * by global group_id.
   *
   * All calling groups must pass the same src/dst/nbytes. Unlike put(),
   * which has each group independently copy its own partition of data, this
   * version has all groups cooperate on the entire buffer.
   *
   * Contrast with send_group(): send_group() writes to the peer GPU's staging
   * buffer via NVLink with pipelined flow control. put_group() copies within
   * local memory without any signaling or flow control.
   *
   * @param group ThreadGroup for cooperative processing
   * @param dst_d Destination pointer (device memory)
   * @param src_d Source pointer (device memory)
   * @param nbytes Number of bytes to copy
   *
   * @return Number of bytes written by the current thread group
   */
  __device__ __forceinline__ std::size_t put_group(
      [[maybe_unused]] ThreadGroup& group,
      [[maybe_unused]] char* dst_d,
      [[maybe_unused]] const char* src_d,
      [[maybe_unused]] std::size_t nbytes) {
#if PIPES_IS_DEVICE_COMPILE
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
   * put - Independent per-group local memory copy
   *
   * Performs a vectorized copy from src_d to dst_d using only threads within
   * the calling group. Each group operates independently on its own data,
   * so different groups can call put() with different src/dst/nbytes.
   *
   * Unlike put_group(), which has all groups cooperate on the same buffer,
   * put() has each group work on its own partition independently.
   *
   * Contrast with send(): send() writes to the peer GPU's staging
   * buffer via NVLink with pipelined flow control and signaling. put()
   * copies within local memory without any signaling or flow control.
   *
   * @param group ThreadGroup for cooperative processing (group-local)
   * @param dst_d Destination pointer (device memory)
   * @param src_d Source pointer (device memory)
   * @param nbytes Number of bytes to copy
   */
  __device__ __forceinline__ void
  put(ThreadGroup& group, char* dst_d, const char* src_d, std::size_t nbytes) {
#if PIPES_IS_DEVICE_COMPILE
    if (nbytes == 0 || dst_d == src_d) {
      return;
    }
    assert_buffer_non_overlap(dst_d, src_d, nbytes, group);
    memcpy_vectorized(dst_d, src_d, nbytes, group);
#endif
  }

  /**
   * signal - Signal peer GPU via NVLink
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
  __device__ __forceinline__ void
  signal(ThreadGroup& group, uint64_t signal_id, SignalOp op, uint64_t value) {
    remoteState_.signalBuffer[signal_id].signal(group, op, value);
  }

  /**
   * wait_signal_until - Wait for signal from peer GPU
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
  __device__ __forceinline__ void wait_signal_until(
      ThreadGroup& group,
      uint64_t signal_id,
      CmpOp op,
      uint64_t value,
      const Timeout& timeout = Timeout()) {
    localState_.signalBuffer[signal_id].wait_until(group, op, value, timeout);
  }

  /**
   * reset_signal - Reset a local signal slot to zero
   *
   * Resets the local signal counter at the specified index to zero.
   * This is safe to call from the receiver side after processing the signal,
   * since the receiver owns the local inbox buffer.
   *
   * The caller must ensure the signal has already been consumed (waited on)
   * before resetting, and that no peer is concurrently signaling the same slot.
   *
   * @param group ThreadGroup for cooperative thread synchronization
   * @param signal_id Index into the signalBuffer array
   */
  __device__ __forceinline__ void reset_signal(
      ThreadGroup& group,
      uint64_t signal_id) {
    if (group.is_leader()) {
      localState_.signalBuffer[signal_id].store(0);
    }
    group.sync();
  }

  /**
   * barrier_sync - Two-sided barrier synchronization with peer GPU
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
  __device__ __forceinline__ void barrier_sync(
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
   * ll128_send_group — Send data to peer's LL128 buffer via NVLink.
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
  __device__ __forceinline__ void ll128_send_group(
      const ThreadGroup& group,
      const char* src,
      size_t nbytes,
      const Timeout& timeout = Timeout()) {
#if PIPES_IS_DEVICE_COMPILE
    PIPES_DEVICE_CHECK(remoteState_.ll128Buffer != nullptr);
    PIPES_DEVICE_CHECK(can_use_ll128(src, nbytes));

    comms::prims::ll128_send(
        group,
        src,
        nbytes,
        remoteState_.ll128Buffer,
        timeout,
        options_.ll128BufferNumPackets);
#endif
  }

  /**
   * ll128_recv_group — Receive data from local LL128 buffer.
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
  __device__ __forceinline__ void ll128_recv_group(
      const ThreadGroup& group,
      char* dst,
      size_t nbytes,
      const Timeout& timeout = Timeout()) {
#if PIPES_IS_DEVICE_COMPILE
    PIPES_DEVICE_CHECK(localState_.ll128Buffer != nullptr);
    PIPES_DEVICE_CHECK(can_use_ll128(dst, nbytes));

    comms::prims::ll128_recv(
        group,
        dst,
        nbytes,
        localState_.ll128Buffer,
        timeout,
        options_.ll128BufferNumPackets);
#endif
  }

  /**
   * ll128_forward_group — Receive from predecessor and forward to successor.
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
  __device__ __forceinline__ void ll128_forward_group(
      const ThreadGroup& group,
      char* dst,
      size_t nbytes,
      const P2pNvlTransportDevice& successor_transport,
      const Timeout& timeout = Timeout()) {
#if PIPES_IS_DEVICE_COMPILE
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

    comms::prims::ll128_forward(
        group,
        dst,
        nbytes,
        localState_.ll128Buffer,
        successor_transport.remoteState_.ll128Buffer,
        timeout,
        effective_packets);
#endif
  }

  /**
   * send - Independent per-group transfer to peer GPU over NVLink
   *
   * Each group independently sends its own tile of data to the peer GPU's
   * staging buffer via NVLink, with per-group pipelined flow control and
   * signaling. Different groups can call send() with different
   * src/nbytes.
   *
   * Unlike send_group(), which has all groups cooperate on the same buffer,
   * send() has each group work on its own partition independently.
   *
   * @param max_signal_bytes Hint for max bytes between DATA_READY signals.
   *   0 means one signal per slot fill. Capped at per_channel_slot.
   */
  __device__ __forceinline__ void send(
      ThreadGroup& group,
      const void* __restrict__ src,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      const Timeout& timeout = Timeout()) {
#if PIPES_IS_DEVICE_COMPILE
    if (nbytes == 0) {
      return;
    }

    const int max_channels = options_.max_num_channels;
    const uint32_t channel = group.group_id;

    if (group.total_groups > static_cast<uint32_t>(max_channels)) {
      printf(
          "send: group.total_groups=%u > max_num_channels=%d. "
          "Channel arrays would be accessed out of bounds.\n",
          group.total_groups,
          max_channels);
      PIPES_DEVICE_TRAP();
    }

    const std::size_t slotSize = options_.dataBufferSize;
    const std::size_t perChannelSlot = options_.per_channel_slot;
    if (perChannelSlot == 0) {
      printf(
          "send: per_channel_slot is 0 (dataBufferSize=%llu, "
          "max_num_channels=%d). Set perChannelSize when channels are "
          "enabled.\n",
          (unsigned long long)slotSize,
          max_channels);
      PIPES_DEVICE_TRAP();
    }
    const std::size_t stagingOff = channel * perChannelSlot;

    const std::size_t chunkSize =
        max_signal_bytes > 0 && max_signal_bytes < perChannelSlot
        ? (max_signal_bytes & ~15ULL)
        : perChannelSlot;
    const std::size_t effectiveChunk =
        chunkSize > 0 ? chunkSize : perChannelSlot;

    const std::size_t pipelineBytes = perChannelSlot * options_.pipelineDepth;

    NvlChannelState& local_ch = local_channels_[channel];
    NvlChannelState& remote_ch = remote_channels_[channel];

    const char* __restrict__ srcPtr = reinterpret_cast<const char*>(src);
    char* __restrict__ stagBuf = remoteState_.dataBuffer;

    const uint64_t baseByte = static_cast<uint64_t>(local_ch.send_cursor);

    const std::size_t protocolBytes = align_tile_protocol_bytes(nbytes);
    for (std::size_t dataOff = 0; dataOff < protocolBytes;) {
      const uint64_t streamStart = baseByte + dataOff;
      const std::size_t pipelineOff =
          static_cast<std::size_t>(streamStart % pipelineBytes);
      const std::size_t slot = pipelineOff / perChannelSlot;
      const std::size_t slotOff = slot * slotSize;
      const std::size_t chunkOff = pipelineOff - slot * perChannelSlot;
      const std::size_t slotRemaining = perChannelSlot - chunkOff;
      const std::size_t dataRemaining = protocolBytes - dataOff;
      std::size_t copyBytes =
          effectiveChunk < dataRemaining ? effectiveChunk : dataRemaining;
      copyBytes = copyBytes < slotRemaining ? copyBytes : slotRemaining;
      const uint64_t streamEnd = streamStart + copyBytes;

      if (streamEnd > pipelineBytes) {
        local_ch.slot_free.wait_until(
            group, CmpOp::CMP_GE, streamEnd - pipelineBytes, timeout);
      }

      const std::size_t validBytes =
          valid_payload_bytes(dataOff, copyBytes, nbytes);
      if (validBytes > 0) {
        memcpy_vectorized(
            stagBuf + slotOff + stagingOff + chunkOff,
            srcPtr + dataOff,
            validBytes,
            group);
      }

      group.sync();
      if (group.is_leader()) {
        remote_ch.data_ready.signal(SignalOp::SIGNAL_SET, streamEnd);
      }
      dataOff += copyBytes;
    }

    if (group.is_leader()) {
      local_ch.send_cursor = static_cast<int64_t>(baseByte + protocolBytes);
    }
    group.sync();
#endif
  }

  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void recv(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      std::size_t max_signal_bytes = 0,
      [[maybe_unused]] const Timeout& timeout = Timeout(),
      [[maybe_unused]] Args... args) {
#if PIPES_IS_DEVICE_COMPILE
    if (nbytes == 0) {
      return;
    }

    const int max_channels = options_.max_num_channels;
    const uint32_t channel = group.group_id;

    if (group.total_groups > static_cast<uint32_t>(max_channels)) {
      printf(
          "recv: group.total_groups=%u > max_num_channels=%d. "
          "Channel arrays would be accessed out of bounds.\n",
          group.total_groups,
          max_channels);
      PIPES_DEVICE_TRAP();
    }

    const std::size_t slotSize = options_.dataBufferSize;
    const std::size_t perChannelSlot = options_.per_channel_slot;
    if (perChannelSlot == 0) {
      printf(
          "recv: per_channel_slot is 0 (dataBufferSize=%llu, "
          "max_num_channels=%d). Set perChannelSize when channels are "
          "enabled.\n",
          (unsigned long long)slotSize,
          max_channels);
      PIPES_DEVICE_TRAP();
    }
    const std::size_t stagingOff = channel * perChannelSlot;

    const std::size_t chunkSize =
        max_signal_bytes > 0 && max_signal_bytes < perChannelSlot
        ? (max_signal_bytes & ~15ULL)
        : perChannelSlot;
    const std::size_t effectiveChunk =
        chunkSize > 0 ? chunkSize : perChannelSlot;

    const std::size_t pipelineBytes = perChannelSlot * options_.pipelineDepth;

    NvlChannelState& local_ch = local_channels_[channel];
    NvlChannelState& remote_ch = remote_channels_[channel];

    char* __restrict__ dstPtr = reinterpret_cast<char*>(dst);
    char* __restrict__ stagBuf = localState_.dataBuffer;

    const uint64_t baseByte = static_cast<uint64_t>(local_ch.recv_cursor);

    const std::size_t protocolBytes = align_tile_protocol_bytes(nbytes);
    for (std::size_t dataOff = 0; dataOff < protocolBytes;) {
      const uint64_t streamStart = baseByte + dataOff;
      const std::size_t pipelineOff =
          static_cast<std::size_t>(streamStart % pipelineBytes);
      const std::size_t slot = pipelineOff / perChannelSlot;
      const std::size_t slotOff = slot * slotSize;
      const std::size_t chunkOff = pipelineOff - slot * perChannelSlot;
      const std::size_t slotRemaining = perChannelSlot - chunkOff;
      const std::size_t dataRemaining = protocolBytes - dataOff;
      std::size_t copyBytes =
          effectiveChunk < dataRemaining ? effectiveChunk : dataRemaining;
      copyBytes = copyBytes < slotRemaining ? copyBytes : slotRemaining;
      const uint64_t streamEnd = streamStart + copyBytes;

      local_ch.data_ready.wait_until(group, CmpOp::CMP_GE, streamEnd, timeout);

      const std::size_t validBytes =
          valid_payload_bytes(dataOff, copyBytes, nbytes);
      if (validBytes > 0) {
        CopyOp::recv(
            dstPtr + dataOff,
            stagBuf + slotOff + stagingOff + chunkOff,
            validBytes,
            group,
            dataOff,
            args...);
      }

      group.sync();
      if (group.is_leader()) {
        if (chunkOff + copyBytes == perChannelSlot ||
            dataOff + copyBytes == protocolBytes) {
          remote_ch.slot_free.signal(SignalOp::SIGNAL_SET, streamEnd);
        }
      }
      dataOff += copyBytes;
    }

    if (group.is_leader()) {
      local_ch.recv_cursor = static_cast<int64_t>(baseByte + protocolBytes);
    }
    group.sync();
#endif
  }

  /**
   * forward - Independent per-channel fused receive-and-forward (tile-style)
   *
   * Each group reads its own channel from this transport's predecessor staging
   * buffer and writes to two destinations simultaneously: the local user
   * buffer (dst) and the successor's remote staging buffer. Halves read
   * bandwidth vs sequential recv + send.
   *
   * PRECONDITIONS:
   * - `this` transport is connected to the predecessor (data arrives in
   *   this->localState_.dataBuffer)
   * - `successor` transport is connected to the next rank (data forwarded
   *   to successor.remoteState_.dataBuffer)
   * - Both transports must have matching options (dataBufferSize,
   *   per_channel_slot, max_num_channels, pipelineDepth).
   *
   * @param group ThreadGroup for cooperative processing (group-local)
   * @param dst Local user buffer to copy data into
   * @param nbytes Number of bytes to forward
   * @param successor Transport to the next rank in the ring
   * @param max_signal_bytes Hint for max bytes between signals.
   *   0 means one signal per slot fill. Capped at per_channel_slot.
   */
  template <typename CopyOp = Memcpy, typename... Args>
  __device__ __forceinline__ void forward(
      ThreadGroup& group,
      void* __restrict__ dst,
      std::size_t nbytes,
      P2pNvlTransportDevice& successor,
      std::size_t max_signal_bytes = 0,
      [[maybe_unused]] const Timeout& timeout = Timeout(),
      [[maybe_unused]] Args... args) {
#if PIPES_IS_DEVICE_COMPILE
    if (nbytes == 0) {
      return;
    }

    const int max_channels = options_.max_num_channels;
    const uint32_t channel = group.group_id;

    if (group.total_groups > static_cast<uint32_t>(max_channels)) {
      printf(
          "forward: group.total_groups=%u > max_num_channels=%d. "
          "Channel arrays would be accessed out of bounds.\n",
          group.total_groups,
          max_channels);
      PIPES_DEVICE_TRAP();
    }

    const std::size_t slotSize = options_.dataBufferSize;
    const std::size_t perChannelSlot = options_.per_channel_slot;
    if (perChannelSlot == 0) {
      printf(
          "forward: per_channel_slot is 0 (dataBufferSize=%llu, "
          "max_num_channels=%d). Set perChannelSize when channels are "
          "enabled.\n",
          (unsigned long long)slotSize,
          max_channels);
      PIPES_DEVICE_TRAP();
    }
    const std::size_t stagingOff = channel * perChannelSlot;

    const std::size_t chunkSize =
        max_signal_bytes > 0 && max_signal_bytes < perChannelSlot
        ? (max_signal_bytes & ~15ULL)
        : perChannelSlot;
    const std::size_t effectiveChunk =
        chunkSize > 0 ? chunkSize : perChannelSlot;

    const std::size_t pipelineBytes = perChannelSlot * options_.pipelineDepth;

    // Recv side: this transport (predecessor → me).
    NvlChannelState& recv_local_ch = local_channels_[channel];
    NvlChannelState& recv_remote_ch = remote_channels_[channel];
    // Send side: successor transport (me → successor).
    NvlChannelState& send_local_ch = successor.local_channels_[channel];
    NvlChannelState& send_remote_ch = successor.remote_channels_[channel];

    char* __restrict__ dstPtr = reinterpret_cast<char*>(dst);
    // Predecessor's staging buffer (local read)
    char* __restrict__ recvBuf = localState_.dataBuffer;
    // Successor's staging buffer (NVLink write)
    char* __restrict__ sendBuf = successor.remoteState_.dataBuffer;

    const uint64_t recvBaseByte =
        static_cast<uint64_t>(recv_local_ch.recv_cursor);
    const uint64_t sendBaseByte =
        static_cast<uint64_t>(send_local_ch.send_cursor);

    const std::size_t protocolBytes = align_tile_protocol_bytes(nbytes);
    for (std::size_t dataOff = 0; dataOff < protocolBytes;) {
      const uint64_t recvStreamStart = recvBaseByte + dataOff;
      const std::size_t recvPipelineOff =
          static_cast<std::size_t>(recvStreamStart % pipelineBytes);
      const std::size_t recvSlot = recvPipelineOff / perChannelSlot;
      const std::size_t recvSlotOff = recvSlot * slotSize;
      const std::size_t recvChunkOff =
          recvPipelineOff - recvSlot * perChannelSlot;
      const uint64_t sendStreamStart = sendBaseByte + dataOff;
      const std::size_t sendPipelineOff =
          static_cast<std::size_t>(sendStreamStart % pipelineBytes);
      const std::size_t sendSlot = sendPipelineOff / perChannelSlot;
      const std::size_t sendSlotOff = sendSlot * slotSize;
      const std::size_t sendChunkOff =
          sendPipelineOff - sendSlot * perChannelSlot;
      const std::size_t recvSlotRemaining = perChannelSlot - recvChunkOff;
      const std::size_t sendSlotRemaining = perChannelSlot - sendChunkOff;
      const std::size_t dataRemaining = protocolBytes - dataOff;
      std::size_t copyBytes =
          effectiveChunk < dataRemaining ? effectiveChunk : dataRemaining;
      copyBytes = copyBytes < recvSlotRemaining ? copyBytes : recvSlotRemaining;
      copyBytes = copyBytes < sendSlotRemaining ? copyBytes : sendSlotRemaining;
      const uint64_t recvStreamEnd = recvStreamStart + copyBytes;
      const uint64_t sendStreamEnd = sendStreamStart + copyBytes;

      // 1. Wait for predecessor data to be ready (recv side, every step).
      recv_local_ch.data_ready.wait_until(
          group, CmpOp::CMP_GE, recvStreamEnd, timeout);

      // 2. Wait for successor's staging slot to be free once we have wrapped
      //    around the pipeline.
      if (sendStreamEnd > pipelineBytes) {
        send_local_ch.slot_free.wait_until(
            group, CmpOp::CMP_GE, sendStreamEnd - pipelineBytes, timeout);
      }

      // 3. Dual-dst copy: predecessor staging → local user buf +
      //    successor remote staging
      const std::size_t validBytes =
          valid_payload_bytes(dataOff, copyBytes, nbytes);
      if (validBytes > 0) {
        CopyOp::forward(
            dstPtr ? dstPtr + dataOff : nullptr,
            sendBuf + sendSlotOff + stagingOff + sendChunkOff,
            recvBuf + recvSlotOff + stagingOff + recvChunkOff,
            validBytes,
            group,
            dataOff,
            args...);
      }

      group.sync();
      if (group.is_leader()) {
        // 4. Signal successor that data is ready (send semantic: every step).
        send_remote_ch.data_ready.signal(SignalOp::SIGNAL_SET, sendStreamEnd);

        // 5. ACK predecessor that buffer is free (recv semantic: only at
        //    slot boundaries).
        if (recvChunkOff + copyBytes == perChannelSlot ||
            dataOff + copyBytes == protocolBytes) {
          recv_remote_ch.slot_free.signal(SignalOp::SIGNAL_SET, recvStreamEnd);
        }
      }
      dataOff += copyBytes;
    }

    if (group.is_leader()) {
      recv_local_ch.recv_cursor =
          static_cast<int64_t>(recvBaseByte + protocolBytes);
      send_local_ch.send_cursor =
          static_cast<int64_t>(sendBaseByte + protocolBytes);
    }
    group.sync();
#endif
  }

  // Test-only accessors for poking at channel state.
  __host__ __device__ NvlChannelState& local_channel_at(int channel) {
    return local_channels_[channel];
  }
  __host__ __device__ NvlChannelState& remote_channel_at(int channel) {
    return remote_channels_[channel];
  }

  // Device accessors for 2D tile kernel (inlined pipeline)
  __host__ __device__ const P2pNvlTransportOptions& options() const {
    return options_;
  }
  __device__ LocalState& local_state() {
    return localState_;
  }
  __device__ RemoteState& remote_state() {
    return remoteState_;
  }

  // ===========================================================================
  // LL Protocol Operations
  // ===========================================================================

  /**
   * ll_send — Send data to peer's LL buffer via NVLink.
   *
   * Packs user data into LL lines and volatile-stores them to the
   * peer's LL buffer with inline flag signaling.
   *
   * PRECONDITION: llBufferSize > 0 in transport config.
   *
   * @param group         ThreadGroup (auto-converted to warp scope)
   * @param src           Local source buffer (8-byte aligned)
   * @param nbytes        Total bytes (must be a multiple of 8)
   * @param active_groups Number of groups calling concurrently.
   *   0 = default to max groups the LL buffer can support.
   *   >0 = explicit group count; buffer partitioned per group.group_id.
   * @param timeout       Timeout for flag polling
   */
  __device__ __forceinline__ void ll_send(
      const ThreadGroup& group,
      const char* src,
      size_t nbytes,
      int active_groups = 0,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__ // NVIDIA-only: depends on ll_send/ll_recv/ll128_* not yet
                     // ported to AMD
    PIPES_DEVICE_CHECK(remoteState_.llBuffer != nullptr);
    PIPES_DEVICE_CHECK(can_use_ll(src, nbytes, options_.llBufferNumLines));

    const int maxGroups =
        (options_.llBufferNumLines >= static_cast<size_t>(kLlLinesPerWarp))
        ? static_cast<int>(options_.llBufferNumLines / kLlLinesPerWarp)
        : 1;
    const int effActive = active_groups > 0 ? active_groups : maxGroups;

    PIPES_DEVICE_CHECK(static_cast<uint32_t>(effActive) <= group.total_groups);
    PIPES_DEVICE_CHECK(group.group_id < static_cast<uint32_t>(effActive));

    const size_t perGroupLines = options_.llBufferNumLines / effActive;
    if (effActive > 1 && options_.llBufferNumLines > 0) {
      PIPES_DEVICE_CHECK(perGroupLines >= kLlLinesPerWarp);
    }
    const size_t bufferOffset = group.group_id * perGroupLines;

    comms::prims::ll_send(
        group,
        src,
        nbytes,
        remoteState_.llBuffer + bufferOffset,
        timeout,
        perGroupLines);
#else
    (void)group;
    (void)src;
    (void)nbytes;
    (void)active_groups;
    (void)timeout;
#endif
  }

  /**
   * ll_recv — Receive data from local LL buffer.
   *
   * Polls the local LL buffer (written remotely by peer), reads
   * payload to output buffer, and ACKs with kLlReadyToWrite.
   *
   * PRECONDITION: llBufferSize > 0 in transport config.
   *
   * @param group         ThreadGroup (auto-converted to warp scope)
   * @param dst           Local output buffer (8-byte aligned)
   * @param nbytes        Total bytes (must be a multiple of 8)
   * @param active_groups Number of groups calling concurrently.
   *   0 = default to max groups the LL buffer can support.
   *   >0 = explicit group count; buffer partitioned per group.group_id.
   * @param timeout       Timeout for flag polling
   */
  __device__ __forceinline__ void ll_recv(
      const ThreadGroup& group,
      char* dst,
      size_t nbytes,
      int active_groups = 0,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__ // NVIDIA-only: depends on ll_send/ll_recv/ll128_* not yet
                     // ported to AMD
    PIPES_DEVICE_CHECK(localState_.llBuffer != nullptr);
    PIPES_DEVICE_CHECK(can_use_ll(dst, nbytes, options_.llBufferNumLines));

    const int maxGroups =
        (options_.llBufferNumLines >= static_cast<size_t>(kLlLinesPerWarp))
        ? static_cast<int>(options_.llBufferNumLines / kLlLinesPerWarp)
        : 1;
    const int effActive = active_groups > 0 ? active_groups : maxGroups;

    PIPES_DEVICE_CHECK(static_cast<uint32_t>(effActive) <= group.total_groups);
    PIPES_DEVICE_CHECK(group.group_id < static_cast<uint32_t>(effActive));

    const size_t perGroupLines = options_.llBufferNumLines / effActive;
    if (effActive > 1 && options_.llBufferNumLines > 0) {
      PIPES_DEVICE_CHECK(perGroupLines >= kLlLinesPerWarp);
    }
    const size_t bufferOffset = group.group_id * perGroupLines;

    comms::prims::ll_recv(
        group,
        dst,
        nbytes,
        localState_.llBuffer + bufferOffset,
        timeout,
        perGroupLines);
#else
    (void)group;
    (void)dst;
    (void)nbytes;
    (void)active_groups;
    (void)timeout;
#endif
  }

  /**
   * ll_forward — Receive from predecessor and forward to successor.
   *
   * Reads from this transport's local LL buffer (predecessor wrote here),
   * forwards to successor_transport's remote LL buffer, copies payload
   * to local output, and ACKs predecessor.
   *
   * PRECONDITION: llBufferSize > 0 in both this and successor transport.
   *
   * @param group                ThreadGroup (auto-converted to warp scope)
   * @param dst                  Local output buffer (8-byte aligned)
   * @param nbytes               Total bytes (must be a multiple of 8)
   * @param successor_transport  Transport for the successor peer
   * @param active_groups        Number of groups calling concurrently.
   *   0 = default to max groups the LL buffer can support.
   *   >0 = explicit group count; buffer partitioned per group.group_id.
   * @param timeout              Timeout for flag polling
   */
  __device__ __forceinline__ void ll_forward(
      const ThreadGroup& group,
      char* dst,
      size_t nbytes,
      const P2pNvlTransportDevice& successor_transport,
      int active_groups = 0,
      const Timeout& timeout = Timeout()) {
#ifdef __CUDA_ARCH__ // NVIDIA-only: depends on ll_send/ll_recv/ll128_* not yet
                     // ported to AMD
    PIPES_DEVICE_CHECK(localState_.llBuffer != nullptr);
    PIPES_DEVICE_CHECK(successor_transport.remoteState_.llBuffer != nullptr);
    PIPES_DEVICE_CHECK(can_use_ll(dst, nbytes, options_.llBufferNumLines));

    const int myMax =
        (options_.llBufferNumLines >= static_cast<size_t>(kLlLinesPerWarp))
        ? static_cast<int>(options_.llBufferNumLines / kLlLinesPerWarp)
        : 1;
    const int succMax = (successor_transport.options_.llBufferNumLines >=
                         static_cast<size_t>(kLlLinesPerWarp))
        ? static_cast<int>(
              successor_transport.options_.llBufferNumLines / kLlLinesPerWarp)
        : 1;
    const int maxGroups = myMax < succMax ? myMax : succMax;
    const int effActive = active_groups > 0 ? active_groups : maxGroups;

    PIPES_DEVICE_CHECK(static_cast<uint32_t>(effActive) <= group.total_groups);
    PIPES_DEVICE_CHECK(group.group_id < static_cast<uint32_t>(effActive));

    const size_t myPerGroup = options_.llBufferNumLines / effActive;
    const size_t succPerGroup =
        successor_transport.options_.llBufferNumLines / effActive;
    if (effActive > 1) {
      if (options_.llBufferNumLines > 0) {
        PIPES_DEVICE_CHECK(myPerGroup >= kLlLinesPerWarp);
      }
      if (successor_transport.options_.llBufferNumLines > 0) {
        PIPES_DEVICE_CHECK(succPerGroup >= kLlLinesPerWarp);
      }
    }
    const size_t myOffset = group.group_id * myPerGroup;
    const size_t succOffset = group.group_id * succPerGroup;

    // Asymmetric buffer sizing: 0 means "pre-sized to fit message."
    // Use the non-zero value when only one side is chunked.
    size_t effectiveLines;
    if (myPerGroup > 0 && succPerGroup > 0) {
      effectiveLines = myPerGroup < succPerGroup ? myPerGroup : succPerGroup;
    } else if (myPerGroup > 0) {
      effectiveLines = myPerGroup;
    } else {
      effectiveLines = succPerGroup;
    }

    comms::prims::ll_forward(
        group,
        dst,
        nbytes,
        localState_.llBuffer + myOffset,
        successor_transport.remoteState_.llBuffer + succOffset,
        timeout,
        effectiveLines);
#else
    (void)group;
    (void)dst;
    (void)nbytes;
    (void)successor_transport;
    (void)active_groups;
    (void)timeout;
#endif
  }

  /**
   * get_ll_buffer_num_lines — Get the number of LL lines in the buffer.
   */
  __device__ __forceinline__ size_t get_ll_buffer_num_lines() const {
    return options_.llBufferNumLines;
  }

 private:
  __device__ __forceinline__ static std::size_t align_tile_protocol_bytes(
      std::size_t nbytes) {
    return (nbytes + 15ULL) & ~15ULL;
  }

  __device__ __forceinline__ static std::size_t valid_payload_bytes(
      std::size_t byteOffset,
      std::size_t chunkBytes,
      std::size_t payloadBytes) {
    if (byteOffset >= payloadBytes) {
      return 0;
    }
    const std::size_t remaining = payloadBytes - byteOffset;
    return chunkBytes < remaining ? chunkBytes : remaining;
  }

  const int myRank_{-1};
  const int peerRank_{-1};
  const P2pNvlTransportOptions options_{};
  LocalState localState_;
  RemoteState remoteState_;
  // Per-channel protocol state. Length = options_.max_num_channels.
  // local_channels_: this rank's endpoint; remote sender / recv write
  //   into it via NVLink (data_ready / slot_free fields).
  // remote_channels_: IPC-mapped pointer to the remote rank's local_channels_
  //   array; this rank's send / recv write into it to signal the remote rank.
  NvlChannelState* local_channels_{nullptr};
  NvlChannelState* remote_channels_{nullptr};
};

} // namespace comms::prims
