// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Tile send/recv — bidirectional pipelined data transfer over NVLink.
//
// OVERVIEW
// ========
// Each GPU simultaneously sends AND receives by partitioning the kernel's
// thread blocks into two roles (sender blocks and receiver blocks) and
// giving each block an independent tile of data. The transport pipelines
// tiles through a staging buffer using monotonic head/tail signal counters
// for backpressure.
//
// KERNEL LAUNCH
// =============
// The kernel is launched with 2 * numSendBlocks total blocks:
//   - Blocks [0, numSendBlocks)              → sender role
//   - Blocks [numSendBlocks, 2*numSendBlocks) → receiver role
//
// Partitioning is done via ThreadGroup::partition(2):
//   auto [role, sub] = group.partition(2);
//   role == 0 → sender,  sub.group_id ∈ [0, numSendBlocks)
//   role == 1 → receiver, sub.group_id ∈ [0, numSendBlocks)
//
// DATA PARTITIONING (caller-side)
// ===============================
// The caller divides the message into numSendBlocks contiguous tiles:
//
//   tileSize = ceil(nBytes / numSendBlocks) aligned to 16B
//   tile i   = [i * tileSize, min((i+1) * tileSize, nBytes))
//
// Each sender block i sends tile i; each receiver block i receives tile i.
// Sender block i is paired with receiver block i on the remote GPU.
//
// PIPELINING (inside send / recv)
// =========================================
// Each block's tile may be larger than the per-block staging area. The tile
// is therefore pipelined through the staging buffer in multiple steps:
//
//   perBlockSlotSize = floor(dataBufferSize / numBlocks) & ~15
//   totalSlotSteps   = ceil(tileBytes / perBlockSlotSize)
//
// Each step uses one of `pipelineDepth` slots (step % pipelineDepth) to
// allow sender and receiver to overlap:
//
//   Step 0: sender writes to slot 0, signals tail=1
//   Step 1: sender writes to slot 1, signals tail=2
//           receiver reads slot 0, signals head=1
//   Step 2: sender waits for head >= 1 (slot 0 freed), writes slot 0
//           receiver reads slot 1, signals head=2
//   ...
//
// SIGNAL PROTOCOL
// ===============
// Uses 2 * numSendBlocks SignalState entries (128-byte aligned, sys scope):
//
//   signal[i]                = tail counter for block pair i
//                              (sender → receiver: "data is ready")
//   signal[numBlocks + i]   = head counter for block pair i
//                              (receiver → sender: "slot is freed")
//
// Both counters are monotonically increasing. The sender waits for
//   head >= step - pipelineDepth + 1
// before reusing a slot (backpressure). The receiver waits for
//   tail >= step + 1
// before reading (data availability).
//
// Memory ordering:
//   - signal() uses st.release.sys.global → all prior writes (memcpy data)
//     are visible to the peer before the signal is observed.
//   - wait_until() uses ld.acquire.sys.global → all subsequent reads see
//     the data the peer wrote before signaling.
//
// MULTI-CALL CORRECTNESS
// ======================
// The step counters are persisted in device memory (`stepState`):
//   stepState[0..numBlocks-1]           = sender step per block
//   stepState[numBlocks..2*numBlocks-1] = receiver step per block
//
// On first call (stepState zeroed), sender starts at step=0, receiver at
// step=0. On subsequent calls, they resume from where they left off.
// Because signals are monotonically increasing, old signal values from
// previous calls are always < current expected values, so no ABA issue.
//
// PRECONDITION: stepState must be zeroed before the first kernel launch.
// The transport's exchange() zeroes the signal buffers. For repeated
// launches with the same transport, the persistent step counters handle
// correctness automatically.
//
//
// CORRECTNESS ANALYSIS
// ====================
// 1. No data race on staging buffer:
//    - Sender writes to slot S, then signals tail.
//    - Receiver waits for tail (acquire), reads slot S, then signals head.
//    - Sender waits for head (acquire) before reusing slot S.
//    → Slot S is never written and read simultaneously.
//
// 2. No signal race:
//    - Each (tail, head) pair is used by exactly one sender block and one
//      receiver block. No two blocks share a signal slot.
//
// 3. No ABA on signals:
//    - Monotonically increasing step values + CMP_GE comparisons ensure
//      stale values from prior steps are always < expected.
//
// 4. group.sync() before signal:
//    - Ensures all threads in the block complete their memcpy_vectorized
//      before the leader signals. Without this, the peer could observe
//      the signal before all data is written.

#pragma once

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/TiledBuffer.cuh"

namespace comms::pipes::benchmark {

constexpr int kNumSlots = 2;
constexpr std::size_t kSlotSize = 16 * 1024 * 1024; // 16MB per slot

// Step state: persistent per-block step counters for multi-call correctness.
// Array of int64: [0..numSendBlocks-1] = sender steps,
//                 [numSendBlocks..2*numSendBlocks-1] = receiver steps.
// Initialize to 0 before first use. The kernel loads at start, stores at end.

/**
 * p2pTileSendRecv — Bidirectional tiled send/recv kernel.
 *
 * Launches with 2 * numSendBlocks blocks. The first half are senders,
 * the second half are receivers. Each sender/receiver pair transfers
 * an independent tile of data through the pipelined staging buffer.
 *
 * Uses TiledBuffer<char> to partition data across blocks, eliminating
 * manual offset math. Each block queries its tile pointer and size from
 * the TiledBuffer.
 *
 * @param p2p           Transport device (passed by value from host memory)
 * @param sendTiles     Tiled view of the send buffer
 * @param recvTiles     Tiled view of the recv buffer
 * @param stepState     Persistent step counters [2 * numSendBlocks int64s],
 *                      zeroed before first use
 * @param timeout       Optional timeout for signal waits
 */
__global__ void p2pTileSendRecv(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    int active_blocks,
    std::size_t max_signal_bytes = 0,
    Timeout timeout = Timeout());

/**
 * p2pTileSendRecvDynamic — Variant using transport-internal tile state
 * with support for dynamic block count changes.
 *
 * Requires tile_max_groups > 0 and p2pBarrierCount >= tile_max_groups.
 * StepState, signals, and maxBlocks are managed internally by the transport.
 *
 * Signal layout uses maxBlocks (constant across launches) so that block k
 * always maps to signal slot k regardless of numBlocks. The staging buffer
 * partition uses numBlocks (variable) for efficient use of staging memory.
 *
 * When numBlocks changes, the caller must set needsBarrier=true. Each block
 * does barrier_sync with its peer to ensure the remote GPU's
 * previous kernel completed all staging reads before the new layout takes
 * effect. See TileSendRecv.cu for the full correctness analysis.
 *
 * CUDA graph compatible: all synchronization is device-side.
 *
 * @param numBlocks    Active block count (controls staging partition).
 *                     Launch with 2 * numBlocks total blocks.
 * @param needsBarrier Set true when numBlocks changed since last call.
 */
__global__ void p2pTileSendRecvDynamic(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    int active_blocks,
    bool needsBarrier,
    Timeout timeout = Timeout());

} // namespace comms::pipes::benchmark
