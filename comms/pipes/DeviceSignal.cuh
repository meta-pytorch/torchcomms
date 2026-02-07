// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>
#include <cstdint>
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/SignalState.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

// Debug-only bounds check helper for device code
// TODO(D91689639): Replace with PIPES_DEVICE_CHECK_MSG once D91689639 lands
#ifdef __CUDA_ARCH__
#define DEVICE_SIGNAL_CHECK_BOUNDS(index, limit)            \
  do {                                                      \
    if (!((index) >= 0 && (index) < (limit))) {             \
      printf(                                               \
          "DeviceSignal: index %d out of range [0, %d) at " \
          "%s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",     \
          (int)(index),                                     \
          (int)(limit),                                     \
          __FILE__,                                         \
          __LINE__,                                         \
          blockIdx.x,                                       \
          blockIdx.y,                                       \
          blockIdx.z,                                       \
          threadIdx.x,                                      \
          threadIdx.y,                                      \
          threadIdx.z);                                     \
      __trap();                                             \
    }                                                       \
  } while (0)

#define DEVICE_SIGNAL_CHECK_SIGNAL_ID(signal_id, signalCount)   \
  do {                                                          \
    if (!((signal_id) >= 0 && (signal_id) < (signalCount))) {   \
      printf(                                                   \
          "DeviceSignal: signal_id %d out of range [0, %d) at " \
          "%s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",         \
          (int)(signal_id),                                     \
          (int)(signalCount),                                   \
          __FILE__,                                             \
          __LINE__,                                             \
          blockIdx.x,                                           \
          blockIdx.y,                                           \
          blockIdx.z,                                           \
          threadIdx.x,                                          \
          threadIdx.y,                                          \
          threadIdx.z);                                         \
      __trap();                                                 \
    }                                                           \
  } while (0)
#else
#define DEVICE_SIGNAL_CHECK_BOUNDS(index, limit) \
  assert((index) >= 0 && (index) < (limit))
#define DEVICE_SIGNAL_CHECK_SIGNAL_ID(signal_id, signalCount) \
  assert((signal_id) >= 0 && (signal_id) < (signalCount))
#endif

/**
 * DeviceSignal - Device-side signal object with per-signal inbox semantics
 *
 * Implements NCCL-style "per-signal inbox" model for multi-peer signaling:
 * - Each rank has a single inbox buffer (local memory)
 * - All peers write to the SAME slot for a given signal_id
 * - Values accumulate: slot[signal_id] += value from each peer
 * - Inbox size = signalCount (one slot per signal)
 *
 * PEER INDEX SPACE:
 * All peer-facing APIs use peer index (0 to nPeers-1), not rank.
 * - For ranks [0, 1, 2, 3, 4] where myRank=2:
 *   - Rank space: 0, 1, 2, 3, 4 (5 entries, includes self)
 *   - Peer space: 0, 1, 2, 3 (4 entries, excludes self)
 * - peerInboxPtrs_ has nPeers entries (not nRanks)
 *
 * SIGNAL VS COUNTER (NCCL semantics):
 * - Signal: Written to REMOTE peer's memory (remote completion notification)
 * - Counter: Written to LOCAL memory (source buffer consumed, safe to reuse)
 *
 * This class implements the Signal semantics. We will later introduce
 * DeviceCounter for local completion tracking.
 *
 * SLOT MAPPING (NCCL-style):
 * All peers write to the same slot: slot = signal_id
 * Values accumulate from all peers who signal with the same signal_id.
 *
 * WAIT SEMANTICS:
 * wait_signal() waits for the accumulated value in the inbox slot to reach
 * the expected threshold. With N peers each signaling value=1:
 *   wait_signal(group, signal_id, CMP_GE, N)  // Wait for all N peers
 *
 * v1: Unicast writes (parallel writes to each peer)
 * v2 (future): Multi-mem multicast optimization
 *
 * RESET OPERATIONS:
 * DeviceSignal intentionally has NO reset operations. Signals are written
 * by REMOTE peers to this rank's inbox via NVLink. A local reset while a
 * remote peer is signaling creates a race condition. Use monotonically
 * increasing values (wait for N, then 2N, ...) instead of resetting.
 *
 * SIGNAL LIFETIME AND REUSE:
 * ==========================
 * Signal values accumulate across operations. For long-running workloads:
 *
 * Option 1: Use monotonically increasing wait values
 *   signal.signal_peer(peer, group, slot, SIGNAL_ADD, 1);
 *   signal.wait_signal(group, slot, CMP_GE, N);  // Wait for N total signals
 *
 * Option 2: Rotate between signal slots
 *   int slot = iteration % signalCount;  // Use different slots per phase
 *
 * Option 3: Reset between phases (requires host synchronization)
 *   // After MPI_Barrier, before new phase:
 *   cudaMemset(signalInboxPtr, 0, signalInboxSize);
 *
 * BOUNDS CHECKING:
 * ================
 * In device builds (__CUDA_ARCH__ defined), invalid peer or signal_id values
 * trigger __trap() with a descriptive error message showing the invalid value,
 * valid range, file/line, and block/thread indices.
 *
 * USAGE:
 *   // Sender (rank 0) signals receiver (rank 1) that data is ready
 *   // Peer index for rank 1 when myRank=0 is 0 (first peer)
 *   signal.signal_peer(0, group, 0, SignalOp::SIGNAL_ADD, 1);
 *
 *   // Receiver (rank 1) waits for accumulated signals
 *   signal.wait_signal(group, 0, CmpOp::CMP_GE, 1);
 *
 *   // Wait for signals from all N-1 peers (barrier-like)
 *   signal.wait_signal(group, 0, CmpOp::CMP_GE, nRanks - 1);
 */
class DeviceSignal {
 public:
  __host__ __device__ DeviceSignal() = default;

  /**
   * Construct a DeviceSignal with inbox semantics
   *
   * @param myRank This rank's ID
   * @param nRanks Total number of ranks
   * @param signalCount Number of signal slots per peer
   * @param localInbox This rank's inbox (local memory, size = signalCount)
   * @param peerInboxPtrs Pointers to all peers' inboxes (for remote writes)
   */
  __host__ __device__ DeviceSignal(
      int myRank,
      int nRanks,
      int signalCount,
      DeviceSpan<SignalState> localInbox,
      DeviceSpan<SignalState*> peerInboxPtrs)
      : myRank_(myRank),
        nRanks_(nRanks),
        signalCount_(signalCount),
        localInbox_(localInbox),
        peerInboxPtrs_(peerInboxPtrs) {}

  // ===========================================================================
  // Signal Operations (writing to target's inbox)
  // ===========================================================================

  /**
   * signal_peer - Signal a specific peer by writing to their inbox.
   *
   * Writes to the target's inbox at the signal_id slot.
   * Uses release semantics for memory ordering.
   *
   * @param peer Peer index (0 to num_peers()-1, NOT rank)
   * @param group ThreadGroup for cooperative processing
   * @param signal_id Index within this rank's slot range (0 to signalCount-1)
   * @param op SIGNAL_SET or SIGNAL_ADD
   * @param value Value to set or add (default: 1)
   */
  __device__ __forceinline__ void signal_peer(
      int peer,
      ThreadGroup& group,
      int signal_id,
      SignalOp op = SignalOp::SIGNAL_ADD,
      uint64_t value = 1) {
    DEVICE_SIGNAL_CHECK_BOUNDS(peer, peerInboxPtrs_.size());
    DEVICE_SIGNAL_CHECK_SIGNAL_ID(signal_id, signalCount_);
    peerInboxPtrs_[peer][signal_id].signal(group, op, value);
  }

  /**
   * signal_all - Signal all peers (parallel unicast writes)
   *
   * Writes to all peers' inboxes via NVLink.
   * Uses thread-level parallelism: each thread signals different peers.
   * v1: Parallel unicast writes to each peer
   * v2 (future): Multi-mem multicast for efficiency
   *
   * MEMORY ORDERING: Syncs before signaling to ensure all threads' prior
   * writes are visible. The single-thread signal() uses release semantics
   * which only guarantees THIS thread's writes are visible. A preceding
   * group.sync() ensures ALL threads' writes are complete before any signal
   * is sent.
   *
   * @param group ThreadGroup for cooperative processing
   * @param signal_id Index within this rank's slot range
   * @param op SIGNAL_SET or SIGNAL_ADD
   * @param value Value to set or add (default: 1)
   */
  __device__ __forceinline__ void signal_all(
      ThreadGroup& group,
      int signal_id,
      SignalOp op = SignalOp::SIGNAL_ADD,
      uint64_t value = 1) {
    DEVICE_SIGNAL_CHECK_SIGNAL_ID(signal_id, signalCount_);

    // Sync BEFORE: ensures all threads' prior writes are complete
    group.sync();

    // Thread-level parallelism: each thread signals different peers
    // peerInboxPtrs_ has nPeers entries (excludes self)
    int nPeers = static_cast<int>(peerInboxPtrs_.size());
    for (int peer = static_cast<int>(group.thread_id_in_group); peer < nPeers;
         peer += static_cast<int>(group.group_size)) {
      peerInboxPtrs_[peer][signal_id].signal(op, value);
    }

    // Sync AFTER: ensures all signals complete before returning
    group.sync();
  }

  // ===========================================================================
  // Wait Operations (reading from local inbox)
  // ===========================================================================

  /**
   * wait_signal - Wait for accumulated signal value
   *
   * Polls local inbox at the signal_id slot until the condition is met.
   * Uses acquire semantics for memory ordering.
   *
   * In the per-signal inbox model, all peers write to the same slot, so
   * the accumulated value represents the sum of all received signals.
   *
   * @param group ThreadGroup for cooperative processing
   * @param signal_id Index of the signal slot (0 to signalCount-1)
   * @param cmp Comparison operation (CMP_EQ, CMP_GE, etc.)
   * @param value Value to compare against (e.g., nRanks-1 for all peers)
   * @param timeout_ns Optional timeout in nanoseconds (default: infinite)
   */
  __device__ __forceinline__ void wait_signal(
      ThreadGroup& group,
      int signal_id,
      CmpOp cmp,
      uint64_t value,
      [[maybe_unused]] uint64_t timeout_ns = UINT64_MAX) {
    DEVICE_SIGNAL_CHECK_SIGNAL_ID(signal_id, signalCount_);
    localInbox_[signal_id].wait_until(group, cmp, value);
  }

  // ===========================================================================
  // Non-blocking Read Operations
  // ===========================================================================

  /**
   * read_signal - Read current signal value from a slot (non-blocking)
   *
   * @param signal_id Index of the signal slot
   * @return Current accumulated signal value
   */
  __device__ __forceinline__ uint64_t read_signal(int signal_id) {
    DEVICE_SIGNAL_CHECK_SIGNAL_ID(signal_id, signalCount_);
    return localInbox_[signal_id].load();
  }

  // ===========================================================================
  // Accessors
  // ===========================================================================

  __device__ __forceinline__ int rank() const {
    return myRank_;
  }

  __device__ __forceinline__ int n_ranks() const {
    return nRanks_;
  }

  __device__ __forceinline__ int num_peers() const {
    return static_cast<int>(peerInboxPtrs_.size());
  }

  __device__ __forceinline__ int signal_count() const {
    return signalCount_;
  }

 private:
  int myRank_{-1};
  int nRanks_{0};
  int signalCount_{0};
  DeviceSpan<SignalState> localInbox_;
  DeviceSpan<SignalState*> peerInboxPtrs_;
};

/**
 * getSignalInboxBufferSize - Calculate buffer size for signal inbox
 *
 * @param signalCount Number of signal slots
 * @return Size in bytes, aligned to 128-byte boundary
 */
__host__ __device__ __forceinline__ std::size_t getSignalInboxBufferSize(
    int signalCount) {
  return getSignalBufferSize(signalCount);
}

} // namespace comms::pipes
