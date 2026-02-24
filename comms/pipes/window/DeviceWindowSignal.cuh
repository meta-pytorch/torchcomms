// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>
#include <cstdint>
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/SignalState.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

// Debug-only bounds check helpers for device code
// TODO(D91689639): Replace with PIPES_DEVICE_CHECK_MSG once D91689639 lands
#ifdef __CUDA_ARCH__
#define DEVICE_WINDOW_SIGNAL_CHECK_RANK(target_rank, nRanks)            \
  do {                                                                  \
    if (!((target_rank) >= 0 && (target_rank) < (nRanks))) {            \
      printf(                                                           \
          "DeviceWindowSignal: target_rank %d out of range [0, %d) at " \
          "%s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",                 \
          (int)(target_rank),                                           \
          (int)(nRanks),                                                \
          __FILE__,                                                     \
          __LINE__,                                                     \
          blockIdx.x,                                                   \
          blockIdx.y,                                                   \
          blockIdx.z,                                                   \
          threadIdx.x,                                                  \
          threadIdx.y,                                                  \
          threadIdx.z);                                                 \
      __trap();                                                         \
    }                                                                   \
  } while (0)

#define DEVICE_WINDOW_SIGNAL_CHECK_NOT_SELF(target_rank, myRank) \
  do {                                                           \
    if ((target_rank) == (myRank)) {                             \
      printf(                                                    \
          "DeviceWindowSignal: cannot signal self (rank %d) at " \
          "%s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",          \
          (int)(target_rank),                                    \
          __FILE__,                                              \
          __LINE__,                                              \
          blockIdx.x,                                            \
          blockIdx.y,                                            \
          blockIdx.z,                                            \
          threadIdx.x,                                           \
          threadIdx.y,                                           \
          threadIdx.z);                                          \
      __trap();                                                  \
    }                                                            \
  } while (0)

#define DEVICE_WINDOW_SIGNAL_CHECK_SIGNAL_ID(signal_id, signalCount)  \
  do {                                                                \
    if (!((signal_id) >= 0 && (signal_id) < (signalCount))) {         \
      printf(                                                         \
          "DeviceWindowSignal: signal_id %d out of range [0, %d) at " \
          "%s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",               \
          (int)(signal_id),                                           \
          (int)(signalCount),                                         \
          __FILE__,                                                   \
          __LINE__,                                                   \
          blockIdx.x,                                                 \
          blockIdx.y,                                                 \
          blockIdx.z,                                                 \
          threadIdx.x,                                                \
          threadIdx.y,                                                \
          threadIdx.z);                                               \
      __trap();                                                       \
    }                                                                 \
  } while (0)
#else
#define DEVICE_WINDOW_SIGNAL_CHECK_RANK(target_rank, nRanks) \
  assert((target_rank) >= 0 && (target_rank) < (nRanks))
#define DEVICE_WINDOW_SIGNAL_CHECK_NOT_SELF(target_rank, myRank) \
  assert((target_rank) != (myRank))
#define DEVICE_WINDOW_SIGNAL_CHECK_SIGNAL_ID(signal_id, signalCount) \
  assert((signal_id) >= 0 && (signal_id) < (signalCount))
#endif

/**
 * DeviceWindowSignal - Device-side signal object with per-peer inbox semantics
 *
 * Implements a per-peer inbox model for multi-peer signaling:
 * - Each rank has a local inbox with per-peer rows of signal slots
 * - Each peer has its own row: inbox layout is [nPeers][signalCount]
 * - signal_peer() writes to the sender's dedicated sub-slot in the target's
 *   inbox (one NVLink write, no dual-write)
 * - wait_signal_from() polls a specific peer's sub-slot (O(1))
 * - wait_signal() sums across all per-peer sub-slots for accumulated semantics
 * - Inbox size = nPeers * signalCount (one row per peer, signalCount slots
 *   per row)
 *
 * INBOX LAYOUT:
 * =============
 *   [peer0_slot_0] [peer0_slot_1] ... [peer0_slot_{signalCount-1}]
 *   [peer1_slot_0] [peer1_slot_1] ... [peer1_slot_{signalCount-1}]
 *   ...
 *   [peer_{nPeers-1}_slot_0] ... [peer_{nPeers-1}_slot_{signalCount-1}]
 *
 * Each SignalState is 128-byte aligned.
 * Total = nPeers * signalCount * sizeof(SignalState) bytes.
 *
 * RANK-BASED API:
 * All peer-facing APIs use global rank in [0, nRanks), not peer index.
 * Internally, peerSignals_ has nPeers entries (excludes self). The
 * rank_to_peer_index() helper converts rank to the internal peer index.
 *
 * SIGNAL VS COUNTER (NCCL semantics):
 * - Signal: Written to REMOTE peer's memory (remote completion notification)
 * - Counter: Written to LOCAL memory (source buffer consumed, safe to reuse)
 *
 * This class implements the Signal semantics. We will later introduce
 * DeviceCounter for local completion tracking.
 *
 * SLOT MAPPING:
 * signal_peer(target, signal_id) writes to:
 *   target.inbox[myPeerIndexOnTarget * signalCount + signal_id]
 * wait_signal_from(source, signal_id) polls:
 *   localInbox[sourcePeer * signalCount + signal_id]
 * wait_signal(signal_id) sums localInbox[peer * signalCount + signal_id]
 *   across all peers, polls until sum meets condition.
 *
 * WAIT SEMANTICS:
 * wait_signal() sums across all per-peer sub-slots on read for accumulated
 * semantics. With N peers each signaling value=1:
 *   wait_signal(group, signal_id, CMP_GE, N)  // Wait for all N peers
 * wait_signal_from() polls a specific peer's sub-slot:
 *   wait_signal_from(group, source_rank, signal_id, CMP_GE, 1)
 *
 * v1: Unicast writes (parallel writes to each peer)
 * v2 (future): Multi-mem multicast optimization
 *
 * RESET OPERATIONS:
 * DeviceWindowSignal intentionally has NO reset operations. Signals are written
 * by REMOTE peers to this rank's inbox via NVLink. A local reset while a
 * remote peer is signaling creates a race condition. Use monotonically
 * increasing values (wait for N, then 2N, ...) instead of resetting.
 *
 * SIGNAL LIFETIME AND REUSE:
 * ==========================
 * Signal values accumulate across operations. For long-running workloads:
 *
 * Option 1: Use monotonically increasing wait values
 *   signal.signal_peer(group, target_rank, slot, SIGNAL_ADD, 1);
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
 * In device builds (__CUDA_ARCH__ defined), invalid target_rank or signal_id
 * values trigger __trap() with a descriptive error message showing the invalid
 * value, valid range, file/line, and block/thread indices.
 *
 * USAGE:
 *   // Sender (rank 0) signals receiver (rank 1) that data is ready
 *   signal.signal_peer(group, 1, 0, SignalOp::SIGNAL_ADD, 1);
 *
 *   // Receiver (rank 1) waits for signal from a specific peer
 *   signal.wait_signal_from(group, 0, 0, CmpOp::CMP_GE, 1);
 *
 *   // Receiver waits for accumulated signals from all peers (barrier-like)
 *   signal.wait_signal(group, 0, CmpOp::CMP_GE, nRanks - 1);
 */
class DeviceWindowSignal {
 public:
  __host__ __device__ DeviceWindowSignal() = default;

  /**
   * Construct a DeviceWindowSignal with per-peer inbox semantics
   *
   * @param myRank This rank's ID
   * @param nRanks Total number of ranks
   * @param signalCount Number of signal slots per peer
   * @param localInbox This rank's inbox (local memory,
   *        size = nPeers * signalCount)
   * @param peerSignals Peers' inboxes (for remote writes, each span points
   *        to the sender's per-peer row in the target's inbox)
   */
  __host__ __device__ DeviceWindowSignal(
      int myRank,
      int nRanks,
      int signalCount,
      DeviceSpan<SignalState> localInbox,
      DeviceSpan<DeviceSpan<SignalState>> peerSignals)
      : myRank_(myRank),
        nRanks_(nRanks),
        signalCount_(signalCount),
        localInbox_(localInbox),
        peerSignals_(peerSignals) {}

  // ===========================================================================
  // Signal Operations (writing to target's inbox)
  // ===========================================================================

  /**
   * signal_peer - Signal a specific peer by writing to their inbox.
   *
   * Writes to the target's inbox at the signal_id slot.
   * Uses release semantics for memory ordering.
   *
   * @param group ThreadGroup for cooperative processing
   * @param target_rank Global rank in [0, nRanks), must not be self
   * @param signal_id Index within this rank's slot range (0 to signalCount-1)
   * @param op SIGNAL_SET or SIGNAL_ADD
   * @param value Value to set or add (default: 1)
   */
  __device__ __forceinline__ void signal_peer(
      ThreadGroup& group,
      int target_rank,
      int signal_id,
      SignalOp op = SignalOp::SIGNAL_ADD,
      uint64_t value = 1) {
    DEVICE_WINDOW_SIGNAL_CHECK_RANK(target_rank, nRanks_);
    DEVICE_WINDOW_SIGNAL_CHECK_NOT_SELF(target_rank, myRank_);
    DEVICE_WINDOW_SIGNAL_CHECK_SIGNAL_ID(signal_id, signalCount_);
    int peer = rank_to_peer_index(target_rank);
    peerSignals_[peer][signal_id].signal(group, op, value);
  }

  /**
   * signal_all - Signal all peers (parallel unicast writes)
   *
   * Writes to each peer's inbox at the sender's per-peer sub-slot via NVLink.
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
    DEVICE_WINDOW_SIGNAL_CHECK_SIGNAL_ID(signal_id, signalCount_);

    // Sync BEFORE: ensures all threads' prior writes are complete
    group.sync();

    // Thread-level parallelism: each thread signals different peers
    // peerSignals_ has nPeers entries (excludes self)
    int nPeers = static_cast<int>(peerSignals_.size());
    for (int peer = static_cast<int>(group.thread_id_in_group); peer < nPeers;
         peer += static_cast<int>(group.group_size)) {
      peerSignals_[peer][signal_id].signal(op, value);
    }

    // Sync AFTER: ensures all signals complete before returning
    group.sync();
  }

  // ===========================================================================
  // Wait Operations (reading from local inbox)
  // ===========================================================================

  /**
   * wait_signal - Wait for accumulated signal value across all peers
   *
   * Sums across all per-peer sub-slots for the given signal_id and polls
   * until the sum meets the comparison condition. Uses acquire semantics
   * for memory ordering.
   *
   * This provides backward-compatible accumulated semantics: with N peers
   * each signaling value=1, wait_signal(group, id, CMP_GE, N) works as
   * expected.
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
    DEVICE_WINDOW_SIGNAL_CHECK_SIGNAL_ID(signal_id, signalCount_);
    int nPeers = nRanks_ - 1;

#ifdef __CUDA_ARCH__
    if (group.scope == SyncScope::WARP) {
      // Optimized path: parallel load + warp shuffle reduction
      while (true) {
        // Parallel load: each thread loads subset of peers
        uint64_t partial_sum = 0;
        for (int peer = static_cast<int>(group.thread_id_in_group);
             peer < nPeers;
             peer += static_cast<int>(group.group_size)) {
          int slot_index = peer * signalCount_ + signal_id;
          partial_sum += localInbox_[slot_index].load();
        }

        // Warp reduction via shuffle
#pragma unroll
        for (unsigned offset = 16; offset > 0; offset /= 2) {
          partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }

        // Broadcast result from thread 0 to ALL threads
        uint64_t total_sum = __shfl_sync(0xffffffff, partial_sum, 0);

        // All threads check condition together and exit together
        if (compare(total_sum, cmp, value)) {
          break;
        }
      }
    } else
#endif
    {
      // Fallback path: all threads poll (for BLOCK/CLUSTER/MULTIWARP scopes)
      while (true) {
        uint64_t sum = 0;
        for (int peer = 0; peer < nPeers; ++peer) {
          int slot_index = peer * signalCount_ + signal_id;
          sum += localInbox_[slot_index].load();
        }
        if (compare(sum, cmp, value)) {
          break;
        }
      }
    }
  }

  /**
   * wait_signal_from - Wait for signal from a specific peer
   *
   * Polls the specific peer's sub-slot in the local inbox until the
   * condition is met. Uses acquire semantics for memory ordering.
   *
   * @param group ThreadGroup for cooperative processing
   * @param source_rank Global rank of the source peer in [0, nRanks),
   *        must not be self
   * @param signal_id Index of the signal slot (0 to signalCount-1)
   * @param cmp Comparison operation (CMP_EQ, CMP_GE, etc.)
   * @param value Value to compare against
   * @param timeout_ns Optional timeout in nanoseconds (default: infinite)
   */
  __device__ __forceinline__ void wait_signal_from(
      ThreadGroup& group,
      int source_rank,
      int signal_id,
      CmpOp cmp,
      uint64_t value,
      [[maybe_unused]] uint64_t timeout_ns = UINT64_MAX) {
    DEVICE_WINDOW_SIGNAL_CHECK_RANK(source_rank, nRanks_);
    DEVICE_WINDOW_SIGNAL_CHECK_NOT_SELF(source_rank, myRank_);
    DEVICE_WINDOW_SIGNAL_CHECK_SIGNAL_ID(signal_id, signalCount_);
    int source_peer = rank_to_peer_index(source_rank);
    int slot_index = source_peer * signalCount_ + signal_id;
    localInbox_[slot_index].wait_until(group, cmp, value);
  }

  // ===========================================================================
  // Non-blocking Read Operations
  // ===========================================================================

  /**
   * read_signal - Read accumulated signal value from a slot (non-blocking)
   *
   * Sums across all per-peer sub-slots for the given signal_id.
   * Consistent with wait_signal() semantics.
   *
   * @param signal_id Index of the signal slot
   * @return Current accumulated signal value (sum across all peers)
   */
  __device__ __forceinline__ uint64_t read_signal(int signal_id) {
    DEVICE_WINDOW_SIGNAL_CHECK_SIGNAL_ID(signal_id, signalCount_);
    int nPeers = nRanks_ - 1;
    uint64_t sum = 0;
    for (int peer = 0; peer < nPeers; ++peer) {
      int slot_index = peer * signalCount_ + signal_id;
      sum += localInbox_[slot_index].load();
    }
    return sum;
  }

  /**
   * read_signal_from - Read signal value from a specific peer (non-blocking)
   *
   * @param source_rank Global rank of the source peer in [0, nRanks),
   *        must not be self
   * @param signal_id Index of the signal slot
   * @return Current signal value from the specified peer
   */
  __device__ __forceinline__ uint64_t
  read_signal_from(int source_rank, int signal_id) {
    DEVICE_WINDOW_SIGNAL_CHECK_RANK(source_rank, nRanks_);
    DEVICE_WINDOW_SIGNAL_CHECK_NOT_SELF(source_rank, myRank_);
    DEVICE_WINDOW_SIGNAL_CHECK_SIGNAL_ID(signal_id, signalCount_);
    int source_peer = rank_to_peer_index(source_rank);
    int slot_index = source_peer * signalCount_ + signal_id;
    return localInbox_[slot_index].load();
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
    return static_cast<int>(peerSignals_.size());
  }

  __device__ __forceinline__ int signal_count() const {
    return signalCount_;
  }

 private:
  /**
   * compare - Helper for sum-based wait_signal() polling
   *
   * @param actual The computed sum value
   * @param cmp Comparison operation
   * @param expected The expected value to compare against
   * @return true if the comparison is satisfied
   */
  __device__ __forceinline__ static bool
  compare(uint64_t actual, CmpOp cmp, uint64_t expected) {
    switch (cmp) {
      case CmpOp::CMP_EQ:
        return actual == expected;
      case CmpOp::CMP_NE:
        return actual != expected;
      case CmpOp::CMP_GE:
        return actual >= expected;
      case CmpOp::CMP_GT:
        return actual > expected;
      case CmpOp::CMP_LE:
        return actual <= expected;
      case CmpOp::CMP_LT:
        return actual < expected;
    }
    return false;
  }

  /**
   * rank_to_peer_index - Convert global rank to internal peer index
   *
   * Peer index excludes self: ranks below myRank_ map directly,
   * ranks above myRank_ are shifted down by one.
   *
   * @param rank Global rank (must NOT be myRank_)
   * @return Peer index for indexing into peerSignals_
   */
  __device__ __forceinline__ int rank_to_peer_index(int rank) const {
    return (rank < myRank_) ? rank : (rank - 1);
  }

  int myRank_{-1};
  int nRanks_{0};
  int signalCount_{0};
  DeviceSpan<SignalState> localInbox_;
  DeviceSpan<DeviceSpan<SignalState>> peerSignals_;
};

/**
 * getSignalInboxBufferSize - Calculate buffer size for per-peer signal inbox
 *
 * @param signalCount Number of signal slots per peer
 * @param nPeers Number of peers (nRanks - 1)
 * @return Size in bytes, aligned to 128-byte boundary
 */
__host__ __device__ __forceinline__ std::size_t getSignalInboxBufferSize(
    int signalCount,
    int nPeers) {
  return getSignalBufferSize(nPeers * signalCount);
}

} // namespace comms::pipes
