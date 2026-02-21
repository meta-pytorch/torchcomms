// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>
#include <cstdint>
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/window/DeviceWindowBarrier.cuh"
#include "comms/pipes/window/DeviceWindowMemory.cuh"
#include "comms/pipes/window/DeviceWindowSignal.cuh"

namespace comms::pipes {

/**
 * MULTI_PEER_CHECK_RANK - Bounds checking macro for rank validation
 *
 * Validates that target_rank is in range [0, nRanks). This macro is used
 * throughout MultiPeerDeviceTransport to catch invalid ranks early.
 *
 * BEHAVIOR:
 * - Device code (__CUDA_ARCH__): Prints debug info including file:line,
 *   block/thread indices, then calls __trap() to abort the kernel.
 *   This is useful for debugging but cannot be caught by host-side testing.
 *
 * - Host code: Uses assert() which can be tested with EXPECT_DEBUG_DEATH
 *   in debug builds, but is a no-op in release builds.
 *
 * USAGE:
 *   MULTI_PEER_CHECK_PEER_INDEX(targetRank, num_peers());
 *
 * TESTED BY:
 * - Valid peer ranges are tested by all functional tests that use peer indices
 * - Invalid peer handling is documented here; device-side __trap() cannot be
 *   unit tested but will produce clear error messages during development
 *
 * TODO(D91689639): Replace with PIPES_DEVICE_CHECK_MSG once D91689639 lands
 */
#ifdef __CUDA_ARCH__
#define MULTI_PEER_CHECK_RANK(target_rank, nRanks)                        \
  do {                                                                    \
    if (!((target_rank) >= 0 && (target_rank) < (nRanks))) {              \
      printf(                                                             \
          "MultiPeerDeviceTransport: target_rank %d out of range [0, %d)" \
          " at %s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",               \
          (int)(target_rank),                                             \
          (int)(nRanks),                                                  \
          __FILE__,                                                       \
          __LINE__,                                                       \
          blockIdx.x,                                                     \
          blockIdx.y,                                                     \
          blockIdx.z,                                                     \
          threadIdx.x,                                                    \
          threadIdx.y,                                                    \
          threadIdx.z);                                                   \
      __trap();                                                           \
    }                                                                     \
  } while (0)
#else
#define MULTI_PEER_CHECK_RANK(target_rank, nRanks) \
  assert((target_rank) >= 0 && (target_rank) < (nRanks))
#endif

/**
 * MULTI_PEER_CHECK_NOT_SELF - Validates target_rank != myRank
 *
 * Operations that target a specific peer (send, recv, put, put_signal,
 * signal_peer, barrier_peer, get_peer_transport) do not support self.
 * Passing target_rank == myRank traps with a diagnostic message.
 *
 * For local (self) operations, use get_self_transport() instead.
 *
 * USAGE:
 *   MULTI_PEER_CHECK_NOT_SELF(target_rank, myRank_);
 */
#ifdef __CUDA_ARCH__
#define MULTI_PEER_CHECK_NOT_SELF(target_rank, myRank)               \
  do {                                                               \
    if ((target_rank) == (myRank)) {                                 \
      printf(                                                        \
          "MultiPeerDeviceTransport: self-rank %d not supported at " \
          "%s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",              \
          (int)(target_rank),                                        \
          __FILE__,                                                  \
          __LINE__,                                                  \
          blockIdx.x,                                                \
          blockIdx.y,                                                \
          blockIdx.z,                                                \
          threadIdx.x,                                               \
          threadIdx.y,                                               \
          threadIdx.z);                                              \
      __trap();                                                      \
    }                                                                \
  } while (0)
#else
#define MULTI_PEER_CHECK_NOT_SELF(target_rank, myRank) \
  assert((target_rank) != (myRank))
#endif

/**
 * MultiPeerDeviceTransport - Unified device-side multi-peer NVLink transport
 *
 * Provides a single device object with:
 * - Rank-based send/recv/put operations
 * - DeviceWindowSignal for inbox-style signaling (remote notification)
 * - DeviceWindowBarrier for multi-peer synchronization
 *
 * RANK-BASED API:
 * All public APIs that target a specific peer accept a target_rank in the
 * range [0, n_ranks()). Passing target_rank == rank() (self) to peer-
 * targeting operations will __trap() with a diagnostic message.
 *
 * For local (self) operations, use get_self_transport() to access the
 * self transport directly.
 *
 * Utility methods peer_index_to_rank() and rank_to_peer_index() are
 * provided for converting between peer index and global rank when needed.
 *
 * DESIGN:
 * - Aligned with TorchComm Device API style (D91172575)
 * - Signal/Barrier use inbox model (one inbox per rank)
 *
 * API COEXISTENCE:
 * This multi-peer API coexists with the existing P2pNvlTransportDevice API:
 * - P2pNvlTransportDevice: P2P (one-to-one), per-peer signal buffers
 * - MultiPeerDeviceTransport: Unified API with inbox model, multi-peer coord
 *
 * MEMORY ORDERING SEMANTICS
 * =========================
 *
 * Signal Operations (signal_peer, signal_all):
 * - Release semantics: All prior writes visible before signal
 * - Uses st_release_sys_global (system-scope release store)
 *
 * Wait Operations (wait_signal):
 * - Acquire semantics: Subsequent reads see peer's writes
 * - Uses ld_acquire_sys_global (system-scope acquire load)
 *
 * Data Transfer (put, send, recv):
 * - No implicit ordering; pair with signal for visibility guarantee
 *
 * Example pattern (rank 0 sends to rank 1, 2-rank setup):
 *
 *   // Rank 0 (Sender): target_rank = 1
 *   send(1, group, src, nbytes);
 *   signal_peer(group, 1, 0);
 *
 *   // Rank 1 (Receiver): target_rank = 0
 *   wait_signal(group, 0, CMP_GE, 1);  // data now visible
 *   recv(0, group, dst, nbytes);
 *
 * ARCHITECTURE
 * ============
 *
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │  Host Control Path                                              │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │  MultiPeerNvlTransport (host RAII object)                       │
 *   │  ├── GpuMemHandler[] (data, state, signal, barrier buffers)     │
 *   │  ├── P2pNvlTransportDevice[] (per-peer NVLink handles)          │
 *   │  └── IBootstrap (collective handle exchange)                    │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │  GPU Data Path (returned by getMultiPeerDeviceTransport())      │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │  MultiPeerDeviceTransport                                       │
 *   │  ├── DeviceSpan<Transport> transports_ (indexed by global rank) │
 *   │  │   ├── Transport[0]: SELF or P2P_NVL                          │
 *   │  │   ├── Transport[1]: P2P_NVL                                  │
 *   │  │   └── ...                                                    │
 *   │  ├── DeviceWindowSignal signal_ (inbox model)                         │
 *   │  │   └── signal_peer() / wait_signal()                          │
 *   │  └── DeviceWindowBarrier barrier_ (N-way sync)                        │
 *   │      └── barrier() / barrier_peer()                             │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * USAGE:
 *   // Host setup
 *   MultiPeerNvlTransport hostTransport(myRank, nRanks, bootstrap, config);
 *   WindowMemory wm(myRank, nRanks, bootstrap, wmConfig);
 *   hostTransport.exchange();
 *   wm.exchange();
 *   auto transport = hostTransport.getMultiPeerDeviceTransport(wm);
 *
 *   // Kernel (pass by const reference to avoid copy)
 *   __global__ void myKernel(const MultiPeerDeviceTransport& transport, ...) {
 *     auto group = make_warp_group();
 *
 *     // Data transfer (target_rank in [0, n_ranks()), must not be self)
 *     transport.send(target_rank, group, src, nbytes);
 *
 *     // Signaling (rank-based)
 *     transport.signal_peer(group, target_rank, 0, SIGNAL_ADD, 1);
 *     transport.wait_signal(group, 0, CMP_GE, 1);
 *
 *     // Barrier
 *     transport.barrier(group, 0);
 *
 *     // Local copy via self transport
 *     transport.get_self_transport()->self.put(group, dst, src, nbytes);
 *   }
 */
class MultiPeerDeviceTransport {
 public:
  __host__ __device__ MultiPeerDeviceTransport() = default;

  MultiPeerDeviceTransport(const MultiPeerDeviceTransport&) = default;
  MultiPeerDeviceTransport& operator=(const MultiPeerDeviceTransport&) = delete;

  // Move is allowed
  MultiPeerDeviceTransport(MultiPeerDeviceTransport&&) = default;
  MultiPeerDeviceTransport& operator=(MultiPeerDeviceTransport&&) = delete;

  __host__ __device__ ~MultiPeerDeviceTransport() = default;

  /**
   * Construct a MultiPeerDeviceTransport
   *
   * @param myRank This rank's ID (must be in range [0, nRanks))
   * @param nRanks Total number of ranks
   * @param transports Span of Transport objects for each peer (size = nRanks)
   * @param wm DeviceWindowMemory bundling signal and barrier primitives
   */
  __host__ __device__ MultiPeerDeviceTransport(
      int myRank,
      int nRanks,
      DeviceSpan<Transport> transports,
      DeviceWindowMemory wm)
      : myRank_(myRank),
        nRanks_(nRanks),
        transports_(transports),
        signal_(wm.signal()),
        barrier_(wm.barrier()) {
    // Validate constraints
    assert(nRanks > 0);
    assert(myRank >= 0 && myRank < nRanks);
  }

  // ===========================================================================
  // Metadata
  // ===========================================================================

  __device__ __forceinline__ int rank() const {
    return myRank_;
  }

  __device__ __forceinline__ int n_ranks() const {
    return nRanks_;
  }

  // ===========================================================================
  // Peer Iteration Helpers
  // ===========================================================================

  /**
   * num_peers - Get number of peer connections (excludes self)
   *
   * @return Number of peers (nRanks - 1)
   */
  __host__ __device__ __forceinline__ int num_peers() const {
    return nRanks_ - 1;
  }

  /**
   * peer_index_to_rank - Convert peer index back to global rank
   *
   * When iterating over peers by index (0 to num_peers()-1), this converts
   * the index back to the corresponding global rank, skipping self.
   *
   * Example for myRank=2, nRanks=4:
   *   peer_index_to_rank(0) -> 0
   *   peer_index_to_rank(1) -> 1
   *   peer_index_to_rank(2) -> 3  (skips self at rank 2)
   *
   * @param index Index into peer list (0 to num_peers()-1)
   * @return Global rank of the peer at this index
   */
  __host__ __device__ __forceinline__ int peer_index_to_rank(int index) const {
    // If index < myRank, rank = index; else rank = index + 1 (skip self)
    return (index < myRank_) ? index : (index + 1);
  }

  /**
   * rank_to_peer_index - Convert global rank to peer index
   *
   * Inverse of peer_index_to_rank(). Only valid for rank != myRank.
   *
   * Example for myRank=2, nRanks=4:
   *   rank_to_peer_index(0) -> 0
   *   rank_to_peer_index(1) -> 1
   *   rank_to_peer_index(3) -> 2  (self at rank 2 is skipped)
   *
   * @param rank Global rank (must NOT be myRank)
   * @return Peer index for use with send/recv/signal_peer
   */
  __host__ __device__ __forceinline__ int rank_to_peer_index(int rank) const {
    assert(rank != myRank_ && "Cannot convert self rank to peer index");
    assert(rank >= 0 && rank < nRanks_ && "Rank out of range");
    return (rank < myRank_) ? rank : (rank - 1);
  }

  // ===========================================================================
  // Signal Object (inbox model)
  // ===========================================================================

  __device__ __forceinline__ DeviceWindowSignal& get_signal() {
    return signal_;
  }

  __device__ __forceinline__ const DeviceWindowSignal& get_signal() const {
    return signal_;
  }

  // ===========================================================================
  // Barrier Object
  // ===========================================================================

  __device__ __forceinline__ DeviceWindowBarrier& get_barrier() {
    return barrier_;
  }

  __device__ __forceinline__ const DeviceWindowBarrier& get_barrier() const {
    return barrier_;
  }

  // ===========================================================================
  // Signal Operations (delegated to signal_)
  // ===========================================================================

  /**
   * signal_peer - Signal a specific peer's inbox
   *
   * @param target_rank Global rank in [0, n_ranks()), must not be self
   * @param group ThreadGroup for cooperative processing
   * @param signal_id Signal slot to use
   * @param op Signal operation (SIGNAL_ADD or SIGNAL_SET)
   * @param value Value for the signal operation (default: 1)
   */
  __device__ __forceinline__ void signal_peer(
      ThreadGroup& group,
      int target_rank,
      int signal_id,
      SignalOp op = SignalOp::SIGNAL_ADD,
      uint64_t value = 1) {
    signal_.signal_peer(group, target_rank, signal_id, op, value);
  }

  __device__ __forceinline__ void signal_all(
      ThreadGroup& group,
      int signal_id,
      SignalOp op = SignalOp::SIGNAL_ADD,
      uint64_t value = 1) {
    signal_.signal_all(group, signal_id, op, value);
  }

  __device__ __forceinline__ void wait_signal(
      ThreadGroup& group,
      int signal_id,
      CmpOp cmp,
      uint64_t value,
      uint64_t timeout_ns = UINT64_MAX) {
    signal_.wait_signal(group, signal_id, cmp, value, timeout_ns);
  }

  __device__ __forceinline__ uint64_t read_signal(int signal_id) {
    return signal_.read_signal(signal_id);
  }

  // ===========================================================================
  // Barrier Operations (delegated to barrier_)
  // ===========================================================================

  __device__ __forceinline__ void barrier(
      ThreadGroup& group,
      int barrier_id,
      const Timeout& timeout = Timeout()) {
    barrier_.barrier(group, barrier_id, timeout);
  }

  /**
   * barrier_peer - Two-sided barrier with a specific peer
   *
   * @param target_rank Global rank in [0, n_ranks()), must not be self
   * @param group ThreadGroup for cooperative processing
   * @param barrier_id Barrier slot to use
   * @param timeout Optional timeout (default: no timeout, infinite wait)
   */
  __device__ __forceinline__ void barrier_peer(
      int target_rank,
      ThreadGroup& group,
      int barrier_id,
      const Timeout& timeout = Timeout()) {
    MULTI_PEER_CHECK_RANK(target_rank, nRanks_);
    MULTI_PEER_CHECK_NOT_SELF(target_rank, myRank_);
    int peer_index = rank_to_peer_index(target_rank);
    barrier_.barrier_peer(peer_index, group, barrier_id, timeout);
  }

  /**
   * wait_signal_from - Wait for signal from a specific peer
   *
   * @param group ThreadGroup for cooperative processing
   * @param source_rank Global rank of the source peer in [0, n_ranks()),
   *        must not be self
   * @param signal_id Signal slot to use
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
      uint64_t timeout_ns = UINT64_MAX) {
    signal_.wait_signal_from(
        group, source_rank, signal_id, cmp, value, timeout_ns);
  }

  /**
   * read_signal_from - Read signal value from a specific peer (non-blocking)
   *
   * @param source_rank Global rank of the source peer in [0, n_ranks()),
   *        must not be self
   * @param signal_id Signal slot to use
   * @return Current signal value from the specified peer
   */
  __device__ __forceinline__ uint64_t
  read_signal_from(int source_rank, int signal_id) {
    return signal_.read_signal_from(source_rank, signal_id);
  }

  // ===========================================================================
  // Send/Recv Operations
  // ===========================================================================

  /**
   * send - Send data to a specific peer over NVLink
   *
   * For local copies (self transport), use:
   *   get_self_transport()->self.put(group, dst, src, nbytes)
   *
   * Uses NVLink pipelined staged transfer.
   *
   * @param target_rank Global rank in [0, n_ranks()), must not be self
   * @param group ThreadGroup for cooperative processing
   * @param srcbuff Source buffer (local GPU memory)
   * @param nbytes Number of bytes to send
   * @param call_index Index for multiple concurrent calls (default: 0)
   */
  __device__ __forceinline__ void send(
      int target_rank,
      ThreadGroup& group,
      void* srcbuff,
      std::size_t nbytes,
      uint32_t call_index = 0) {
    DEVICE_WINDOW_SIGNAL_CHECK_RANK(target_rank, nRanks_);
    DEVICE_WINDOW_SIGNAL_CHECK_NOT_SELF(target_rank, myRank_);
    transports_[target_rank].p2p_nvl.send(group, srcbuff, nbytes, call_index);
  }

  /**
   * recv - Receive data from a specific peer over NVLink
   *
   * For local copies (self transport), use:
   *   get_self_transport()->self.put(group, dst, src, nbytes)
   *
   * @param target_rank Global rank in [0, n_ranks()), must not be self
   * @param group ThreadGroup for cooperative processing
   * @param dstbuff Destination buffer (local GPU memory)
   * @param nbytes Number of bytes to receive
   * @param call_index Index for multiple concurrent calls (default: 0)
   */
  __device__ __forceinline__ void recv(
      int target_rank,
      ThreadGroup& group,
      void* dstbuff,
      std::size_t nbytes,
      uint32_t call_index = 0) {
    DEVICE_WINDOW_SIGNAL_CHECK_RANK(target_rank, nRanks_);
    DEVICE_WINDOW_SIGNAL_CHECK_NOT_SELF(target_rank, myRank_);
    transports_[target_rank].p2p_nvl.recv(group, dstbuff, nbytes, call_index);
  }

  // ===========================================================================
  // Zero-Copy Operations
  // ===========================================================================

  /**
   * put - Zero-copy write to peer's memory
   *
   * Writes data directly to the remote buffer via NVLink without staging.
   *
   * Caller is responsible for signaling completion separately.
   * The put operation itself has no implicit memory ordering guarantees.
   *
   * Memory ordering: Caller must use signal_peer() after put() to ensure
   * visibility. The signal operation includes a release fence.
   *
   * @param target_rank Global rank in [0, n_ranks()), must not be self
   * @param group ThreadGroup for cooperative processing
   * @param remoteDst Remote destination buffer (on peer's GPU)
   * @param localSrc Local source buffer
   * @param nbytes Number of bytes to transfer
   */
  __device__ __forceinline__ void put(
      int target_rank,
      ThreadGroup& group,
      void* remoteDst,
      const void* localSrc,
      std::size_t nbytes) {
    MULTI_PEER_CHECK_RANK(target_rank, nRanks_);
    MULTI_PEER_CHECK_NOT_SELF(target_rank, myRank_);
    transports_[target_rank].p2p_nvl.put(
        group,
        static_cast<char*>(remoteDst),
        static_cast<const char*>(localSrc),
        nbytes);
  }

  // ===========================================================================
  // Combined Put + Signal Operations
  // ===========================================================================

  /**
   * put_signal - Zero-copy write with atomic signal
   *
   * Performs put() followed by signal_peer() with proper memory ordering.
   * The signal is guaranteed to be visible only after data transfer completes.
   *
   * Memory ordering: Release semantics - all prior writes visible before
   * signal.
   *
   * @param target_rank Global rank in [0, n_ranks()), must not be self
   * @param group ThreadGroup for cooperative processing
   * @param remoteDst Remote destination buffer
   * @param localSrc Local source buffer
   * @param nbytes Number of bytes to transfer
   * @param signalId Signal slot to increment
   * @param signalVal Value to atomically add (default: 1)
   */
  __device__ __forceinline__ void put_signal(
      int target_rank,
      ThreadGroup& group,
      void* remoteDst,
      const void* localSrc,
      std::size_t nbytes,
      int signalId,
      uint64_t signalVal = 1) {
    // 1. Copy data to remote
    put(target_rank, group, remoteDst, localSrc, nbytes);

    // 2. Ensure all threads complete copy before signaling
    group.sync();

    // 3. Signal with release semantics (leader only in signalPeer)
    signal_peer(group, target_rank, signalId, SignalOp::SIGNAL_ADD, signalVal);
  }

  // ===========================================================================
  // Advanced Access
  // ===========================================================================

  /**
   * get_peer_transport - Get Transport pointer for specific peer
   *
   * @param target_rank Global rank in [0, n_ranks()), must not be self
   * @return Pointer to Transport object for the peer
   */
  __device__ __forceinline__ Transport* get_peer_transport(int target_rank) {
    DEVICE_WINDOW_SIGNAL_CHECK_RANK(target_rank, nRanks_);
    DEVICE_WINDOW_SIGNAL_CHECK_NOT_SELF(target_rank, myRank_);
    return &transports_[target_rank];
  }

  /**
   * get_peer_transport - Const version for read-only access
   *
   * @param target_rank Global rank in [0, n_ranks()), must not be self
   * @return Const pointer to Transport object for the peer
   */
  __device__ __forceinline__ const Transport* get_peer_transport(
      int target_rank) const {
    DEVICE_WINDOW_SIGNAL_CHECK_RANK(target_rank, nRanks_);
    DEVICE_WINDOW_SIGNAL_CHECK_NOT_SELF(target_rank, myRank_);
    return &transports_[target_rank];
  }

  /**
   * get_self_transport - Get Transport pointer for this rank
   *
   * @return Pointer to this rank's own Transport object
   */
  __device__ __forceinline__ Transport* get_self_transport() {
    return &transports_[myRank_];
  }

  /**
   * get_self_transport - Const version for read-only access
   *
   * @return Const pointer to this rank's own Transport object
   */
  __device__ __forceinline__ const Transport* get_self_transport() const {
    return &transports_[myRank_];
  }

 private:
  // ===========================================================================
  // Member Variables
  // ===========================================================================

  const int myRank_{-1};
  const int nRanks_{0};
  const DeviceSpan<Transport> transports_;
  DeviceWindowSignal signal_;
  DeviceWindowBarrier barrier_;
};

} // namespace comms::pipes
