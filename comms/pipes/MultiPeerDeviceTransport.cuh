// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>
#include <cstdint>
#include "comms/pipes/DeviceSignal.cuh"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes {

/**
 * MULTI_PEER_CHECK_PEER_INDEX - Bounds checking macro for peer index validation
 *
 * Validates that peer_index is in range [0, num_peers). This macro is used
 * throughout MultiPeerDeviceTransport to catch invalid peer indices early.
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
 *   MULTI_PEER_CHECK_PEER_INDEX(peerIndex, num_peers());
 *
 * TESTED BY:
 * - Valid peer ranges are tested by all functional tests that use peer indices
 * - Invalid peer handling is documented here; device-side __trap() cannot be
 *   unit tested but will produce clear error messages during development
 *
 * TODO(D91689639): Replace with PIPES_DEVICE_CHECK_MSG once D91689639 lands
 */
#ifdef __CUDA_ARCH__
#define MULTI_PEER_CHECK_PEER_INDEX(peer_index, num_peers)                  \
  do {                                                                      \
    if (!((peer_index) >= 0 && (peer_index) < (num_peers))) {               \
      printf(                                                               \
          "MultiPeerDeviceTransport: peer_index %d out of range [0, %d) at" \
          " %s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",                    \
          (int)(peer_index),                                                \
          (int)(num_peers),                                                 \
          __FILE__,                                                         \
          __LINE__,                                                         \
          blockIdx.x,                                                       \
          blockIdx.y,                                                       \
          blockIdx.z,                                                       \
          threadIdx.x,                                                      \
          threadIdx.y,                                                      \
          threadIdx.z);                                                     \
      __trap();                                                             \
    }                                                                       \
  } while (0)
#else
#define MULTI_PEER_CHECK_PEER_INDEX(peer_index, num_peers) \
  assert((peer_index) >= 0 && (peer_index) < (num_peers))
#endif

/**
 * MultiPeerDeviceTransport - Unified device-side multi-peer NVLink transport
 *
 * Provides a single device object with:
 * - DeviceSignal for inbox-style signaling (remote notification)
 * - Peer-indexed send/recv operations
 *
 * PEER INDEX SPACE:
 * All public APIs that target a specific peer accept a peer_index in the
 * range [0, num_peers()), which excludes self. Use peer_index_to_rank()
 * and rank_to_peer_index() to convert between peer index and global rank.
 *
 * For local (self) operations, use get_self_transport() to access the
 * self transport directly.
 *
 * DESIGN:
 * - Aligned with TorchComm Device API style (D91172575)
 * - Signal/Barrier use inbox model (one inbox per rank)
 * - Self-rank operations handled transparently
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
 *   // Rank 0 (Sender): peer_index 0 maps to rank 1
 *   send(0, group, src, nbytes);
 *   signal_peer(0, group, 0);
 *
 *   // Rank 1 (Receiver): peer_index 0 maps to rank 0
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
 *   │  ├── DeviceSpan<Transport> transports_ (indexed by peer rank)   │
 *   │  │   ├── Transport[0]: SELF or P2P_NVL                          │
 *   │  │   ├── Transport[1]: P2P_NVL                                  │
 *   │  │   └── ...                                                    │
 *   │  ├── DeviceSignal signal_ (inbox model)                         │
 *   │  │   └── signal_peer() / wait_signal()                          │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * USAGE:
 *   // Host setup
 *   MultiPeerNvlTransport hostTransport(myRank, nRanks, bootstrap, config);
 *   hostTransport.exchange();
 *   auto transport = hostTransport.getMultiPeerDeviceTransport();
 *
 *   // Kernel (pass by const reference to avoid copy)
 *   __global__ void myKernel(const MultiPeerDeviceTransport& transport, ...) {
 *     auto group = make_warp_group();
 *
 *     // Data transfer (peer_index in [0, num_peers()))
 *     transport.send(peer_index, group, src, nbytes);
 *
 *     // Signaling (peer_index based)
 *     transport.signal_peer(peer_index, group, 0, SIGNAL_ADD, 1);
 *     transport.wait_signal(group, 0, CMP_GE, 1);
 *
 *     // Barrier
 *     transport.barrier(group, 0);
 *
 *     // Local copy via self transport
 *     transport.get_self_transport()->self.put(group, dst, src, nbytes);
 *   }
 *
 * SELF-TRANSPORT LIMITATIONS:
 * send(myRank, ...) and recv(myRank, ...) are NOT supported and will trap.
 * For local copies, use: get_transport(myRank)->self.put(group, dst, src, n)
 */
class MultiPeerDeviceTransport {
 public:
  __host__ __device__ MultiPeerDeviceTransport() = default;

  // Copy is allowed (shallow copy of DeviceSpan)
  MultiPeerDeviceTransport(const MultiPeerDeviceTransport&) = default;
  MultiPeerDeviceTransport& operator=(const MultiPeerDeviceTransport&) = delete;

  // Move is allowed
  MultiPeerDeviceTransport(MultiPeerDeviceTransport&&) = default;
  MultiPeerDeviceTransport& operator=(MultiPeerDeviceTransport&&) = delete;

  /**
   * Construct a MultiPeerDeviceTransport
   *
   * @param myRank This rank's ID (must be in range [0, nRanks))
   * @param nRanks Total number of ranks
   * @param transports Span of Transport objects for each peer (size = nRanks)
   * @param signal DeviceSignal for inbox-style signaling
   */
  __host__ __device__ MultiPeerDeviceTransport(
      int myRank,
      int nRanks,
      DeviceSpan<Transport> transports,
      DeviceSignal signal)
      : myRank_(myRank),
        nRanks_(nRanks),
        transports_(transports),
        signal_(signal) {
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
   * Inverse of peer_index_to_rank(). Useful when the caller has a global rank
   * and needs to call a peer-indexed API.
   *
   * @param rank Global rank (must NOT be myRank)
   * @return Peer index in [0, num_peers())
   */
  __host__ __device__ __forceinline__ int rank_to_peer_index(int rank) const {
    assert(rank != myRank_);
    return (rank < myRank_) ? rank : (rank - 1);
  }

  // ===========================================================================
  // Signal Object (inbox model)
  // ===========================================================================

  __device__ __forceinline__ DeviceSignal& get_signal() {
    return signal_;
  }

  __device__ __forceinline__ const DeviceSignal& get_signal() const {
    return signal_;
  }

  // ===========================================================================
  // Signal Operations (delegated to signal_)
  // ===========================================================================

  /**
   * send - Send data to a specific peer over NVLink
   * signal_peer - Signal a specific peer's inbox
   *
   * @param peer_index Peer index in [0, num_peers())
   * @param group ThreadGroup for cooperative processing
   * @param signal_id Signal slot to use
   * @param op Signal operation (SIGNAL_ADD or SIGNAL_SET)
   * @param value Value for the signal operation (default: 1)
   */
  __device__ __forceinline__ void signal_peer(
      int peer_index,
      ThreadGroup& group,
      int signal_id,
      SignalOp op = SignalOp::SIGNAL_ADD,
      uint64_t value = 1) {
    int rank = peer_index_to_rank(peer_index);
    signal_.signal_peer(rank, group, signal_id, op, value);
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
  // Send/Recv Operations (peer index as input)
  // ===========================================================================

  /**
   * send - Send data to a specific peer over NVLink
   *
   * Uses NVLink pipelined staged transfer.
   *
   * @param peer_index Peer index in [0, num_peers())
   * @param group ThreadGroup for cooperative processing
   * @param srcbuff Source buffer (local GPU memory)
   * @param nbytes Number of bytes to send
   * @param call_index Index for multiple concurrent calls (default: 0)
   */
  __device__ __forceinline__ void send(
      int peer_index,
      ThreadGroup& group,
      void* srcbuff,
      std::size_t nbytes,
      uint32_t call_index = 0) {
    MULTI_PEER_CHECK_PEER_INDEX(peer_index, num_peers());
    int rank = peer_index_to_rank(peer_index);
    transports_[rank].p2p_nvl.send(group, srcbuff, nbytes, call_index);
  }

  /**
   * recv - Receive data from a specific peer over NVLink
   *
   * Uses NVLink pipelined staged transfer.
   *
   * @param peer_index Peer index in [0, num_peers())
   * @param group ThreadGroup for cooperative processing
   * @param dstbuff Destination buffer (local GPU memory)
   * @param nbytes Number of bytes to receive
   * @param call_index Index for multiple concurrent calls (default: 0)
   */
  __device__ __forceinline__ void recv(
      int peer_index,
      ThreadGroup& group,
      void* dstbuff,
      std::size_t nbytes,
      uint32_t call_index = 0) {
    MULTI_PEER_CHECK_PEER_INDEX(peer_index, num_peers());
    int rank = peer_index_to_rank(peer_index);
    transports_[rank].p2p_nvl.recv(group, dstbuff, nbytes, call_index);
  }

  // ===========================================================================
  // Advanced Access
  // ===========================================================================

  /**
   * get_peer_transport - Get Transport pointer for specific peer
   *
   * For advanced use cases or custom communication patterns.
   *
   * @param peer_index Peer index in [0, num_peers())
   * @return Pointer to Transport object for the peer
   */
  __device__ __forceinline__ Transport* get_peer_transport(int peer_index) {
    MULTI_PEER_CHECK_PEER_INDEX(peer_index, num_peers());
    int rank = peer_index_to_rank(peer_index);
    return &transports_[rank];
  }

  /**
   * get_peer_transport - Const version for read-only access
   *
   * @param peer_index Peer index in [0, num_peers())
   * @return Const pointer to Transport object for the peer
   */
  __device__ __forceinline__ const Transport* get_peer_transport(
      int peer_index) const {
    MULTI_PEER_CHECK_PEER_INDEX(peer_index, num_peers());
    int rank = peer_index_to_rank(peer_index);
    return &transports_[rank];
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
  // Private Peer Index Helpers
  // ===========================================================================

  /**
   * get_peer_by_index - Get transport by peer index (not rank)
   *
   * Direct access to peer transports by index. Used internally for
   * iterating over all peers without needing to skip self manually.
   *
   * @param index Index into peer list (0 to num_peers()-1)
   * @return Pointer to Transport object for the peer at this index
   */
  __device__ __forceinline__ Transport* get_peer_by_index(int index) {
    int peerRank = peer_index_to_rank(index);
    return &transports_[peerRank];
  }

  /**
   * get_peer_by_index - Const version for read-only access
   */
  __device__ __forceinline__ const Transport* get_peer_by_index(
      int index) const {
    int peerRank = peer_index_to_rank(index);
    return &transports_[peerRank];
  }

  // ===========================================================================
  // Member Variables
  // ===========================================================================

  const int myRank_{-1};
  const int nRanks_{0};
  const DeviceSpan<Transport> transports_;
  DeviceSignal signal_;
};

} // namespace comms::pipes
