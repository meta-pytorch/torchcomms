// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>
#include <cstdint>
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Transport.cuh"

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
 * Example pattern (rank 0 sends to rank 1):
 *
 * // Rank 0 (Sender): target_rank = 1
 * send(1, group, src, nbytes);
 * signal_peer(1, group, 0);
 *
 * // Rank 1 (Receiver): target_rank = 0
 * wait_signal(group, 0, CMP_GE, 1);  // data now visible
 * recv(0, group, dst, nbytes);
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
 *     // Data transfer (target_rank in [0, n_ranks()), must not be self)
 *     transport.send(target_rank, group, src, nbytes);
 *
 *     // Signaling (rank-based)
 *     transport.signal_peer(target_rank, group, 0, SIGNAL_ADD, 1);
 *     transport.wait_signal(group, 0, CMP_GE, 1);
 *
 *     // Barrier
 *     transport.barrier(group, 0);
 *   }
 */
class MultiPeerDeviceTransport {
 public:
  __host__ __device__ MultiPeerDeviceTransport() = default;

  MultiPeerDeviceTransport(const MultiPeerDeviceTransport&) = delete;
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
   */
  __host__ __device__ MultiPeerDeviceTransport(
      int myRank,
      int nRanks,
      DeviceSpan<Transport> transports)
      : myRank_(myRank), nRanks_(nRanks), transports_(transports) {
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
    MULTI_PEER_CHECK_RANK(target_rank, nRanks_);
    MULTI_PEER_CHECK_NOT_SELF(target_rank, myRank_);
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
    MULTI_PEER_CHECK_RANK(target_rank, nRanks_);
    MULTI_PEER_CHECK_NOT_SELF(target_rank, myRank_);
    transports_[target_rank].p2p_nvl.recv(group, dstbuff, nbytes, call_index);
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
    MULTI_PEER_CHECK_RANK(target_rank, nRanks_);
    MULTI_PEER_CHECK_NOT_SELF(target_rank, myRank_);
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
    MULTI_PEER_CHECK_RANK(target_rank, nRanks_);
    MULTI_PEER_CHECK_NOT_SELF(target_rank, myRank_);
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
};

} // namespace comms::pipes
