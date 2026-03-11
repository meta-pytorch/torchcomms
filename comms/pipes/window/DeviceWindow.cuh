// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>
#include <cstdint>
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/SignalState.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/Transport.cuh"

#ifdef __CUDACC__
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#else
namespace comms::pipes {
class P2pIbgdaTransportDevice;
} // namespace comms::pipes
#endif

namespace comms::pipes {

// Forward declaration for test helper
namespace test {
struct NvlOnlyDeviceWindowBuffers;
struct IbgdaOnlyDeviceWindowBuffers;
} // namespace test

// ===========================================================================
// Buffer Registration Types (for DeviceWindow generic put)
// ===========================================================================

struct LocalBufferRegistration {
  const void* base{nullptr};
  std::size_t size{0};
  NetworkLKey lkey{};
};

struct RemoteBufferRegistration {
  const void* base{nullptr};
  std::size_t size{0};
  NetworkRKey rkey{};
};

// Bounds checking macros for device code
#ifdef __CUDA_ARCH__
#define DEVICE_WINDOW_CHECK_RANK(target_rank, nRanks)         \
  do {                                                        \
    if (!((target_rank) >= 0 && (target_rank) < (nRanks))) {  \
      printf(                                                 \
          "DeviceWindow: target_rank %d out of range [0, %d)" \
          " at %s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",   \
          (int)(target_rank),                                 \
          (int)(nRanks),                                      \
          __FILE__,                                           \
          __LINE__,                                           \
          blockIdx.x,                                         \
          blockIdx.y,                                         \
          blockIdx.z,                                         \
          threadIdx.x,                                        \
          threadIdx.y,                                        \
          threadIdx.z);                                       \
      __trap();                                               \
    }                                                         \
  } while (0)
#define DEVICE_WINDOW_CHECK_NOT_SELF(target_rank, myRank) \
  do {                                                    \
    if ((target_rank) == (myRank)) {                      \
      printf(                                             \
          "DeviceWindow: self-rank %d not supported at "  \
          "%s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",   \
          (int)(target_rank),                             \
          __FILE__,                                       \
          __LINE__,                                       \
          blockIdx.x,                                     \
          blockIdx.y,                                     \
          blockIdx.z,                                     \
          threadIdx.x,                                    \
          threadIdx.y,                                    \
          threadIdx.z);                                   \
      __trap();                                           \
    }                                                     \
  } while (0)
#define DEVICE_WINDOW_CHECK_IBGDA_SIGNAL_ADD(op)                 \
  do {                                                           \
    if ((op) != SignalOp::SIGNAL_ADD) {                          \
      printf(                                                    \
          "DeviceWindow: IBGDA only supports SIGNAL_ADD, got %d" \
          " at %s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",      \
          (int)(op),                                             \
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
#define DEVICE_WINDOW_CHECK_SIGNAL_ID(signal_id, count)     \
  do {                                                      \
    if (!((signal_id) >= 0 && (signal_id) < (count))) {     \
      printf(                                               \
          "DeviceWindow: signal_id %d out of range [0, %d)" \
          " at %s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n", \
          (int)(signal_id),                                 \
          (int)(count),                                     \
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
#define DEVICE_WINDOW_CHECK_BARRIER_ID(barrier_id, count)    \
  do {                                                       \
    if (!((barrier_id) >= 0 && (barrier_id) < (count))) {    \
      printf(                                                \
          "DeviceWindow: barrier_id %d out of range [0, %d)" \
          " at %s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",  \
          (int)(barrier_id),                                 \
          (int)(count),                                      \
          __FILE__,                                          \
          __LINE__,                                          \
          blockIdx.x,                                        \
          blockIdx.y,                                        \
          blockIdx.z,                                        \
          threadIdx.x,                                       \
          threadIdx.y,                                       \
          threadIdx.z);                                      \
      __trap();                                              \
    }                                                        \
  } while (0)
#define DEVICE_WINDOW_CHECK_IBGDA_PEER(ibgda_idx, rank)     \
  do {                                                      \
    if ((ibgda_idx) < 0) {                                  \
      printf(                                               \
          "DeviceWindow: rank %d is not an IBGDA peer"      \
          " at %s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n", \
          (int)(rank),                                      \
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
#else
#define DEVICE_WINDOW_CHECK_RANK(target_rank, nRanks) \
  assert((target_rank) >= 0 && (target_rank) < (nRanks))
#define DEVICE_WINDOW_CHECK_NOT_SELF(target_rank, myRank) \
  assert((target_rank) != (myRank))
#define DEVICE_WINDOW_CHECK_IBGDA_SIGNAL_ADD(op) \
  assert((op) == SignalOp::SIGNAL_ADD)
#define DEVICE_WINDOW_CHECK_SIGNAL_ID(signal_id, count) \
  assert((signal_id) >= 0 && (signal_id) < (count))
#define DEVICE_WINDOW_CHECK_BARRIER_ID(barrier_id, count) \
  assert((barrier_id) >= 0 && (barrier_id) < (count))
#define DEVICE_WINDOW_CHECK_IBGDA_PEER(ibgda_idx, rank) assert((ibgda_idx) >= 0)
#endif

// NVL peer type check — validates that target rank is an NVL peer before
// accessing the p2p_nvl union member. Calling get_nvl() on a rank whose
// transport type is not P2P_NVL (e.g., an IBGDA-only peer when both ranks
// share the same GPU) reads the wrong union member, producing garbage
// pointers and illegal memory accesses.
#ifdef __CUDA_ARCH__
#define DEVICE_WINDOW_CHECK_NVL_PEER(handle, rank)             \
  do {                                                         \
    if ((handle).get_type(rank) != TransportType::P2P_NVL) {   \
      printf(                                                  \
          "DeviceWindow: rank %d is not an NVL peer (type=%d)" \
          " at %s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",    \
          (int)(rank),                                         \
          (int)(handle).get_type(rank),                        \
          __FILE__,                                            \
          __LINE__,                                            \
          blockIdx.x,                                          \
          blockIdx.y,                                          \
          blockIdx.z,                                          \
          threadIdx.x,                                         \
          threadIdx.y,                                         \
          threadIdx.z);                                        \
      __trap();                                                \
    }                                                          \
  } while (0)
#else
// Host fallback is no-op: get_nvl() is only available under __CUDACC__,
// so this check can never be reached from host code.
#define DEVICE_WINDOW_CHECK_NVL_PEER(handle, rank) ((void)0)
#endif

/**
 * DeviceWindow - Unified device-side window for data + signal + barrier
 *
 * All signal, barrier, and counter state is held directly — no separate
 * DeviceWindowSignal or DeviceWindowBarrier sub-objects. Transport dispatch
 * uses MultiPeerDeviceHandle::get_type(rank) and pre-computed peer index
 * maps for O(1) rank-to-peer-index lookup.
 *
 * CONSTRUCTION:
 *   // Host side
 *   MultiPeerTransport transport(myRank, nRanks, deviceId, bootstrap, config);
 *   transport.exchange();
 *   HostWindow window(transport, windowConfig);
 *   window.exchange();
 *   window.registerBuffer(myBuf, bufSize);  // collective, for put/put_signal
 *   DeviceWindow dw = window.getDeviceWindow();
 *
 *   // Kernel
 *   __global__ void myKernel(DeviceWindow dw, ...) {
 *     auto group = make_warp_group();
 *     dw.signal_peer(target_rank, signal_id);  // thread-level, no group
 *     dw.wait_signal(group, signal_id, CmpOp::CMP_GE, nPeers);
 *     dw.barrier(group, barrier_id);
 *   }
 *
 * DATA TRANSFER APIS:
 * - put/put_signal: Generic one-sided write, dispatches to NVL or IBGDA
 *   internally. Buffers must be registered via HostWindow::registerBuffer().
 * - send/recv are NOT on DeviceWindow — use get_handle().get_nvl(rank)
 *   directly for two-sided operations.
 *
 * TRANSPORT ACCESS: get_nvl(rank), get_ibgda(rank), get_type(rank)
 *
 * BARRIER SEMANTICS:
 * Each copy of DeviceWindow has its own barrierExpected_ counter.
 * When passed by value to kernels, each thread block gets an independent
 * copy. Barriers work per-block: each block must use distinct barrier_id
 * slots to avoid cross-block interference.
 */
class DeviceWindow {
 public:
  __host__ __device__ DeviceWindow() = default;

  DeviceWindow(const DeviceWindow&) = default;
  DeviceWindow& operator=(const DeviceWindow&) = delete;
  DeviceWindow(DeviceWindow&&) = default;
  DeviceWindow& operator=(DeviceWindow&&) = delete;
  __host__ __device__ ~DeviceWindow() = default;

  // ===========================================================================
  // Metadata
  // ===========================================================================

  __device__ __forceinline__ int rank() const {
    return handle_.myRank;
  }

  __device__ __forceinline__ int n_ranks() const {
    return handle_.nRanks;
  }

  __device__ __forceinline__ int num_peers() const {
    return handle_.nRanks - 1;
  }

  __device__ __forceinline__ int num_nvl_peers() const {
    return nNvlPeers_;
  }

  __device__ __forceinline__ int num_ibgda_peers() const {
    return nIbgdaPeers_;
  }

  // ===========================================================================
  // Peer Iteration Helpers
  // ===========================================================================

  __host__ __device__ __forceinline__ int peer_index_to_rank(int index) const {
    return (index < handle_.myRank) ? index : (index + 1);
  }

  __host__ __device__ __forceinline__ int rank_to_peer_index(int r) const {
    assert(r != handle_.myRank && "Cannot convert self rank to peer index");
    return (r < handle_.myRank) ? r : (r - 1);
  }

  // ===========================================================================
  // Transport Access
  // ===========================================================================

#ifdef __CUDACC__
  __device__ __forceinline__ TransportType get_type(int r) const {
    return handle_.get_type(r);
  }

  __device__ __forceinline__ P2pNvlTransportDevice& get_nvl(int r) {
    DEVICE_WINDOW_CHECK_NVL_PEER(handle_, r);
    return handle_.get_nvl(r);
  }

  __device__ __forceinline__ const P2pNvlTransportDevice& get_nvl(int r) const {
    DEVICE_WINDOW_CHECK_NVL_PEER(handle_, r);
    return handle_.get_nvl(r);
  }

  __device__ __forceinline__ P2pIbgdaTransportDevice& get_ibgda(int r) {
    return handle_.get_ibgda(r);
  }

  __device__ __forceinline__ const P2pIbgdaTransportDevice& get_ibgda(
      int r) const {
    return handle_.get_ibgda(r);
  }
#endif

  // ===========================================================================
  // Signal Operations
  // ===========================================================================

#ifdef __CUDACC__
  /**
   * signal_peer - Signal a specific peer on a given signal slot.
   *
   * Thread-level API: any single thread may call this independently.
   * There is no internal synchronization — the caller is responsible for
   * ensuring ordering (e.g., calling group.sync() before signaling if
   * the signal must be visible after a preceding data transfer).
   *
   * For NVL peers: performs an atomic write to the peer's signal inbox
   * via GPU load/store (NVLink).
   * For IBGDA peers: posts an RDMA atomic fetch-add to the peer's
   * signal inbox via the NIC.
   *
   * @param target_rank  Rank to signal (must not be self).
   * @param signal_id    Signal slot index in [0, peerSignalCount).
   * @param op           Signal operation (default: SIGNAL_ADD).
   *                     IBGDA peers only support SIGNAL_ADD.
   * @param value        Value to add/set (default: 1).
   */
  __device__ __forceinline__ void signal_peer(
      int target_rank,
      int signal_id,
      SignalOp op = SignalOp::SIGNAL_ADD,
      uint64_t value = 1) {
    DEVICE_WINDOW_CHECK_RANK(target_rank, handle_.nRanks);
    DEVICE_WINDOW_CHECK_NOT_SELF(target_rank, handle_.myRank);
    DEVICE_WINDOW_CHECK_SIGNAL_ID(signal_id, peerSignalCount_);
    if (handle_.get_type(target_rank) == TransportType::P2P_NVL) {
      int nvlIdx = rankToNvlPeerIndex_[target_rank];
      nvlPeerSignalSpans_[nvlIdx][signal_id].signal(op, value);
    } else {
      DEVICE_WINDOW_CHECK_IBGDA_SIGNAL_ADD(op);
      int ibgdaIdx = rank_to_peer_index(target_rank);
      // Remote buffer is pre-offset to "my row" in the peer's inbox
      // (computed once at exchange time in HostWindow), so signal_id
      // is the only offset needed here.
      handle_.get_ibgda(target_rank)
          .signal_remote(
              ibgdaPeerSignalRemoteBufs_[ibgdaIdx], signal_id, value);
    }
  }

  /**
   * signal_peer (group overload) - Signal a specific peer with
   *                                group synchronization.
   *
   * Group-level API: all threads in the group must call this together.
   * Performs group.sync() for ordering, then the global leader executes
   * signal_peer(). Use this when the signal must be ordered after a
   * preceding group data transfer (e.g., put + sync + signal).
   *
   * @param group        ThreadGroup for group coordination.
   * @param target_rank  Rank to signal (must not be self).
   * @param signal_id    Signal slot index in [0, peerSignalCount).
   * @param op           Signal operation (default: SIGNAL_ADD).
   * @param value        Value to add/set (default: 1).
   */
  __device__ __forceinline__ void signal_peer(
      ThreadGroup& group,
      int target_rank,
      int signal_id,
      SignalOp op = SignalOp::SIGNAL_ADD,
      uint64_t value = 1) {
    group.sync();
    if (group.is_global_leader()) {
      signal_peer(target_rank, signal_id, op, value);
    }
  }

  /**
   * signal_all - Signal all peers on a given signal slot.
   *
   * Group-level API: all threads in all groups must call this together.
   * Peers are horizontally partitioned across all thread groups to avoid
   * duplicate signaling — each peer is signaled by exactly one thread.
   * Contains internal group.sync() barriers for ordering.
   *
   * @param group      ThreadGroup for group coordination.
   * @param signal_id  Signal slot index in [0, peerSignalCount).
   * @param op         Signal operation (default: SIGNAL_ADD).
   *                   IBGDA peers only support SIGNAL_ADD.
   * @param value      Value to add/set per peer (default: 1).
   */
  __device__ __forceinline__ void signal_all(
      ThreadGroup& group,
      int signal_id,
      SignalOp op = SignalOp::SIGNAL_ADD,
      uint64_t value = 1) {
    DEVICE_WINDOW_CHECK_SIGNAL_ID(signal_id, peerSignalCount_);
    group.sync();
    int nPeers = num_peers();
    int globalThreadIdx = static_cast<int>(
        group.group_id * group.group_size + group.thread_id_in_group);
    int totalThreads = static_cast<int>(group.total_groups * group.group_size);
    for (int peer_index = globalThreadIdx; peer_index < nPeers;
         peer_index += totalThreads) {
      int r = peer_index_to_rank(peer_index);
      if (handle_.get_type(r) == TransportType::P2P_NVL) {
        int nvlIdx = rankToNvlPeerIndex_[r];
        nvlPeerSignalSpans_[nvlIdx][signal_id].signal(op, value);
      } else {
        DEVICE_WINDOW_CHECK_IBGDA_SIGNAL_ADD(op);
        handle_.get_ibgda(r).signal_remote(
            ibgdaPeerSignalRemoteBufs_[peer_index], signal_id, value);
      }
    }
    group.sync();
  }

  /**
   * wait_signal_from - Wait for a specific peer's signal to satisfy a
   *                     comparison.
   *
   * Thread-level API: any single thread may call this independently.
   * The caller spins on the inbox slot until the comparison is satisfied
   * or the timeout expires.
   *
   * @param source_rank  Rank to wait on (must not be self).
   * @param signal_id    Signal slot index in [0, peerSignalCount).
   * @param cmp          Comparison operator (CMP_GE, CMP_EQ, etc.).
   * @param value        Threshold value for comparison.
   * @param timeout      Optional timeout (traps on expiry).
   */
  __device__ __forceinline__ void wait_signal_from(
      int source_rank,
      int signal_id,
      CmpOp cmp,
      uint64_t value,
      const Timeout& timeout = Timeout()) {
    DEVICE_WINDOW_CHECK_RANK(source_rank, handle_.nRanks);
    DEVICE_WINDOW_CHECK_NOT_SELF(source_rank, handle_.myRank);
    DEVICE_WINDOW_CHECK_SIGNAL_ID(signal_id, peerSignalCount_);
    if (handle_.get_type(source_rank) == TransportType::P2P_NVL) {
      int nvlIdx = rankToNvlPeerIndex_[source_rank];
      int slot = nvlIdx * peerSignalCount_ + signal_id;
      while (!compare(nvlPeerSignalInbox_[slot].load(), cmp, value)) {
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "DeviceWindow::wait_signal_from(source_rank=%d,"
            " signal_id=%d, value=%llu) rank=%d",
            source_rank,
            signal_id,
            static_cast<unsigned long long>(value),
            handle_.myRank);
      }
    } else {
      int ibgdaIdx = rank_to_peer_index(source_rank);
      int slot = ibgdaIdx * peerSignalCount_ + signal_id;
      // volatile: bypass L1 to read from L2 where RDMA atomics land
      volatile uint64_t* sig = &ibgdaPeerSignalInbox_[slot];
      while (!compare(*sig, cmp, value)) {
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "DeviceWindow::wait_signal_from(source_rank=%d,"
            " signal_id=%d, value=%llu) rank=%d",
            source_rank,
            signal_id,
            static_cast<unsigned long long>(value),
            handle_.myRank);
      }
    }
  }

  /**
   * wait_signal_from (group overload) - Wait for a specific peer's
   *                                     signal with group coordination.
   *
   * Group-level API: all threads in the group must call this together.
   * Only the group leader polls; other threads block at the trailing
   * group.sync().
   *
   * @param group        ThreadGroup for group coordination.
   * @param source_rank  Rank to wait on (must not be self).
   * @param signal_id    Signal slot index in [0, peerSignalCount).
   * @param cmp          Comparison operator (CMP_GE, CMP_EQ, etc.).
   * @param value        Threshold value for comparison.
   * @param timeout      Optional timeout (traps on expiry).
   */
  __device__ __forceinline__ void wait_signal_from(
      ThreadGroup& group,
      int source_rank,
      int signal_id,
      CmpOp cmp,
      uint64_t value,
      const Timeout& timeout = Timeout()) {
    if (group.is_leader()) {
      wait_signal_from(source_rank, signal_id, cmp, value, timeout);
    }
    group.sync();
  }

  /**
   * wait_signal - Wait for the aggregate signal across all peers to
   *               satisfy a comparison.
   *
   * Group-level API: all threads in the group must call this together.
   * Only the group leader polls; it sums all NVL + IBGDA inbox slots
   * for signal_id and checks the total against (cmp, value).
   * Other threads block at the trailing group.sync().
   *
   * @param group      ThreadGroup for group coordination.
   * @param signal_id  Signal slot index in [0, peerSignalCount).
   * @param cmp        Comparison operator (CMP_GE, CMP_EQ, etc.).
   * @param value      Threshold value for the aggregate sum.
   * @param timeout    Optional timeout (traps on expiry).
   */
  __device__ __forceinline__ void wait_signal(
      ThreadGroup& group,
      int signal_id,
      CmpOp cmp,
      uint64_t value,
      const Timeout& timeout = Timeout()) {
    DEVICE_WINDOW_CHECK_SIGNAL_ID(signal_id, peerSignalCount_);
    if (group.is_leader()) {
      int nPeers = num_peers();
      while (true) {
        uint64_t total = 0;
        for (int peer_index = 0; peer_index < nPeers; ++peer_index) {
          int r = peer_index_to_rank(peer_index);
          if (handle_.get_type(r) == TransportType::P2P_NVL) {
            int nvlIdx = rankToNvlPeerIndex_[r];
            total += nvlPeerSignalInbox_[nvlIdx * peerSignalCount_ + signal_id]
                         .load();
          } else {
            // volatile: bypass L1 to read from L2 where RDMA atomics land
            volatile uint64_t* p =
                &ibgdaPeerSignalInbox_
                    [peer_index * peerSignalCount_ + signal_id];
            total += *p;
          }
        }
        if (compare(total, cmp, value)) {
          break;
        }
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "DeviceWindow::wait_signal(signal_id=%d, value=%llu)"
            " rank=%d",
            signal_id,
            static_cast<unsigned long long>(value),
            handle_.myRank);
      }
    }
    group.sync();
  }

  /**
   * read_signal - Non-blocking read of the aggregate signal across all
   *               peers.
   *
   * Thread-level API: any single thread may call this independently.
   * Returns the sum of all NVL + IBGDA inbox values for signal_id.
   *
   * @param signal_id  Signal slot index in [0, peerSignalCount).
   * @return           Aggregate signal value.
   */
  __device__ __forceinline__ uint64_t read_signal(int signal_id) {
    DEVICE_WINDOW_CHECK_SIGNAL_ID(signal_id, peerSignalCount_);
    uint64_t total = 0;
    int nPeers = num_peers();
    for (int peer_index = 0; peer_index < nPeers; ++peer_index) {
      int r = peer_index_to_rank(peer_index);
      if (handle_.get_type(r) == TransportType::P2P_NVL) {
        int nvlIdx = rankToNvlPeerIndex_[r];
        total +=
            nvlPeerSignalInbox_[nvlIdx * peerSignalCount_ + signal_id].load();
      } else {
        // volatile: bypass L1 to read from L2 where RDMA atomics land
        volatile uint64_t* p =
            &ibgdaPeerSignalInbox_[peer_index * peerSignalCount_ + signal_id];
        total += *p;
      }
    }
    return total;
  }

  /**
   * read_signal_from - Non-blocking read of a specific peer's signal.
   *
   * Thread-level API: any single thread may call this independently.
   *
   * @param source_rank  Rank to read from (must not be self).
   * @param signal_id    Signal slot index in [0, peerSignalCount).
   * @return             Signal value from the specified peer.
   */
  __device__ __forceinline__ uint64_t
  read_signal_from(int source_rank, int signal_id) {
    DEVICE_WINDOW_CHECK_RANK(source_rank, handle_.nRanks);
    DEVICE_WINDOW_CHECK_NOT_SELF(source_rank, handle_.myRank);
    DEVICE_WINDOW_CHECK_SIGNAL_ID(signal_id, peerSignalCount_);
    if (handle_.get_type(source_rank) == TransportType::P2P_NVL) {
      int nvlIdx = rankToNvlPeerIndex_[source_rank];
      return nvlPeerSignalInbox_[nvlIdx * peerSignalCount_ + signal_id].load();
    }
    int ibgdaIdx = rank_to_peer_index(source_rank);
    // volatile: bypass L1 to read from L2 where RDMA atomics land
    volatile uint64_t* p =
        &ibgdaPeerSignalInbox_[ibgdaIdx * peerSignalCount_ + signal_id];
    return *p;
  }

  // ===========================================================================
  // Counter Operations (IBGDA-only)
  // ===========================================================================

  /**
   * wait_counter - Wait for an IBGDA peer's NIC completion counter to
   *                satisfy a comparison.
   *
   * Thread-level API: any single thread may call this independently.
   * The counter buffer is written by the NIC via companion-QP loopback
   * RDMA atomics, tracking data-transfer completions.
   *
   * @param peer_rank   IBGDA peer rank (traps if not an IBGDA peer).
   * @param counter_id  Counter slot index.
   * @param cmp         Comparison operator.
   * @param value       Threshold value for comparison.
   * @param timeout     Optional timeout (traps on expiry).
   */
  __device__ __forceinline__ void wait_counter(
      int peer_rank,
      int counter_id,
      CmpOp cmp,
      uint64_t value,
      const Timeout& timeout = Timeout()) {
    DEVICE_WINDOW_CHECK_RANK(peer_rank, handle_.nRanks);
    DEVICE_WINDOW_CHECK_NOT_SELF(peer_rank, handle_.myRank);
    // Counter operations are IBGDA-only; no-op for NVL peers
    if (handle_.get_type(peer_rank) == TransportType::P2P_NVL) {
      return;
    }
    int ibgdaIdx = rank_to_peer_index(peer_rank);
    int slot = ibgdaIdx * peerCounterCount_ + counter_id;
    // volatile: bypass L1 to read from L2 where RDMA atomics land
    volatile uint64_t* ctr = &ibgdaPeerCounterBuf_[slot];
    while (!compare(*ctr, cmp, value)) {
      TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
          timeout,
          "DeviceWindow::wait_counter(peer_rank=%d,"
          " counter_id=%d, value=%llu) rank=%d",
          peer_rank,
          counter_id,
          static_cast<unsigned long long>(value),
          handle_.myRank);
    }
  }

  /**
   * wait_counter (group overload) - Wait for an IBGDA peer's NIC
   *                                 completion counter with group
   *                                 coordination.
   *
   * Group-level API: all threads in the group must call this together.
   * Only the group leader polls; other threads block at the trailing
   * group.sync().
   *
   * @param group       ThreadGroup for group coordination.
   * @param peer_rank   IBGDA peer rank (traps if not an IBGDA peer).
   * @param counter_id  Counter slot index.
   * @param cmp         Comparison operator.
   * @param value       Threshold value for comparison.
   * @param timeout     Optional timeout (traps on expiry).
   */
  __device__ __forceinline__ void wait_counter(
      ThreadGroup& group,
      int peer_rank,
      int counter_id,
      CmpOp cmp,
      uint64_t value,
      const Timeout& timeout = Timeout()) {
    if (group.is_leader()) {
      wait_counter(peer_rank, counter_id, cmp, value, timeout);
    }
    group.sync();
  }

  /**
   * read_counter - Non-blocking read of an IBGDA peer's NIC completion
   *                counter.
   *
   * Thread-level API: any single thread may call this independently.
   *
   * @param peer_rank   IBGDA peer rank (traps if not an IBGDA peer).
   * @param counter_id  Counter slot index.
   * @return            Current counter value.
   */
  __device__ __forceinline__ uint64_t
  read_counter(int peer_rank, int counter_id) {
    DEVICE_WINDOW_CHECK_RANK(peer_rank, handle_.nRanks);
    DEVICE_WINDOW_CHECK_NOT_SELF(peer_rank, handle_.myRank);
    // Counter operations are IBGDA-only; return 0 for NVL peers
    if (handle_.get_type(peer_rank) == TransportType::P2P_NVL) {
      return 0;
    }
    int ibgdaIdx = rank_to_peer_index(peer_rank);
    // volatile: bypass L1 to read from L2 where RDMA atomics land
    volatile uint64_t* ctr =
        &ibgdaPeerCounterBuf_[ibgdaIdx * peerCounterCount_ + counter_id];
    return *ctr;
  }

  /**
   * reset_counter - Reset an IBGDA peer's NIC completion counter to 0.
   *
   * Thread-level API: any single thread may call this independently.
   *
   * @param peer_rank   IBGDA peer rank (traps if not an IBGDA peer).
   * @param counter_id  Counter slot index.
   */
  __device__ __forceinline__ void reset_counter(int peer_rank, int counter_id) {
    DEVICE_WINDOW_CHECK_RANK(peer_rank, handle_.nRanks);
    DEVICE_WINDOW_CHECK_NOT_SELF(peer_rank, handle_.myRank);
    // Counter operations are IBGDA-only; no-op for NVL peers
    if (handle_.get_type(peer_rank) == TransportType::P2P_NVL) {
      return;
    }
    int ibgdaIdx = rank_to_peer_index(peer_rank);
    // volatile: ensure the store is not optimized away by the compiler
    volatile uint64_t* ctr =
        &ibgdaPeerCounterBuf_[ibgdaIdx * peerCounterCount_ + counter_id];
    *ctr = 0;
  }

  // ===========================================================================
  // Barrier Operations
  // ===========================================================================

  /**
   * barrier - Full barrier across all peers on a given barrier slot.
   *
   * Group-level API: all threads in all groups must call this together.
   * Signals all peers, increments the expected count, then waits for
   * all peers to signal back. Each copy of DeviceWindow maintains its
   * own barrierExpected_ counter — when passed by value to kernels,
   * each thread block gets an independent copy, so distinct barrier_id
   * slots must be used across blocks to avoid interference.
   *
   * @param group       ThreadGroup for group coordination.
   * @param barrier_id  Barrier slot index.
   * @param timeout     Optional timeout (traps on expiry).
   */
  __device__ __forceinline__ void barrier(
      ThreadGroup& group,
      int barrier_id,
      const Timeout& timeout = Timeout()) {
    barrier_arrive(group, barrier_id);
    barrierExpected_ += static_cast<uint64_t>(handle_.nRanks - 1);
    barrier_wait(group, barrier_id, CmpOp::CMP_GE, barrierExpected_, timeout);
  }

  /**
   * barrier_peer - Pairwise barrier with a single peer.
   *
   * Group-level API: all threads in all groups must call this together.
   * Signals one peer, increments expected count by 1, then waits.
   *
   * @param target_rank  Peer rank to barrier with (must not be self).
   * @param group        ThreadGroup for group coordination.
   * @param barrier_id   Barrier slot index.
   * @param timeout      Optional timeout (traps on expiry).
   */
  __device__ __forceinline__ void barrier_peer(
      int target_rank,
      ThreadGroup& group,
      int barrier_id,
      const Timeout& timeout = Timeout()) {
    barrier_arrive_peer(group, target_rank, barrier_id);
    barrierExpected_ += 1;
    barrier_wait(group, barrier_id, CmpOp::CMP_GE, barrierExpected_, timeout);
  }

  /**
   * barrier_arrive - Signal all peers on a barrier slot without waiting.
   *
   * Group-level API: all threads in all groups must call this together.
   * Peers are horizontally partitioned across all thread groups to avoid
   * duplicate signaling. Does NOT wait — pair with barrier_wait() for a
   * full barrier, or use barrier() which combines both.
   *
   * @param group       ThreadGroup for group coordination.
   * @param barrier_id  Barrier slot index.
   */
  __device__ __forceinline__ void barrier_arrive(
      ThreadGroup& group,
      int barrier_id) {
    DEVICE_WINDOW_CHECK_BARRIER_ID(barrier_id, barrierCount_);
    group.sync();
    int nPeers = num_peers();
    int globalThreadIdx = static_cast<int>(
        group.group_id * group.group_size + group.thread_id_in_group);
    int totalThreads = static_cast<int>(group.total_groups * group.group_size);
    for (int peer_index = globalThreadIdx; peer_index < nPeers;
         peer_index += totalThreads) {
      int r = peer_index_to_rank(peer_index);
      if (handle_.get_type(r) == TransportType::P2P_NVL) {
        int nvlIdx = rankToNvlPeerIndex_[r];
        nvlBarrierPeerPtrs_[nvlIdx][barrier_id].signal(SignalOp::SIGNAL_ADD, 1);
      } else {
        handle_.get_ibgda(r).signal_remote(
            ibgdaBarrierRemoteBufs_[peer_index], barrier_id, 1);
      }
    }
    group.sync();
  }

  /**
   * barrier_arrive_peer - Signal a single peer on a barrier slot without
   *                       waiting.
   *
   * Group-level API: all threads in all groups must call this together.
   * Only the global leader sends the signal. Does NOT wait.
   *
   * @param group        ThreadGroup for group coordination.
   * @param target_rank  Peer rank to signal (must not be self).
   * @param barrier_id   Barrier slot index.
   */
  __device__ __forceinline__ void
  barrier_arrive_peer(ThreadGroup& group, int target_rank, int barrier_id) {
    DEVICE_WINDOW_CHECK_BARRIER_ID(barrier_id, barrierCount_);
    group.sync();
    if (group.is_global_leader()) {
      if (handle_.get_type(target_rank) == TransportType::P2P_NVL) {
        int nvlIdx = rankToNvlPeerIndex_[target_rank];
        nvlBarrierPeerPtrs_[nvlIdx][barrier_id].signal(SignalOp::SIGNAL_ADD, 1);
      } else {
        int ibgdaIdx = rank_to_peer_index(target_rank);
        handle_.get_ibgda(target_rank)
            .signal_remote(ibgdaBarrierRemoteBufs_[ibgdaIdx], barrier_id, 1);
      }
    }
    group.sync();
  }

  /**
   * barrier_wait - Wait for the barrier inbox to satisfy a comparison.
   *
   * Group-level API: all threads in the group must call this together.
   * Only the group leader polls the inbox; other threads block at the
   * trailing group.sync().
   *
   * @param group       ThreadGroup for group coordination.
   * @param barrier_id  Barrier slot index.
   * @param cmp         Comparison operator.
   * @param value       Threshold value for comparison.
   * @param timeout     Optional timeout (traps on expiry).
   */
  __device__ __forceinline__ void barrier_wait(
      ThreadGroup& group,
      int barrier_id,
      CmpOp cmp,
      uint64_t value,
      const Timeout& timeout = Timeout()) {
    DEVICE_WINDOW_CHECK_BARRIER_ID(barrier_id, barrierCount_);
    if (group.is_leader()) {
      while (true) {
        uint64_t total = 0;
        if (nNvlPeers_ > 0) {
          total += nvlBarrierInbox_[barrier_id].load();
        }
        if (nIbgdaPeers_ > 0) {
          // volatile: bypass L1 to read from L2 where RDMA atomics land
          volatile uint64_t* p = &ibgdaBarrierInbox_[barrier_id];
          total += *p;
        }
        if (compare(total, cmp, value)) {
          break;
        }
        TIMEOUT_TRAP_IF_EXPIRED_SINGLE(
            timeout,
            "DeviceWindow::barrier_wait(barrier_id=%d, value=%llu)"
            " rank=%d",
            barrier_id,
            static_cast<unsigned long long>(value),
            handle_.myRank);
      }
    }
    group.sync();
  }

  // ===========================================================================
  // Put (generic — dispatches to NVL or IBGDA internally)
  // ===========================================================================

  /**
   * put - One-sided write to a peer's remote buffer.
   *
   * Group-level API: all threads in the group must call this together.
   * Dispatches internally based on peer transport type:
   * - NVL: direct vectorized memcpy over NVLink (no staging buffer)
   * - IBGDA: RDMA Write via NIC (lkey/rkey resolved from registration table)
   *
   * NOTE: This does NOT use the NVL staging buffer allocated by
   * MultiPeerNvlTransportConfig.dataBufferSize. The staging buffer is
   * only used by P2pNvlTransportDevice::send()/recv().
   *
   * PRECONDITION: Both localSrc and remoteDst must be within buffers
   * previously registered via HostWindow::registerBuffer().
   *
   * @param target_rank  Rank to put to (must not be self).
   * @param group        ThreadGroup for group coordination.
   * @param remoteDst    Destination buffer on the target peer.
   * @param localSrc     Source buffer on this rank.
   * @param nbytes       Number of bytes to transfer.
   */
  __device__ __forceinline__ void put(
      int target_rank,
      ThreadGroup& group,
      void* remoteDst,
      const void* localSrc,
      std::size_t nbytes) {
    DEVICE_WINDOW_CHECK_RANK(target_rank, handle_.nRanks);
    DEVICE_WINDOW_CHECK_NOT_SELF(target_rank, handle_.myRank);
    if (handle_.get_type(target_rank) == TransportType::P2P_NVL) {
      handle_.get_nvl(target_rank)
          .put(
              group,
              static_cast<char*>(remoteDst),
              static_cast<const char*>(localSrc),
              nbytes);
    } else {
      int ibgdaIdx = rank_to_peer_index(target_rank);
      IbgdaLocalBuffer localBuf(
          const_cast<void*>(localSrc), lookupLocalLkey(localSrc));
      IbgdaRemoteBuffer remoteBuf(
          remoteDst, lookupRemoteRkey(ibgdaIdx, remoteDst));
      handle_.get_ibgda(target_rank)
          .put_group_global(group, localBuf, remoteBuf, nbytes);
    }
  }

  // ===========================================================================
  // Combined Put + Signal (generic — dispatches to NVL or IBGDA internally)
  // ===========================================================================

  /**
   * put_signal - One-sided write + signal to a peer.
   *
   * Group-level API: all threads in the group must call this together.
   * Dispatches internally:
   * - NVL: vectorized memcpy + group.sync() + atomic signal via NVLink
   * - IBGDA: RDMA Write + NIC-fenced atomic signal (HW-ordered),
   *   single signal from global leader to match NVL semantics.
   *
   * PRECONDITION: Both localSrc and remoteDst must be within buffers
   * previously registered via HostWindow::registerBuffer().
   *
   * @param target_rank  Rank to put to (must not be self).
   * @param group        ThreadGroup for group coordination.
   * @param remoteDst    Destination buffer on the target peer.
   * @param localSrc     Source buffer on this rank.
   * @param nbytes       Number of bytes to transfer.
   * @param signalId     Signal slot index in [0, peerSignalCount).
   * @param signalVal    Value to add to the signal (default: 1).
   */
  __device__ __forceinline__ void put_signal(
      int target_rank,
      ThreadGroup& group,
      void* remoteDst,
      const void* localSrc,
      std::size_t nbytes,
      int signalId,
      uint64_t signalVal = 1) {
    DEVICE_WINDOW_CHECK_RANK(target_rank, handle_.nRanks);
    DEVICE_WINDOW_CHECK_NOT_SELF(target_rank, handle_.myRank);
    if (handle_.get_type(target_rank) == TransportType::P2P_NVL) {
      handle_.get_nvl(target_rank)
          .put(
              group,
              static_cast<char*>(remoteDst),
              static_cast<const char*>(localSrc),
              nbytes);
      group.sync();
      signal_peer(
          group, target_rank, signalId, SignalOp::SIGNAL_ADD, signalVal);
    } else {
      int ibgdaIdx = rank_to_peer_index(target_rank);
      IbgdaLocalBuffer localBuf(
          const_cast<void*>(localSrc), lookupLocalLkey(localSrc));
      IbgdaRemoteBuffer remoteBuf(
          remoteDst, lookupRemoteRkey(ibgdaIdx, remoteDst));
      handle_.get_ibgda(target_rank)
          .put_group_global(group, localBuf, remoteBuf, nbytes);
      // Single fenced signal from global leader (matches NVL semantics)
      if (group.is_global_leader()) {
        // Remote buffer is pre-offset to "my row" in the peer's inbox
        // (computed once at exchange time in HostWindow), so signalId
        // is the only offset needed here.
        handle_.get_ibgda(target_rank)
            .signal_remote_with_fence(
                ibgdaPeerSignalRemoteBufs_[ibgdaIdx], signalId, signalVal);
      }
      group.sync();
    }
  }
#endif // __CUDACC__

  // ===========================================================================
  // Direct Access
  // ===========================================================================

  __device__ __forceinline__ MultiPeerDeviceHandle& get_handle() {
    return handle_;
  }

  __device__ __forceinline__ const MultiPeerDeviceHandle& get_handle() const {
    return handle_;
  }

 private:
#ifdef __CUDACC__
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
   * Lookup local lkey for a source pointer from the registration table.
   * Linear scan over registered buffers (typically 1-5 entries).
   * Traps if pointer is not in any registered buffer.
   */
  __device__ __forceinline__ NetworkLKey
  lookupLocalLkey(const void* ptr) const {
    const auto* p = static_cast<const char*>(ptr);
    for (int i = 0; i < static_cast<int>(localBufferRegistry_.size()); ++i) {
      const auto& reg = localBufferRegistry_[i];
      const auto* base = static_cast<const char*>(reg.base);
      if (p >= base && p < base + reg.size) {
        return reg.lkey;
      }
    }
    printf(
        "DeviceWindow: localSrc %p not in any registered buffer"
        " at %s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",
        ptr,
        __FILE__,
        __LINE__,
        blockIdx.x,
        blockIdx.y,
        blockIdx.z,
        threadIdx.x,
        threadIdx.y,
        threadIdx.z);
    __trap();
    return NetworkLKey{};
  }

  /**
   * Lookup remote rkey for a destination pointer on a specific IBGDA peer.
   * Linear scan over registered buffers.
   * Traps if pointer is not in any registered buffer for that peer.
   */
  __device__ __forceinline__ NetworkRKey
  lookupRemoteRkey(int ibgdaPeerIdx, const void* remotePtr) const {
    const auto* p = static_cast<const char*>(remotePtr);
    int nRegs = static_cast<int>(localBufferRegistry_.size());
    for (int i = 0; i < nRegs; ++i) {
      const auto& reg = remoteBufferRegistry_[i * nIbgdaPeers_ + ibgdaPeerIdx];
      const auto* base = static_cast<const char*>(reg.base);
      if (p >= base && p < base + reg.size) {
        return reg.rkey;
      }
    }
    printf(
        "DeviceWindow: remoteDst %p not registered for IBGDA peer %d"
        " at %s:%d block=(%u,%u,%u) thread=(%u,%u,%u)\n",
        remotePtr,
        ibgdaPeerIdx,
        __FILE__,
        __LINE__,
        blockIdx.x,
        blockIdx.y,
        blockIdx.z,
        threadIdx.x,
        threadIdx.y,
        threadIdx.z);
    __trap();
    return NetworkRKey{};
  }
#endif // __CUDACC__

  // Transport handle (provides get_type, get_nvl, get_ibgda, myRank, nRanks)
  MultiPeerDeviceHandle handle_;

  // Pre-computed peer index map: O(1) rank → NVL peer index lookup
  // rankToNvlPeerIndex_[rank] = NVL peer index, or -1 if not NVL peer
  // IBGDA peer index is not stored — it equals rank_to_peer_index(rank)
  // since all non-self ranks are IBGDA peers.
  DeviceSpan<int> rankToNvlPeerIndex_;

  // Peer counts
  int nNvlPeers_{0};
  int nIbgdaPeers_{0};

  // --- Per-peer signal buffers ---
  int peerSignalCount_{0};
  DeviceSpan<SignalState> nvlPeerSignalInbox_;
  DeviceSpan<DeviceSpan<SignalState>> nvlPeerSignalSpans_;
  uint64_t* ibgdaPeerSignalInbox_{nullptr};
  DeviceSpan<IbgdaRemoteBuffer> ibgdaPeerSignalRemoteBufs_;

  // --- Per-peer counter buffers (IBGDA-only, local) ---
  int peerCounterCount_{0};
  uint64_t* ibgdaPeerCounterBuf_{nullptr};

  // --- Barrier buffers (flat, per-peer-type) ---
  int barrierCount_{0};
  SignalState* nvlBarrierInbox_{nullptr};
  DeviceSpan<SignalState*> nvlBarrierPeerPtrs_;
  uint64_t* ibgdaBarrierInbox_{nullptr};
  DeviceSpan<IbgdaRemoteBuffer> ibgdaBarrierRemoteBufs_;
  uint64_t barrierExpected_{0};

  // --- Buffer registration table (for generic put/put_signal) ---
  // Local: lkey lookup for source buffers
  DeviceSpan<LocalBufferRegistration> localBufferRegistry_;
  // Remote: rkey lookup for destination buffers
  // Flattened: [regIdx * nIbgdaPeers + ibgdaPeerIdx]
  DeviceSpan<RemoteBufferRegistration> remoteBufferRegistry_;

  // HostWindow constructs DeviceWindow directly
  friend class HostWindow;

  // Test helper for unit tests (constructs minimal DeviceWindow without
  // HostWindow)
  friend struct comms::pipes::test::NvlOnlyDeviceWindowBuffers;
  friend struct comms::pipes::test::IbgdaOnlyDeviceWindowBuffers;
};

} // namespace comms::pipes
