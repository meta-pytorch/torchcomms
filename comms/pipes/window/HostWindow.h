// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <vector>

#include "comms/pipes/GpuMemHandler.h"
#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/Transport.cuh"
#include "comms/utils/CudaRAII.h"

namespace comms::pipes {

// Forward declarations
class DeviceWindow;
class MultiPeerTransport;
class P2pIbgdaTransportDevice;
struct LocalBufferRegistration;
struct RemoteBufferRegistration;
enum class TransportType : uint8_t;

/**
 * Configuration for unified window memory allocation.
 *
 * Supports per-peer signals, IBGDA-only counters, and dedicated barriers.
 * All types are optional — set count to 0 to skip allocation.
 */
struct WindowConfig {
  // Per-peer signals: one uint64_t per (peer, signal_id).
  // Use for point-to-point "has rank X signaled?" wait_signal_from().
  std::size_t peerSignalCount{0};

  // IBGDA-only per-peer counters: per-peer local NIC completion.
  std::size_t peerCounterCount{0};

  // Number of barrier slots (dedicated buffers, flat accumulation model).
  std::size_t barrierCount{0};
};

/**
 * HostWindow - Host-side RAII manager for unified signal/counter buffers
 *
 * Manages dual NVL + IBGDA signal buffers and IBGDA-only counter buffers.
 * Each transport domain has physically separate backing buffers to avoid
 * cross-transport atomicity hazards (GPU atomics vs NIC RDMA atomics).
 *
 * Optionally registers/exchanges a user-provided data buffer on both
 * NVL (IPC) and IBGDA (RDMA) sides.
 *
 * SIGNAL TYPES:
 * - Per-peer: one row per peer, supports wait_signal_from(). Wait sums
 *   across all peer rows.
 *
 * BARRIER:
 * - Dedicated buffers using flat accumulation model. All peers atomicAdd
 *   to one slot, O(1) wait.
 *
 * COUNTER SEMANTICS (IBGDA-only):
 * - Counters are local NIC completion notifications (not remote peer acks)
 * - Companion QP writes to sender's own local counter buffer
 * - NVL doesn't need counters (stores are synchronous)
 *
 * COMMUNICATOR SEMANTICS:
 * - Constructor allocates local GPU memory
 * - exchange() is COLLECTIVE (all ranks must call)
 * - getDeviceWindow() returns device object after exchange()
 */
class HostWindow {
 public:
  HostWindow(const HostWindow&) = delete;
  HostWindow& operator=(const HostWindow&) = delete;
  HostWindow(HostWindow&&) = delete;
  HostWindow& operator=(HostWindow&&) = delete;

  /**
   * Construct a HostWindow from MultiPeerTransport.
   *
   * Extracts topology, rank info, and transport handles from the
   * MultiPeerTransport instance. The transport must outlive this object.
   *
   * If a user buffer is provided, HostWindow auto-registers and exchanges it
   * via registerAndExchangeBuffer() during exchange(). The buffer is then
   * accessible through the DeviceWindow's buffer registration table.
   *
   * @param transport MultiPeerTransport providing topology and buffer APIs
   * @param config Window memory configuration
   * @param userBuffer Optional user-allocated GPU data buffer
   * @param userBufferSize Size of user buffer in bytes (0 if no buffer)
   */
  HostWindow(
      MultiPeerTransport& transport,
      const WindowConfig& config,
      void* userBuffer = nullptr,
      std::size_t userBufferSize = 0);

  ~HostWindow();

  /**
   * exchange - Exchange memory handles across all ranks
   *
   * COLLECTIVE OPERATION: All NVL ranks must call for NVL exchange,
   * all IBGDA ranks must call for IBGDA exchange.
   */
  void exchange();

  bool isExchanged() const {
    return exchanged_;
  }

  /**
   * getDeviceWindow - Get the flattened device-side window handle
   *
   * Returns a DeviceWindow with all signal, barrier, counter, and
   * transport state held directly (no sub-objects). The DeviceWindow
   * uses MultiPeerDeviceHandle for transport dispatch and pre-computed
   * peer index maps for O(1) rank-to-peer-index lookup.
   *
   * Must be called after exchange() and after all registerLocalBuffer()
   * and registerAndExchangeBuffer() calls.
   */
  DeviceWindow getDeviceWindow() const;

  // --- Buffer registration for generic put/put_signal ---

  /**
   * Register a local buffer for use as a source in
   * DeviceWindow::put/put_signal.
   *
   * NOT collective: only registers locally for IBGDA (gets lkey).
   * Does not exchange with peers. Use for source-only buffers.
   *
   * May be called before or after exchange(). Call getDeviceWindow() after
   * exchange() and all registration calls to get a DeviceWindow with the
   * registration table.
   *
   * @param ptr   Local GPU buffer pointer
   * @param size  Buffer size in bytes
   * @return      Local buffer registration index
   */
  int registerLocalBuffer(void* ptr, std::size_t size);

  /**
   * Register and exchange a buffer for use as a destination in
   * DeviceWindow::put/put_signal.
   *
   * COLLECTIVE: all ranks must call together, each with their local buffer.
   * Registers locally for IBGDA (gets lkey) and exchanges with all IBGDA
   * peers (gets rkeys). For NVL peers, exchanges IPC/fabric handles.
   *
   * Each DeviceWindow supports exactly one exchanged dst buffer. Calling
   * this more than once is an error.
   *
   * May be called before or after exchange(). Call getDeviceWindow() after
   * exchange() and all registration calls to get a DeviceWindow with the
   * registration table.
   *
   * @param ptr   Local GPU buffer pointer
   * @param size  Buffer size in bytes
   * @return      Local buffer registration index
   */
  int registerAndExchangeBuffer(void* ptr, std::size_t size);

  int rank() const {
    return myRank_;
  }
  int nRanks() const {
    return nRanks_;
  }

  const WindowConfig& config() const {
    return config_;
  }

  int numNvlPeers() const {
    return static_cast<int>(nvlPeerRanks_.size());
  }
  int numIbgdaPeers() const {
    return static_cast<int>(ibgdaPeerRanks_.size());
  }

  /**
   * get_nvlink_address - Get the NVLink-mapped pointer to a peer's window buf.
   *
   * Returns the IPC-mapped device pointer for the given peer's registered
   * window buffer. Returns nullptr if the peer is not NVLink-accessible,
   * is self, or no buffer has been registered/exchanged.
   *
   * @param peer   Global rank of the peer.
   * @param offset Byte offset into the peer's window buffer (default 0).
   * @return Host-visible device pointer, or nullptr.
   */
  void* get_nvlink_address(int peer, std::size_t offset = 0) const;

 private:
  void uploadRegistrationsToDevice();

  // Transport reference (non-owning, must outlive this object)
  MultiPeerTransport& transport_;

  // Rank info (cached from transport)
  const int myRank_{-1};
  const int nRanks_{-1};
  const WindowConfig config_;

  // Topology (derived from transport)
  std::vector<int> nvlPeerRanks_;
  std::vector<int> ibgdaPeerRanks_;
  int nvlLocalRank_{-1};
  int nvlNRanks_{0};

  // --- Pre-computed peer index map (device-accessible, O(1) lookup) ---
  // rankToNvlPeerIndex_[rank] = NVL peer index, or -1 if not NVL peer
  // IBGDA peer index is not stored — it equals rank_to_peer_index(rank)
  // since all non-self ranks are IBGDA peers.
  std::unique_ptr<meta::comms::DeviceBuffer> peerIndexMapsDevice_;

  // --- Barrier buffers (NVL side) ---
  std::unique_ptr<GpuMemHandler> nvlBarrierHandler_;
  std::unique_ptr<meta::comms::DeviceBuffer> nvlBarrierPeerPtrsDevice_;

  // --- Barrier buffers (IBGDA side) ---
  // IbgdaLocalBuffer.ptr holds the cudaMalloc'd GPU memory.
  // After registerIbgdaBuffer(), .lkey is populated.
  IbgdaLocalBuffer ibgdaBarrierLocalBuf_;
  std::unique_ptr<meta::comms::DeviceBuffer> ibgdaBarrierRemoteBufsDevice_;

  // --- Per-peer signals (NVL side) ---
  std::unique_ptr<GpuMemHandler> nvlPeerSignalHandler_;
  std::unique_ptr<meta::comms::DeviceBuffer> nvlPeerSignalSpansDevice_;

  // --- Per-peer signals (IBGDA side) ---
  IbgdaLocalBuffer ibgdaPeerSignalLocalBuf_;
  std::unique_ptr<meta::comms::DeviceBuffer> ibgdaPeerSignalRemoteBufsDevice_;

  // --- Per-peer counters (IBGDA-only, local — no exchange) ---
  IbgdaLocalBuffer ibgdaPeerCounterLocalBuf_;

  // --- User data buffer (optional, auto-registered via
  //     registerAndExchangeBuffer) ---
  void* userBuffer_{nullptr};
  std::size_t userBufferSize_{0};

  // --- Buffer registration table (for generic put/put_signal) ---
  std::vector<LocalBufferRegistration> localRegistrations_;
  // Remote registrations for the single exchanged dst buffer (one per IBGDA
  // peer)
  std::vector<RemoteBufferRegistration> remoteRegistrations_;
  // NVL mapped pointers for the exchanged dst buffer (per-rank, includes self
  // as nullptr)
  std::vector<void*> exchangedNvlMappedPtrs_;
  std::unique_ptr<meta::comms::DeviceBuffer> remoteRegistrationsDevice_;

  // --- Window buffer NVL peer pointers (for offset-based put/put_signal) ---
  // Device copy of NVL peers' IPC-mapped window buffer pointers.
  std::unique_ptr<meta::comms::DeviceBuffer> userNvlPeerPtrsDevice_;

  bool userBufferRegistered_{false};
  bool exchanged_{false};
};

} // namespace comms::pipes
