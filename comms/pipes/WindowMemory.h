// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>

#include "comms/pipes/GpuMemHandler.h"
#include "comms/utils/CudaRAII.h"

namespace comms::pipes {

// Forward declaration - users must include DeviceSignal.cuh to use the returned
// object
class DeviceSignal;

/**
 * Configuration for window memory allocation.
 */
struct WindowMemoryConfig {
  // Number of signal slots (inbox size)
  // Typical: 1 for simple signaling, more for multi-phase patterns
  std::size_t signalCount{1};
};

/**
 * WindowMemory - Host-side RAII manager for signal slot buffers
 *
 * Manages GPU memory allocation and handle exchange for DeviceSignal.
 * This class can be used standalone or internally by MultiPeerNvlTransport.
 *
 * MEMORY MODEL (Inbox):
 * Each rank has a local "inbox" buffer. All peers can write to this rank's
 * inbox to signal it. This provides a many-to-one signaling pattern.
 *
 * MEMORY ALIGNMENT:
 * Signal slots are 128-byte aligned to avoid false sharing between slots.
 *
 * COMMUNICATOR SEMANTICS:
 * - Constructor allocates local GPU memory
 * - exchange() is COLLECTIVE (all ranks must call)
 * - getDeviceSignal() returns device object after exchange()
 *
 * USAGE (Standalone):
 *   WindowMemory windowMemory(myRank, nRanks, bootstrap, config);
 *   windowMemory.exchange();  // Collective - all ranks must call
 *   auto deviceSignal = windowMemory.getDeviceSignal();
 *   myKernel<<<...>>>(deviceSignal, ...);
 *
 * USAGE (Via MultiPeerNvlTransport):
 *   MultiPeerNvlTransport transport(...);  // Uses WindowMemory internally
 *   transport.exchange();
 *   auto device = transport.getMultiPeerDeviceTransport();
 */
class WindowMemory {
 public:
  WindowMemory(const WindowMemory&) = delete;
  WindowMemory& operator=(const WindowMemory&) = delete;
  WindowMemory(WindowMemory&&) = delete;
  WindowMemory& operator=(WindowMemory&&) = delete;

  /**
   * Constructor - Allocate signal slot buffers
   *
   * @param myRank This rank's ID (0 to nRanks-1)
   * @param nRanks Total number of ranks
   * @param bootstrap Bootstrap interface for collective handle exchange
   * @param config Window memory configuration
   * @param memSharingMode Optional: Memory sharing mode (auto-detected if not
   * specified)
   */
  WindowMemory(
      int myRank,
      int nRanks,
      std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
      const WindowMemoryConfig& config,
      MemSharingMode memSharingMode = GpuMemHandler::detectBestMode());

  ~WindowMemory() = default;

  /**
   * exchange - Exchange memory handles across all ranks
   *
   * COLLECTIVE OPERATION: All ranks must call before getDeviceSignal().
   */
  void exchange();

  /**
   * isExchanged - Check if exchange() has been called
   */
  bool isExchanged() const {
    return exchanged_;
  }

  /**
   * getDeviceSignal - Get device-side signal object
   *
   * PRECONDITION: exchange() must have completed on all ranks.
   *
   * Returns by value since DeviceSignal is a lightweight handle (~40 bytes)
   * containing only pointers and integers - no heap allocations or RAII.
   *
   * @return DeviceSignal for use in CUDA kernels
   * @throws std::runtime_error if called before exchange()
   */
  DeviceSignal getDeviceSignal() const;

  /**
   * Get the memory sharing mode being used.
   */
  MemSharingMode getMemSharingMode() const {
    return inboxHandler_->getMode();
  }

  /**
   * Get signal count.
   */
  std::size_t signalCount() const {
    return config_.signalCount;
  }

  /**
   * Get rank.
   */
  int rank() const {
    return myRank_;
  }

  /**
   * Get total number of ranks.
   */
  int nRanks() const {
    return nRanks_;
  }

 private:
  /**
   * peerToRank - Convert peer index to global rank
   *
   * @param peer Peer index (0 to nRanks-2)
   * @return Global rank of the peer
   */
  int peerToRank(int peer) const {
    return peer < myRank_ ? peer : peer + 1;
  }

  const int myRank_;
  const int nRanks_;
  std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap_;
  const WindowMemoryConfig config_;
  const MemSharingMode memSharingMode_;

  // Signal inbox buffer (local memory, peers write here)
  // Uses GpuMemHandler because this IS shared with peers via exchange
  std::unique_ptr<GpuMemHandler> inboxHandler_;

  // Peer inbox pointers array (device-accessible, LOCAL-only)
  // Uses DeviceBuffer because this is NOT shared with peers
  // Size = nPeers (not nRanks) - excludes self
  std::unique_ptr<meta::comms::DeviceBuffer> peerInboxPtrsDevice_;

  // Inbox size in bytes
  std::size_t inboxSize_{0};

  // Flag to track if exchange has been called
  bool exchanged_{false};
};

} // namespace comms::pipes
