// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include <cuda_runtime.h>

#include <folly/futures/Future.h>
#include <folly/io/async/EventBase.h>

#include <comms/ctran/ibverbx/IbvCommon.h>
#include <comms/ctran/ibverbx/IbvDevice.h>
#include <comms/ctran/ibverbx/IbvMr.h>
#include <comms/ctran/ibverbx/IbvPd.h>
#include <comms/ctran/ibverbx/IbvVirtualCq.h>
#include <comms/ctran/ibverbx/IbvVirtualQp.h>
#include <comms/ctran/ibverbx/IbvVirtualWr.h>
#include <comms/utils/commSpecs.h>

namespace torch::comms {

// Configuration for staged RDMA transfers.
struct StagedTransferConfig {
  // Size of the GPU staging buffer on each side. Each chunk transfer moves
  // at most this many bytes via a single RDMA_WRITE_WITH_IMM.
  size_t stagingBufSize = 64 * 1024 * 1024; // 64 MB

  // Timeout for waiting on the recvReady flag or CQ poll between chunks.
  std::chrono::milliseconds chunkTimeout{30000};
};

// Information exchanged between server and client during setupLocalTransport()
// so each side knows the remote staging buffer address and rkey for one-sided
// RDMA operations. Serialized into the connection info string by
// setupLocalTransport() and deserialized/stored by connectRemoteTransport().
struct StagingRendezvousInfo {
  struct BufferInfo {
    uintptr_t addr{0};
    uint32_t rkey{0};
    size_t size{0};
  };

  BufferInfo stagingBuf{};

  // Only populated on the server side — the client uses this to RDMA_WRITE
  // the recvReady acknowledgement back to the server.
  std::optional<BufferInfo> recvReady;
};

// RAII wrapper for a GPU staging buffer registered for GPUDirect RDMA via
// dmabuf. Allocates GPU memory with cudaMalloc, exports a dmabuf fd, and
// registers it with the IB protection domain for zero-copy RDMA access.
//
// Destruction order: deregister MR → close dmabuf fd → cudaFree.
class StagedBuffer {
 public:
  StagedBuffer(size_t size, int cudaDev, ibverbx::IbvPd& pd);
  ~StagedBuffer();

  // Move-only
  StagedBuffer(StagedBuffer&& other) noexcept;
  StagedBuffer& operator=(StagedBuffer&& other) noexcept;
  StagedBuffer(const StagedBuffer&) = delete;
  StagedBuffer& operator=(const StagedBuffer&) = delete;

  void* data() const {
    return buf_;
  }
  size_t size() const {
    return size_;
  }
  int cudaDev() const {
    return cudaDev_;
  }
  uint32_t lkey() const {
    return mr_->mr()->lkey;
  }
  uint32_t rkey() const {
    return mr_->mr()->rkey;
  }

 private:
  void* buf_{nullptr};
  size_t size_{0};
  int cudaDev_{-1};
  int dmabufFd_{-1};
  std::optional<ibverbx::IbvMr> mr_;
};

// Describes GPU memory regions for staged RDMA transfers. A single entry
// represents a contiguous buffer; multiple entries describe non-contiguous
// regions for scatter/gather transfers.
struct ScatterGatherDescriptor {
  struct Entry {
    void* ptr;
    size_t size;
  };
  std::vector<Entry> entries;

  size_t totalBytes() const {
    size_t total = 0;
    for (const auto& e : entries) {
      total += e.size;
    }
    return total;
  }
};

// Base class for staged RDMA transports. Holds shared IB resources and
// provides protected helpers for IB setup and QP connection.
//
// Not intended to be used directly — use StagedRdmaServerTransport or
// StagedRdmaClientTransport.
//
// Bootstrap workflow:
//   1. Construct server/client transport(cudaDev, evb)
//   2. connInfo = setupLocalTransport() — creates IB resources + staging buffer
//   3. Exchange connInfo out-of-band (e.g. via Thrift RPC)
//   4. Server: connectRemoteTransport(clientConnInfo)
//   5. Client: connectRemoteTransport(serverConnInfo)
//   6. Server: send(src)    — chunked D2D + RDMA pipeline
//      Client: recv(dst)    — chunked RDMA + D2D pipeline
class StagedRdmaTransportBase {
 public:
  explicit StagedRdmaTransportBase(
      int cudaDev,
      folly::EventBase* evb = nullptr,
      StagedTransferConfig config = {});
  ~StagedRdmaTransportBase();

  // Non-copyable, non-movable (owns IB resources tied to device state)
  StagedRdmaTransportBase(const StagedRdmaTransportBase&) = delete;
  StagedRdmaTransportBase& operator=(const StagedRdmaTransportBase&) = delete;
  StagedRdmaTransportBase(StagedRdmaTransportBase&&) = delete;
  StagedRdmaTransportBase& operator=(StagedRdmaTransportBase&&) = delete;

  // Return the staging buffer size from config.
  size_t stagingBufSize() const {
    return config_.stagingBufSize;
  }

 protected:
  int cudaDev_;
  StagedTransferConfig config_;

  // EventBase for running transfers. Optional for construction/setup/connect,
  // but required for send/recv (CHECK_THROW if nullptr).
  folly::EventBase* evb_{nullptr};

  // Peer's staging info — populated by connectQp() from the peer's serialized
  // connection info. Used by send (to know remote staging buffer addr)
  // and recv (to know remote recvReady addr).
  StagingRendezvousInfo peerStaging_;

  // IB resources (H100: single device/PD; GB200 diff expands to vectors)
  std::optional<ibverbx::IbvDevice> device_;
  std::optional<ibverbx::IbvPd> pd_;
  std::optional<ibverbx::IbvVirtualCq> vcq_;
  std::optional<StagedBuffer> stagingBuf_;
  cudaStream_t stream_{nullptr};

  // Virtual QP — declared last so it is destroyed first.
  std::optional<ibverbx::IbvVirtualQp> vqp_;

  // Get the device ID for building deviceKeys maps.
  int32_t getDeviceId() const;

  // Protected helpers — called explicitly by subclasses, no virtual dispatch.

  // Initialize IB resources: device, PD, CQ, VirtualQP, staging buffer,
  // CUDA stream. Must be called before connectQp().
  void initIbResources();

  // Connect to the peer using their serialized connection info. Deserializes
  // peer info, stores peerStaging_, transitions QP through INIT → RTR → RTS.
  void connectQp(const std::string& peerConnInfo);

  // Serialize local connection info (business card + GID/port/MTU + staging)
  // into a JSON string for exchange with the peer.
  std::string serializeConnInfo(const StagingRendezvousInfo& localStaging);
};

// Server-side staged RDMA transport. Sends data to the client via chunked
// D2D copy → RDMA_WRITE_WITH_IMM pipeline.
class StagedRdmaServerTransport : public StagedRdmaTransportBase {
 public:
  using StagedRdmaTransportBase::StagedRdmaTransportBase;
  ~StagedRdmaServerTransport();

  // Initialize IB resources, allocate recvReady flag, and return serialized
  // connection info for the peer.
  std::string setupLocalTransport();

  // Connect to the client using their serialized connection info.
  void connectRemoteTransport(const std::string& peerConnInfo);

  // Transfer GPU memory regions described by src to the client's staging
  // buffer, pipelining in stagingBufSize chunks. All entry pointers must be
  // on cudaDev_. Requires evb_ (CHECK_THROW if nullptr).
  folly::SemiFuture<commResult_t> send(const ScatterGatherDescriptor& src);

 private:
  // recvReady flag — CPU-pinned, cache-line aligned, RDMA-registered.
  // The client writes kRecvReadyValue here via RDMA_WRITE to signal that it
  // has finished copying data out of its staging buffer.
  // Pre-initialized to kRecvReadyValue so the first send() proceeds
  // immediately.
  struct AlignedDelete {
    void operator()(std::atomic<uint64_t>* p) const {
      ::operator delete(p, std::align_val_t{64});
    }
  };
  std::unique_ptr<std::atomic<uint64_t>, AlignedDelete> recvReadyFlag_;
  std::optional<ibverbx::IbvMr> recvReadyServerMr_;
};

// Client-side staged RDMA transport. Receives data from the server via
// RDMA_WRITE_WITH_IMM → D2D copy pipeline.
class StagedRdmaClientTransport : public StagedRdmaTransportBase {
 public:
  using StagedRdmaTransportBase::StagedRdmaTransportBase;
  ~StagedRdmaClientTransport();

  // Initialize IB resources and return serialized connection info for the
  // peer.
  std::string setupLocalTransport();

  // Connect to the server using their serialized connection info. Registers
  // recvReady source MR and posts initial recv WR.
  void connectRemoteTransport(const std::string& peerConnInfo);

  // Receive into GPU memory regions described by dst from the server's
  // staging buffer, pipelining in stagingBufSize chunks. All entry pointers
  // must be on cudaDev_ (CPU buffers are not supported). numChunks is
  // computed internally from totalBytes and the server's staging buffer
  // size (exchanged during connectRemoteTransport()).
  // Requires evb_ (CHECK_THROW if nullptr).
  folly::SemiFuture<commResult_t> recv(const ScatterGatherDescriptor& dst);

 private:
  // MR for &kRecvReadyValue — used as source for RDMA_WRITE to server's
  // recvReadyFlag_.
  std::optional<ibverbx::IbvMr> recvReadyClientMr_;
};

} // namespace torch::comms
