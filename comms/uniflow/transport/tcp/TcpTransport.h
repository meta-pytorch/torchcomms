// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "comms/uniflow/controller/TcpController.h"
#include "comms/uniflow/executor/EventBase.h"
#include "comms/uniflow/transport/Transport.h"
#include "comms/uniflow/transport/tcp/TcpPinnedSlabPool.h"

namespace uniflow {

class TcpRemoteRegistrationHandle;

/// Transport-local behavior plus the controller's socket configuration.
/// The converting constructor preserves existing callsites that pass a bare
/// TcpSocketConfig as the transport/factory constructor argument.
/// Defaults for frontend NIC discovery. Kept with the transport that consumes
/// them so MultiTransport and anything building a TcpTransportFactory directly
/// stripe across the same NICs.
inline constexpr std::string_view kDefaultFrontendDevicePrefix = "eth";
inline constexpr size_t kDefaultMaxFrontendDevices = 2;

/// Frontend data NICs to stripe lanes across, lowest name first and one port
/// per PCI card before taking a second port from any card. Only devices that
/// are up and carry a usable global address are returned.
///
/// Leaving TcpTransportConfig::bindToDevices empty means no binding at all, so
/// egress falls to the routing table and lands on one NIC. MultiTransport calls
/// this to fill that in; it is declared here so a caller bypassing
/// MultiTransport shares the selection instead of re-deriving and drifting.
std::vector<std::string> enumerateFrontendDevices(
    const std::string& prefix,
    size_t maxDevices);

struct TcpTransportConfig {
  controller::TcpSocketConfig socketConfig{};
  bool asyncGetH2d{true};
  // Number of parallel data sockets ("lanes") per bound device, not per peer.
  // The total for a connection is this times the device count, so the default
  // of 4 gives 4 lanes on one NIC and 8 across two -- each device gets a full
  // complement rather than the connection's lanes being divided among them.
  //
  // One TCP connection is bounded by what a single sender thread can push, so
  // more lanes are the only way past that. Throughput scales with lane count
  // until a NIC saturates; past that extra lanes only add threads, which is why
  // 4 is the default. That ceiling is per NIC, which is what makes per-device
  // the right unit.
  //
  // Both peers must agree, and so must their device counts: above 1 total lane
  // the lanes exchange a TcpLaneHello, so a peer defaulting to 4 cannot talk to
  // one that predates lanes or is pinned to 1. Set this to 1 with no bound
  // devices on both sides for mixed-version interoperability.
  size_t numSocketsPerDevice{4};

  /// Network devices to stripe lanes across, e.g. {"eth1", "eth2"}. Empty (the
  /// default) leaves egress to the routing table, exactly as before.
  ///
  /// Lane i is placed on device i % bindToDevices.size(), so the listener needs
  /// one bound socket per device: for a `get` the bulk data flows listener to
  /// dialer and the listener's egress follows its own routing table, so binding
  /// only the dialer would leave `get` traffic on a single NIC. Both peers must
  /// name the same number of devices, because each side derives the
  /// lane-to-device mapping from the lane index alone.
  ///
  /// numSocketsPerDevice applies to each device, so striping does not dilute
  /// the per-NIC lane count: the default 4 gives 8 lanes across 2 devices.
  /// Dividing a fixed lane total across devices instead leaves every NIC
  /// half-fed -- the configuration this per-device unit exists to make
  /// unreachable.
  ///
  /// Only worth enabling for large transfers. Small transfers are latency-bound
  /// rather than bandwidth-bound, where striping is neutral to slightly
  /// negative; the gain grows with transfer size.
  ///
  /// Pairing two ports on one PCI card measured no worse than two ports on
  /// separate cards, even though raw iperf3 on this hardware shows a clear card
  /// limit: this path does not yet reach what a single card sustains, so the
  /// card is not the binding constraint. The receive-side H2D copy is, and card
  /// affinity starts to matter once that is fixed.
  ///
  /// Device names are per-host and need not match between peers: the same
  /// physical port is eth3 on one MI350 host and eth0 on the next.
  std::vector<std::string> bindToDevices;

  TcpTransportConfig() = default;
  /* implicit */ TcpTransportConfig(controller::TcpSocketConfig config)
      : socketConfig(std::move(config)) {}
};

/// Completion state shared by all requests in a single put()/get()/send()/
/// recv() call. The call's promise is fulfilled once every request's reply has
/// arrived, or on the first error. Thread-safe: touched by the caller thread,
/// the reader thread and the sender thread.
struct TcpOpState {
  std::mutex mu;
  size_t remaining{0};
  bool done{false};
  std::promise<Status> promise;

  void completeOne() {
    std::lock_guard<std::mutex> lk(mu);
    completeOneLocked();
  }

  void fail(Status status) {
    std::lock_guard<std::mutex> lk(mu);
    failLocked(std::move(status));
  }

  /// Reserves the right to write into a caller-owned destination. Promise
  /// resolution is deferred until every reservation retires, so the caller
  /// cannot free the destination while a write is still in flight.
  bool tryBeginWrite() {
    std::lock_guard<std::mutex> lk(mu);
    if (done) {
      return false;
    }
    ++writesInFlight_;
    return true;
  }

  /// Retires one write reservation and records that chunk's result. The first
  /// failure wins, but is not published until every in-flight write is done.
  void endWrite(Status status) {
    std::lock_guard<std::mutex> lk(mu);
    if (writesInFlight_ == 0) {
      return;
    }
    --writesInFlight_;
    if (status.hasError()) {
      latchFailureLocked(std::move(status));
    } else if (remaining > 0) {
      --remaining;
    }
    settleLocked();
  }

  /// Copies a reply payload into the caller's destination buffer (`write`) and
  /// records that chunk's completion as one atomic lifetime reservation.
  ///
  /// Resolving this op's promise is what releases the caller to free or reuse
  /// the destination. The reservation keeps the promise unresolved while
  /// `write` runs without holding `mu`; a concurrent failure is latched and
  /// published only after the write retires. This is the same lifetime rule
  /// used by asynchronous writes whose completion happens on another thread.
  ///
  /// `mu` is a leaf lock: nothing in this struct acquires another mutex while
  /// holding it, and no caller may hold one of the transport's container
  /// mutexes (`inflightMu_`, `recvMu_`, a lane's `mu`) when calling in.
  template <typename Fn>
  void writeAndComplete(Fn&& write) {
    if (!tryBeginWrite()) {
      return;
    }
    try {
      endWrite(write());
    } catch (...) {
      endWrite(
          Err(ErrCode::TransportError,
              "tcp: destination write raised an exception"));
      throw;
    }
  }

 private:
  void completeOneLocked() {
    if (done) {
      return;
    }
    if (remaining > 0) {
      --remaining;
    }
    settleLocked();
  }

  void failLocked(Status status) {
    if (done) {
      return;
    }
    if (writesInFlight_ > 0) {
      latchFailureLocked(std::move(status));
      return;
    }
    done = true;
    promise.set_value(std::move(status));
  }

  void latchFailureLocked(Status status) {
    if (!firstFailure_.has_value()) {
      firstFailure_.emplace(std::move(status));
    }
  }

  void settleLocked() {
    if (done || writesInFlight_ > 0) {
      return;
    }
    if (firstFailure_.has_value()) {
      done = true;
      promise.set_value(std::move(*firstFailure_));
    } else if (remaining == 0) {
      done = true;
      promise.set_value(Ok());
    }
  }

  size_t writesInFlight_{0};
  std::optional<Status> firstFailure_;
};

struct TcpTransportInfo {
  struct __attribute__((packed)) Header {
    uint16_t port{0};
    uint16_t hostLen{0};
  };

  std::string host{"127.0.0.1"};
  uint16_t port{0};

  /// One listener socket's address, as advertised to the peer.
  struct Endpoint {
    std::string host;
    uint16_t port{0};
  };

  /// Additional listeners, one per extra device when striping. Endpoint 0 is
  /// (host, port) above, so this holds devices 1..D-1 and stays empty unless
  /// TcpTransportConfig::bindToDevices names more than one device.
  ///
  /// Appended after the host bytes instead of being counted in Header, so a
  /// single-device transport serializes byte-identically to a build that
  /// predates striping. A multi-device one does not: the extra bytes fail the
  /// old exact-size check, so striping requires this build on both peers.
  std::vector<Endpoint> extraEndpoints;

  /// Endpoint `index`, counting (host, port) as 0. Maps a lane to the listener
  /// it belongs on. Indices past the end clamp to endpoint 0.
  Endpoint endpointAt(size_t index) const;
  size_t endpointCount() const {
    return 1 + extraEndpoints.size();
  }

  TransportInfo serialize() const;
  static Result<TcpTransportInfo> deserialize(std::span<const uint8_t> data);
};

class CudaApi;
// Wire header, defined in TcpWireProtocol.h. Only referenced by private member
// declarations here, so this header stays free of the wire format.
struct TcpMsgHeader;

/// Factory-shared registry mapping a locally-assigned segment id to the
/// registered host buffer. registerSegment() (on the factory) populates it;
/// every TcpTransport created by that factory shares it so an inbound WRITE or
/// READ_REQUEST naming a segId can be resolved to the local buffer.
/// Thread-safe.
///
/// The registry hands out leases rather than bare entries because the mutex
/// protects the map, not the lifetime of the memory the map points at. `ptr` is
/// the application's buffer, borrowed at registerSegment() time; the registry
/// never owns it and so cannot extend its life. Without a lease the reader
/// thread could copy an entry out, the owner could deregister and free the
/// buffer, and the reader would then write into freed memory. erase() therefore
/// waits for outstanding leases to drain, which is the barrier ibv_dereg_mr
/// gives the RDMA path for free.
class TcpSegmentRegistry {
 public:
  struct Entry {
    void* ptr{nullptr};
    size_t len{0};
    MemoryType memType{MemoryType::DRAM};
    int deviceId{-1};
  };

  /// A borrow of a registered segment that blocks its deregistration. While a
  /// lease is alive, erase() for that segId waits, so `ptr` stays valid for as
  /// long as the holder keeps the lease in scope. Hold one across every access
  /// to `ptr` -- a memcpy or a device copy -- and let it go as soon as the copy
  /// is done, because a lease held longer stalls any thread deregistering that
  /// segment. Empty (`!lease`) means the segment is unregistered or already
  /// being torn down.
  class Lease {
   public:
    Lease() = default;
    Lease(TcpSegmentRegistry* registry, uint64_t segId, Entry entry)
        : registry_(registry), segId_(segId), entry_(entry) {}

    ~Lease() {
      reset();
    }

    Lease(Lease&& other) noexcept
        : registry_(other.registry_),
          segId_(other.segId_),
          entry_(other.entry_) {
      other.registry_ = nullptr;
    }

    Lease& operator=(Lease&& other) noexcept {
      if (this != &other) {
        reset();
        registry_ = other.registry_;
        segId_ = other.segId_;
        entry_ = other.entry_;
        other.registry_ = nullptr;
      }
      return *this;
    }

    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

    explicit operator bool() const {
      return registry_ != nullptr;
    }
    const Entry* operator->() const {
      return &entry_;
    }
    const Entry& operator*() const {
      return entry_;
    }

    void reset() {
      if (registry_ != nullptr) {
        registry_->releaseLease(segId_);
        registry_ = nullptr;
      }
    }

   private:
    TcpSegmentRegistry* registry_{nullptr};
    uint64_t segId_{0};
    Entry entry_{};
  };

  void
  add(uint64_t segId, void* ptr, size_t len, MemoryType memType, int deviceId) {
    std::lock_guard<std::mutex> lk(mu_);
    segs_[segId] = Slot{Entry{ptr, len, memType, deviceId}, 0, false};
  }

  /// Never blocks: a segment being torn down yields an empty lease rather than
  /// waiting, so the reader thread is never parked behind a deregistration.
  Lease find(uint64_t segId) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = segs_.find(segId);
    if (it == segs_.end() || it->second.dying) {
      return Lease{};
    }
    ++it->second.leases;
    return Lease{this, segId, it->second.entry};
  }

  /// Blocks until every outstanding lease on `segId` is released, so that when
  /// this returns no thread is still reading from or writing to the segment and
  /// the owner may free it.
  ///
  /// Callers must hold no other transport lock: this waits on another thread
  /// making progress. No lock cycle is possible -- find() and releaseLease()
  /// are the only other acquirers of mu_ and neither waits, and nothing
  /// acquires mu_ while holding another transport mutex.
  ///
  /// How long this can block is bounded by the slowest lease holder, and on the
  /// VRAM path that is not bounded by the transport. A read reply's lease is
  /// held across its staging copy, and that copy runs on the null stream
  /// (handleFrame passes no stream), which waits on every blocking stream on
  /// the device -- so a caller here can wait for unrelated application GPU
  /// work. A read that arrived with the staging pool exhausted holds its lease
  /// for longer still: its copy has not been issued yet and starts only once a
  /// slab frees up.
  ///
  /// What this can no longer do is deadlock. The reader thread does not wait
  /// for any of it -- staging is asynchronous and exhaustion defers rather than
  /// blocks -- so it keeps delivering inbound frames, including whatever the
  /// application's GPU work is itself waiting on. Giving the staging copies a
  /// dedicated non-blocking stream would remove the remaining wait on unrelated
  /// device work.
  void erase(uint64_t segId) {
    std::unique_lock<std::mutex> lk(mu_);
    auto it = segs_.find(segId);
    if (it == segs_.end()) {
      return;
    }
    // Stop handing out leases first, or a steady stream of inbound frames for
    // this segment could keep the count above zero and starve the wait.
    it->second.dying = true;
    drained_.wait(lk, [this, segId]() {
      auto slot = segs_.find(segId);
      return slot == segs_.end() || slot->second.leases == 0;
    });
    segs_.erase(segId);
  }

 private:
  struct Slot {
    Entry entry;
    int leases{0};
    // Set by erase() so find() stops issuing leases while it drains.
    bool dying{false};
  };

  void releaseLease(uint64_t segId) {
    bool notify = false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = segs_.find(segId);
      if (it == segs_.end()) {
        return;
      }
      if (it->second.leases > 0) {
        --it->second.leases;
      }
      // Only an erase() is ever waiting, so skip the notify on the hot path.
      notify = it->second.dying && it->second.leases == 0;
    }
    if (notify) {
      drained_.notify_all();
    }
  }

  mutable std::mutex mu_;
  std::condition_variable drained_;
  std::unordered_map<uint64_t, Slot> segs_;
};

/// Tracks a single in-flight request awaiting its reply. Shared completion
/// state is aggregated per put()/get() call in TcpOpState.
struct TcpInflight {
  std::shared_ptr<TcpOpState> state;
  void* dst{nullptr}; // READ destination; nullptr for WRITE.
  size_t len{0};
  bool isRead{false};
  MemoryType memType{MemoryType::DRAM};
  int deviceId{-1};
  // Caller's CUDA stream (as void*) for VRAM staging on the get() consumer
  // path, so the H2D is ordered against the framework's stream (not the null
  // stream). nullptr => null stream.
  void* stream{nullptr};
};

/// One chunk of a put(), fully planned before anything is queued. Holding the
/// length separately keeps it readable after `entry` has been moved into the
/// inflight map.
struct PlannedChunk {
  uint64_t reqId{0};
  size_t len{0};
  TcpInflight entry;
};

/// The same chunk resolved down to the addresses a Write frame needs. Built in
/// put()'s pre-flight pass, so the commit pass has no decisions left to make.
struct PlannedPutFrame {
  uint64_t reqId{0};
  uint64_t segId{0};
  uint64_t offset{0};
  const uint8_t* src{nullptr};
  size_t len{0};
  bool vram{false};
  int deviceId{-1};
};

/// One outbound frame's storage: either a plain vector or a pinned staging
/// slab. Move-only, because both kinds own the buffer the socket reads from and
/// a slab additionally owns its place in the pool.
///
/// The vector constructor is deliberately implicit: most frames are small
/// header-only messages built by serializeTcpHeader(), and they should stay
/// written that way.
class TcpFrame {
 public:
  TcpFrame() = default;
  /* implicit */ TcpFrame(std::vector<uint8_t> bytes)
      : vec_(std::move(bytes)), len_(vec_.size()) {}
  /// `len` is the transmitted length, which may be shorter than the slab.
  TcpFrame(TcpPinnedSlab slab, size_t len)
      : slab_(std::move(slab)), len_(len) {}

  ~TcpFrame() = default;
  TcpFrame(TcpFrame&&) = default;
  TcpFrame& operator=(TcpFrame&&) = default;
  TcpFrame(const TcpFrame&) = delete;
  TcpFrame& operator=(const TcpFrame&) = delete;

  /// Writable view over the whole frame, for filling in the header and staging
  /// a payload into it.
  uint8_t* mutableData() {
    return slab_ ? slab_.data() : vec_.data();
  }
  const uint8_t* data() const {
    return slab_ ? slab_.data() : vec_.data();
  }
  size_t size() const {
    return len_;
  }
  std::span<const uint8_t> bytes() const {
    return {data(), len_};
  }

 private:
  std::vector<uint8_t> vec_;
  TcpPinnedSlab slab_;
  size_t len_{0};
};

/// One queued outbound frame. `onSent`, when set, is completed by the sender
/// thread once the frame is flushed to the socket (two-sided send()).
struct TcpOutItem {
  TcpFrame frame;
  std::shared_ptr<TcpOpState> onSent;
};

/// A ReadReply whose payload is still being copied out of VRAM. The frame is
/// already built, header and all, in a pinned staging slab; only the copy into
/// its payload is outstanding.
///
/// The lease lives here rather than on the reader thread. It has to outlive the
/// copy, because it is what stops the owner deregistering and freeing the
/// source buffer underneath the GPU, but nothing requires the reader to be the
/// thread holding it. Moving it here is what lets the reader return to the
/// socket while the segment stays protected.
struct PendingReadReply {
  TcpFrame frame;
  TcpSegmentRegistry::Lease lease;
  /// cudaEvent_t, kept opaque so this header does not need the CUDA runtime.
  void* event{nullptr};
  uint64_t reqId{0};
  /// Needed at teardown to wait on the right device's stream.
  int deviceId{-1};
};

/// One slab-backed get destination copy waiting for its CUDA event. The write
/// reservation and receive slab stay owned here until the GPU has finished
/// reading the source and writing the caller's destination.
struct PendingH2d {
  std::shared_ptr<TcpOpState> state;
  TcpPinnedSlab slab;
  void* event{nullptr};
  int deviceId{-1};
  void* stream{nullptr};
  uint64_t reqId{0};
  std::chrono::steady_clock::time_point launchedAt;
};

/// One wave of put frames whose D2H copies have been launched but not yet
/// waited for. The frames own their staging slabs, so holding this record is
/// what keeps the pinned destination alive while the DMA is still writing it.
///
/// `startIdx` is the wave's first index into put()'s chunk plan, so a failure
/// here can abandon exactly the admissions from this wave onward.
///
/// Single-device by construction: put() breaks a wave at a device boundary, so
/// one event recorded on `deviceId` covers every copy in it.
struct PendingPutWave {
  std::vector<TcpFrame> frames;
  /// cudaEvent_t, kept opaque so this header does not need the CUDA runtime.
  void* event{nullptr};
  int deviceId{-1};
  /// The caller's stream, so the fallback wait covers the stream the copies
  /// were launched on rather than whatever the null stream happens to hold.
  void* stream{nullptr};
  size_t startIdx{0};
  size_t bytes{0};
  std::chrono::steady_clock::time_point launchedAt;
};

/// Shared independently of TcpTransport so queued EventBase callbacks never
/// retain or dereference a transport that shutdown has already destroyed.
struct H2dPollState {
  std::mutex mu;
  std::condition_variable drained;
  std::deque<PendingH2d> pending;
  std::shared_ptr<TcpOpState> retiringState;
  // There are exactly two receive slabs, so at most two copies can become
  // unquiesceable. Fixed slots avoid allocating on this driver-failure path.
  std::array<std::optional<PendingH2d>, 2> quarantined;
  std::shared_ptr<H2dPollState> quarantineKeepalive;
  bool pollScheduled{false};
  bool stopping{false};
  size_t activeRetirements{0};
  EventBase* evb{nullptr};
  std::shared_ptr<CudaApi> cudaApi;
  std::atomic<uint64_t> copyNs{0};
  std::atomic<uint64_t> copyCount{0};
};

/// A VRAM ReadReply that has not been started because no staging slab was free.
/// Everything needed to build and launch it later is here, so the reader can
/// record it and go straight back to the socket.
///
/// The lease is held even though no copy has been issued yet. It is what keeps
/// the source buffer registered, and the point of deferring rather than
/// dropping is that this read will still be answered out of that same buffer. A
/// deregistration therefore waits for the deferred copy to run, not only for
/// the ones already launched.
struct DeferredReadReply {
  TcpSegmentRegistry::Lease lease;
  uint64_t reqId{0};
  uint64_t segId{0};
  uint64_t offset{0};
  size_t len{0};
};

/// A posted recv() awaiting a matching inbound SEND frame.
struct TcpPendingRecv {
  void* dst{nullptr};
  size_t cap{0};
  std::shared_ptr<TcpOpState> state;
  MemoryType memType{MemoryType::DRAM};
  int deviceId{-1};
  // Caller's CUDA stream (as void*) for the H2D that lands the payload, so the
  // copy is ordered against the framework's stream rather than the null stream.
  // Retained here because the matching SEND may not arrive until long after
  // recv() returned, by which point the caller's RequestOptions is gone.
  // Mirrors TcpInflight::stream on the get() path.
  void* stream{nullptr};
};

/// Native, self-contained TCP data transport.
///
/// Establishes `numSocketsPerDevice` full-duplex data connections ("lanes") per
/// bound device
/// (deterministic listener/dialer role by host:port ordering, lane identity
/// from an explicit hello). Per connection:
///  - reader, one per lane: blocking recv + demultiplex; never blocks on a
///  send,
///    so it always drains inbound (avoids the mutual-READ deadlock).
///  - sender, one per transport: drains an outbound frame queue and performs
///  all
///    socket sends (requests, ACKs, READ replies), serialized by construction.
///    Sends every frame on lane 0 -- the extra lanes are established and
///    drained but not yet used outbound, so throughput is unchanged until
///    striping schedules across them.
///
/// put() copies a local buffer into a peer segment (WRITE + ACK). get() is
/// emulated as a pull: READ_REQUEST -> peer replies READ_REPLY with the data.
///
/// Independent of NIXL/UCX/folly. send()/recv() implement a two-sided
/// rendezvous (FIFO-matched via pendingRecvs_/unmatchedSends_).
class TcpTransport : public Transport {
 public:
  TcpTransport(
      int deviceId,
      EventBase* evb,
      std::shared_ptr<TcpSegmentRegistry> registry,
      TcpTransportConfig config = {},
      std::string host = "127.0.0.1",
      std::shared_ptr<CudaApi> cudaApi = nullptr);

  ~TcpTransport() override;

  TcpTransport(const TcpTransport&) = delete;
  TcpTransport& operator=(const TcpTransport&) = delete;
  TcpTransport(TcpTransport&&) = delete;
  TcpTransport& operator=(TcpTransport&&) = delete;

  const std::string& name() const noexcept override {
    return name_;
  }

  TransportType transportType() const noexcept override {
    return TransportType::TCP;
  }

  TransportState state() const noexcept override {
    return state_;
  }

  bool asyncGetH2dEnabled() const noexcept {
    return config_.asyncGetH2d;
  }

  TransportInfo bind() override;
  Status connect(std::span<const uint8_t> remoteInfo) override;

  /// Writes the local spans into the peer's registered segments.
  ///
  /// A failed put() promises nothing about the peer's memory. Treat the
  /// destination as undefined until a later put over it succeeds, and do not
  /// read it in between:
  ///
  /// - Any put may partially apply. Work is staged and queued in waves of at
  ///   most kMaxPutWaveChunks chunks -- fewer where a wave meets a device
  ///   boundary or a DRAM chunk -- and a failure in wave N+1 leaves wave N
  ///   queued, and queued frames are frames the peer applies.
  /// - What did apply is not a prefix. Frames are striped across lanes, which
  ///   are independent sockets that fail independently, so a break leaves an
  ///   arbitrary subset with holes at offsets the caller cannot determine.
  ///   There is no offset to resume from; the only recovery is to re-put the
  ///   whole range.
  /// - The peer is never told. It applies each frame as it arrives and has no
  ///   notion of the put it belonged to, so its application can read a
  ///   half-written segment and see nothing wrong. Only the caller learns of
  ///   the failure, and not how much of it landed.
  /// - A host-staging failure is contained to the wave it happened in, so it
  ///   cannot put anything on the wire that a successful wave had not already
  ///   sent. That bounds one failure mode; it is not a transaction, and it
  ///   bounds nothing at the sizes callers actually use.
  ///
  /// A real abort needs a wire-level change, tracked in T285791201.
  std::future<Status> put(
      std::span<const TransferRequest> requests,
      const RequestOptions& options = {}) override;

  std::future<Status> get(
      std::span<const TransferRequest> requests,
      const RequestOptions& options = {}) override;

  std::future<Status> send(
      RegisteredSegment::Span src,
      const RequestOptions& options = {}) override;

  std::future<Status> recv(
      RegisteredSegment::Span dst,
      const RequestOptions& options = {}) override;

  std::future<Status> send(
      Segment::Span src,
      const RequestOptions& options = {}) override;

  std::future<Status> recv(
      Segment::Span dst,
      const RequestOptions& options = {}) override;

  void shutdown() override;

 private:
  // Needs handleFrame() plus the outbound queue to drive peer-supplied frames
  // directly and assert the bounds checks reject them. Those checks are the
  // memory-safety boundary of this transport, so they are worth pinning.
  friend class TcpTransportFrameTest;
  friend class TcpReceivePoolTest;

  Result<const TcpRemoteRegistrationHandle*> findRemoteHandle(
      const RemoteRegisteredSegment::Span& span) const;

  // Reader thread, one per lane: blocking recv + demultiplex on that lane's
  // socket until it closes. Takes the lane index rather than reading a single
  // member, because each reader owns exactly one socket.
  void readerLoop(size_t laneIdx) noexcept;

 public:
  /// Logs the get-path phase split (first-byte / drain / destination copy) and
  /// zeroes it so callers can bracket a single measurement.
  void logAndResetPhaseStats(std::string_view label);

 private:
  // Sender thread, one per lane: drains that lane's queue and performs its
  // socket sends. One writer per socket, which is what keeps TcpConn's
  // single-writer requirement satisfied.
  void senderLoop(size_t laneIdx) noexcept;
  // Handles one fully-received inbound frame (reader thread). `receiveSlab`
  // owns frame.data() when the reader received directly into pinned memory.
  void handleFrame(
      std::span<const uint8_t> frame,
      TcpPinnedSlab receiveSlab = {}) noexcept;
  // The body of handleFrame(), allowed to throw. Peer-supplied lengths size
  // allocations in here and the VRAM staging path throws on an invalid
  // deviceId, so handleFrame() is the boundary that stops either reaching the
  // top of the reader thread.
  void handleFrameImpl(
      std::span<const uint8_t> frame,
      TcpPinnedSlab receiveSlab);
  // Queues one fire-and-forget framed message for the sender thread.
  // `mayBlock` must be true only for caller threads. Returns false if the frame
  // was not queued (transport closing, or the reader hit the cap and refused
  // the connection).
  [[nodiscard]] bool enqueueFrame(TcpFrame frame, bool mayBlock);
  // Queues a run of frames as one indivisible step: either every frame is
  // queued or none is. Enqueuing them one at a time would let a sender thread
  // start transmitting a partially built group, which for a multi-chunk put
  // means the peer applies a prefix of a transfer this side may still fail to
  // finish staging.
  //
  // The whole group therefore goes on ONE lane, so a single lock makes it
  // atomic. Spreading a group across lanes would need every target lane's mutex
  // held at once, and the room-waiting inside would then be a deadlock: a
  // producer holding lanes 0..k-1 while waiting for room on lane k blocks the
  // very senders that would free it.
  //
  // `mayBlock` behaves as in enqueueFrame, and is judged against the group's
  // total size, so a group larger than the queue cap still drains rather than
  // waiting forever on room that will never appear.
  [[nodiscard]] bool enqueueFrames(std::vector<TcpFrame> frames, bool mayBlock);
  // Queues a framed message whose completion is reported via `onSent` once the
  // frame is flushed (two-sided send()).
  void enqueueSendFrame(TcpFrame frame, std::shared_ptr<TcpOpState> onSent);
  // Registers an in-flight request, but only if teardown has not already swept
  // the map. Returns false if the transport broke first, in which case the
  // caller must fail the op: nothing else will ever resolve it.
  //
  // The entry-time state_/connBroken_ checks in put()/get() are not enough on
  // their own -- a failAllPending() can land between that check and this
  // insert, leaving an entry no sweep will ever see and a caller blocked in
  // future.get() forever. Checking connBroken_ *under* inflightMu_ makes the
  // two outcomes exclusive: either the insert precedes the sweep and the sweep
  // fails it, or the sweep precedes the insert and this returns false.
  Status admitInflight(uint64_t reqId, TcpInflight entry);
  // Admits every chunk of one put() or none. A per-chunk reservation that runs
  // out of slots partway through has already delivered the earlier chunks to
  // the peer, which is a partial remote write the caller is told nothing about.
  Status admitInflightBulk(std::span<PlannedChunk> chunks);
  // Drops reservations for chunks that were never queued, from `fromIdx` on.
  // Chunks already handed to enqueueFrame keep theirs, so their Acks still
  // match.
  void abandonInflight(std::span<const PlannedChunk> chunks, size_t fromIdx);
  // Confirms a device can be selected before any frame is queued.
  // CudaDeviceGuard throws on an unusable device, and from inside the send loop
  // that exception would escape put() itself, abandoning the promise while
  // whatever was already queued still lands on the peer.
  Status validateDeviceForStaging(int deviceId);
  // Answers one VRAM ReadRequest. Acquires a staging slab without waiting; if
  // none is free the request is deferred rather than staged, so the reader is
  // never parked on the pool. Consumes the lease.
  Status respondToVramRead(
      const TcpMsgHeader& replyHeader,
      TcpSegmentRegistry::Lease lease);
  // Builds the reply in `slab` and starts its D2H copy, handing the frame to
  // the staging queue so the reader can go back to draining the socket instead
  // of waiting on the device. Consumes the lease and the slab. Returns an error
  // only if the copy could not be started, in which case nothing was queued.
  Status startReadReply(
      const TcpMsgHeader& replyHeader,
      TcpSegmentRegistry::Lease lease,
      TcpPinnedSlab slab);
  // Stages one wave of VRAM Write frames in two halves, so a wave's copies can
  // run while the previous wave is being queued and sent. A single call that
  // launched and waited kept the copy engine idle for the whole
  // queue-and-transmit gap, which measured as ~55% of a VRAM put's wall time
  // spent staging against a D2H path twice as fast as the put itself.
  //
  // launchPutWave takes a slab per chunk, issues every copy, and records one
  // event -- correct with one event because put() never forms a wave spanning
  // devices. It blocks until the pool can hand out the whole wave, so it must
  // not be called holding outMu_: the thread that frees those slabs is the
  // sender, and it needs that mutex to make progress. That block is also the
  // backpressure bounding how many waves can be in flight.
  //
  // The returned record owns the slabs, so the caller must either retire it or
  // abandon it. Dropping one outright while its copies still run would hand a
  // slab back under an active DMA, and the next staging copy would then race
  // the GPU over the same bytes.
  Result<PendingPutWave> launchPutWave(
      std::span<const PlannedPutFrame> wave,
      void* stream,
      size_t startIdx);
  // Waits for one launched wave's copies to finish and destroys its event.
  // Waits on the event rather than the stream, because the stream also carries
  // later waves' copies and waiting for those would give back the overlap.
  //
  // Leaves the frames in place for the caller to queue, so a failure here can
  // still drop them before any of them reaches a lane.
  Status retirePutWave(PendingPutWave& wave);
  // Waits out every launched wave without queueing any of it, for the unwind
  // paths. The wait is not optional: the slabs cannot be released while the
  // copies are still writing into them.
  void abandonPutWaves(std::deque<PendingPutWave>& waves) noexcept;
  // Records a VRAM read to start once a slab frees up. Fails the request (the
  // caller answers with an Error frame) if the deferred queue is full: dropping
  // it silently would leave the peer waiting forever, and blocking here would
  // stop the reader draining the socket.
  Status deferReadReply(
      const TcpMsgHeader& replyHeader,
      TcpSegmentRegistry::Lease lease);
  // Starts as many deferred reads as there are free slabs, oldest first. Runs
  // on the EventBase, never inline at the point a slab is released: that point
  // is the sender thread, and launching copies there would block the drain that
  // frees the next slab.
  void startDeferredReadReplies();
  // Kicks startDeferredReadReplies() on the EventBase. Called wherever a
  // staging slab goes back to the pool. Cheap and safe when nothing is
  // deferred.
  void scheduleDeferredReadReplies();
  // The staging pool, created on first use. Building it in the constructor
  // would charge every peer ~64 MiB of pinned host memory -- there is one
  // transport per peer -- including the DRAM-only peers that never stage at
  // all.
  Result<std::shared_ptr<TcpPinnedSlabPool>> stagingPool();
  // The independent inbound pool. It is created only for async-enabled,
  // non-zero VRAM get(), and is sized to the wire cap because recv(span)
  // consumes the length prefix before it can reject an undersized buffer.
  std::shared_ptr<TcpPinnedSlabPool> ensureReceivePool();
  // Reader-only lookup: never allocates and never blocks.
  std::shared_ptr<TcpPinnedSlabPool> receivePoolIfCreated();
  // Starts a slab-backed get destination copy. On success the pending queue
  // owns `slab` and the caller's write reservation until event retirement.
  Status startAsyncH2d(
      const TcpInflight& entry,
      std::span<const uint8_t> payload,
      TcpPinnedSlab slab,
      uint64_t reqId);
  void schedulePendingH2dPoll();
  static void pollPendingH2d(std::shared_ptr<H2dPollState> state) noexcept;
  static Status waitForH2dCopy(
      const std::shared_ptr<H2dPollState>& state,
      int deviceId,
      void* stream) noexcept;
  static void destroyH2dEvent(
      const std::shared_ptr<H2dPollState>& state,
      int deviceId,
      void* event) noexcept;
  static void quarantineH2d(
      const std::shared_ptr<H2dPollState>& state,
      PendingH2d copy) noexcept;
  void drainPendingH2d();
  // Retires staged replies whose copy has finished, oldest first. Runs on the
  // EventBase, never on the reader thread: polling from the reader would put
  // the head-of-line block straight back.
  void pollPendingReadReplies();
  // Kicks the poll loop if it is not already running. Safe from any thread.
  void schedulePendingReplyPoll();
  /// Waits for a staged D2H copy on `deviceId` whose completion is otherwise
  /// unknown, so the frame it targets can be released. Used on the paths where
  /// a copy was issued but the event never became a usable completion signal.
  void waitForStagedCopy(int deviceId) noexcept;

  /// Waits for outstanding staging copies and drops the frames, then discards
  /// whatever was still deferred. Called during teardown, because the GPU may
  /// still be writing into a pending frame's payload and that memory must not
  /// be freed underneath it.
  void drainPendingReadReplies();

  // Shared bodies for the zero-copy and copy-based send/recv overloads.
  // `options` is threaded through rather than dropped so a VRAM caller's stream
  // reaches the host-staging copy, as put()/get() already do.
  std::future<Status> sendImpl(
      const void* data,
      size_t len,
      MemoryType memType,
      int deviceId,
      const RequestOptions& options);
  std::future<Status> recvImpl(
      void* dst,
      size_t cap,
      MemoryType memType,
      int deviceId,
      const RequestOptions& options);
  // Host-staging copies used for VRAM segments (plain memcpy for DRAM):
  // hostFromDevice = D2H, deviceFromHost = H2D. Synchronous. `stream` (a
  // cudaStream_t as void*, nullptr => null stream) orders the copy against the
  // caller's framework stream, matching the RDMA CopyEngine.
  Status hostFromDevice(
      void* hostDst,
      const void* devSrc,
      size_t len,
      int deviceId,
      void* stream = nullptr);
  Status deviceFromHost(
      void* devDst,
      const void* hostSrc,
      size_t len,
      int deviceId,
      void* stream = nullptr);
  // Fails every pending put/get/recv/queued-send; marks the conn broken.
  void failAllPending(const char* message);
  // Closes every lane's socket, each at most once for the lifetime of the
  // transport. Both the reader threads and shutdown() refuse the connection, so
  // this has concurrent callers. Any lane failing takes the whole transport
  // down, so there is no partial-close path.
  void closeLanesOnce();
  // Marks every lane's outbound queue closed and wakes its waiters. Used by
  // shutdown() and by a sender whose send failed, since one failed lane takes
  // the whole transport down.
  void closeAllLaneQueues();
  // Establishes `laneCount` data sockets, either by accepting or by dialing,
  // and fills lanes_ so that index i on one peer is index i on the other.
  Status establishLanes(
      bool listener,
      const TcpTransportInfo& peer,
      size_t laneCount,
      std::chrono::seconds handshakeTimeout);
  // Lane 0's socket, or nullptr before connect(). Phase 1 sends everything on
  // lane 0 and reports its stats.
  controller::Conn* primaryConn() const;

  [[maybe_unused]] int deviceId_{-1};
  EventBase* evb_{nullptr};
  std::shared_ptr<TcpSegmentRegistry> registry_;
  std::shared_ptr<CudaApi> cudaApi_;
  TcpTransportConfig config_;
  std::string name_{"tcp"};
  // Atomic: written by bind()/connect()/shutdown() and read by the
  // put/get/send/recv guards and state(), which callers may invoke from any
  // thread (e.g. a shutdown racing an in-flight transfer). A plain member here
  // is a data race, matching why running_/connBroken_ below are atomic too.
  std::atomic<TransportState> state_{TransportState::Disconnected};

  std::string host_{"127.0.0.1"};
  uint16_t port_{0};

  /// One listener per entry in TcpTransportConfig::bindToDevices, or exactly
  /// one unbound listener when that list is empty. servers_[d] owns the lanes
  /// with index % servers_.size() == d.
  std::vector<std::unique_ptr<controller::AsyncTcpServer>> servers_;

  /// Endpoints advertised by bind(), parallel to servers_.
  std::vector<TcpTransportInfo::Endpoint> localEndpoints_;

  /// One data socket plus the reader thread that drains it. Phase 1 establishes
  /// the configured lanes but sends everything on lane 0, so only
  /// establishment, reader topology and teardown differ from the single-socket
  /// path.
  struct TcpLane {
    std::unique_ptr<controller::Conn> conn;
    std::thread reader;
    std::thread sender;
    // Gates this lane's close(). Conn::close() is a check-then-act on a bare
    // fd, so two callers seeing it open both close it, and the second reaps a
    // descriptor some unrelated thread may already have been handed.
    std::atomic<bool> closed{false};

    // This lane's outbound queue, drained by its own sender. Per-lane rather
    // than shared so the senders never contend on one mutex -- a shared queue
    // with N senders would serialise exactly the work striping exists to
    // parallelise, and would also let one frame of a group be picked up while
    // the rest are still being queued.
    std::mutex mu;
    std::condition_variable cv;
    std::deque<TcpOutItem> queue;
    size_t bytes{0};
    bool outClosed{false};
  };
  // Indirected because TcpLane holds an atomic and a thread, so it is neither
  // copyable nor movable and the vector must never have to relocate it. Written
  // only by connect() under lifecycleMu_, and read by the reader threads, the
  // sender thread and shutdown() -- all of which run only after connect() has
  // installed every lane.
  std::vector<std::unique_ptr<TcpLane>> lanes_;

  // get-path copy accounting, paired with Conn::RecvPhaseStats. Reader thread
  // only; relaxed because a torn read across a reset misattributes a sample and
  // nothing more.
  std::atomic<uint64_t> dstCopyNs_{0};
  std::atomic<uint64_t> dstCopyCount_{0};
  std::atomic<uint64_t> receiveSlabAttempts_{0};
  std::atomic<uint64_t> receiveSlabMisses_{0};
  std::atomic<uint64_t> vectorReceiveCount_{0};
  // put-path staging accounting. `stagingNs_` covers a wave's residency: from
  // the memcpyAsync launches to the event wait that retires them. Because waves
  // now overlap, that span includes time the next wave was being launched in,
  // so the per-wave intervals overlap each other and their sum can exceed wall
  // time. Read it as how long a wave stays in flight, NOT as D2H bandwidth --
  // bytes/ns here is no longer a rate the device achieves. For the real D2H
  // figure see the commit that added these counters, which measured it at 43-47
  // GB/s while staging was still serialised against transmit.
  //
  // Written only by caller threads inside retirePutWave; relaxed because a torn
  // read misreports one sample.
  std::atomic<uint64_t> stagingNs_{0};
  std::atomic<uint64_t> stagingBytes_{0};
  std::atomic<uint64_t> stagingWaves_{0};
  std::atomic<uint64_t> stagingChunks_{0};
  std::shared_ptr<H2dPollState> h2dState_;

  // Serialises bind()/connect()/shutdown(), which together own server_,
  // lanes_ and their threads. Without it a shutdown() concurrent with an
  // in-progress connect() races a unique_ptr and two std::thread objects, and
  // -- because connect() parks for the whole handshake -- can let connect()
  // install a live connection and both threads *after* shutdown() has already
  // torn down and returned.
  //
  // Held for the entire body of each of the three, including connect()'s
  // handshake wait, so a concurrent shutdown() waits out the handshake (bounded
  // by config_.connTimeout) instead of interleaving with it. Deliberately NOT
  // taken by put/get/send/recv, which keep reading state_/connBroken_
  // atomically so a transfer is never serialised against a connect. Safe to
  // hold across the per-lane reader/sender joins in shutdown(): neither loop,
  // nor failAllPending(), ever acquires it. Bounds on the outbound queue and
  // the in-flight map, for work this side originates: the lane queues hold
  // whole payloads, and inflight_ one entry per outstanding chunk, so an
  // application issuing put/get faster than the link drains would otherwise
  // grow both without limit. Caller threads wait for room, which is real
  // backpressure -- the application slows down and nothing breaks.
  //
  // kMaxOutQueueBytes deliberately does NOT apply to the reader thread's
  // replies. A get of N bytes legitimately requires up to N bytes of ReadReply
  // queued, and N is bounded only by the segment size, so no fixed cap can
  // distinguish a large honest get from abuse. The reader also cannot wait
  // without ceasing to drain the socket, which is the mutual-READ deadlock the
  // reader/sender split exists to avoid. Bounding the reply side properly needs
  // credit-based flow control in the wire protocol; until then the drain rate
  // is what bounds it.
  static constexpr size_t kMaxOutQueueBytes = 64UL * 1024 * 1024;
  static constexpr size_t kMaxInflightRequests = 4096;
  // put()'s chunk size, and the payload capacity of one staging slab. A
  // same-version peer therefore never asks for a read larger than one slab.
  //
  // The controller frames each TcpConn message with a 4-byte length and caps it
  // at 64 MiB; a chunk (header + payload) stays safely under that, so large
  // put/get transfers are split across multiple frames.
  static constexpr size_t kMaxChunkSize = 4UL * 1024 * 1024;
  // Floor for adaptive get chunking. Splitting below this loses more to
  // per-frame cost than it gains from the extra lane: at a 4 MiB get, 512 KiB
  // and 1 MiB chunks measured the same, 256 KiB gave up part of the gain, and
  // 128 KiB was worse than not splitting at all.
  static constexpr size_t kMinAdaptiveChunkSize = 512UL * 1024;

  // Chunk size for one get request. A transfer no larger than kMaxChunkSize is
  // a single frame, and a frame goes to a single lane, so it uses one lane
  // however many are configured. Splitting it across lanes is worth up to 1.44x
  // at 4 MiB.
  //
  // Only transfers that would otherwise be one frame are split. Larger ones
  // already span lanes, and measured worse when chunked smaller -- past a few
  // frames per-frame cost dominates and the largest chunk wins.
  //
  // Requester-side only, so it needs no protocol change: replies stay within
  // kMaxChunkSize, which is all a peer enforces. The count in get() and the
  // request loop must derive their chunk from this same function or the reply
  // count will not match what the operation waits for.
  static size_t adaptiveGetChunk(size_t len, size_t laneCount);
  // Withheld from put(), so a saturated put cannot leave the get responder with
  // nothing to stage into. The responder is the side the peer is already
  // blocked on.
  static constexpr size_t kStagingSlabsReservedForReader = 1;
  // Chunks staged, and so queued, as one indivisible wave. Everything put()
  // promises about partial writes is bounded by this: a put of at most this
  // many chunks either reaches the queue whole or not at all.
  //
  // 28 MiB, well under kMaxOutQueueBytes so a whole wave is admissible to the
  // queue in one step rather than being refused by its cap. This is the primary
  // knob of the three: the staging pool is sized from it below, not the other
  // way round.
  static constexpr size_t kMaxPutWaveChunks = 7;
  // How many launched-but-unretired waves put() keeps. This is what buys the
  // overlap: at 1 the caller waits for a wave's copies immediately after
  // launching them, so the copy engine idles for the whole queue-and-transmit
  // gap between waves. At 2 the next wave's copies are already running while
  // the previous one is being waited for, queued and sent.
  static constexpr size_t kMaxPutWavesInFlight = 2;
  // Sized so acquire() does not block a caller that is holding a fully staged
  // wave. At the moment put() acquires for a new wave, three things are already
  // outstanding: the kMaxPutWavesInFlight - 1 waves it holds in its own deque,
  // up to kMaxOutQueueBytes worth of frames it has queued but the sender has
  // not drained yet, and the wave it is asking for -- plus the reader's
  // reservation, which acquire() counts on top of the request.
  //
  // Getting this wrong is not a small loss, which is why it is derived rather
  // than picked. At 16 slabs the arithmetic leaves acquire() 2 free against the
  // 8 it needs, so every wave blocks -- and blocks while withholding a wave of
  // already-copied frames from the sender, because they are only queued on the
  // next trip round put()'s loop. Measured on two hosts at 1 GiB that was 16.9
  // GB/s against 22.9 for waiting on each wave in turn: 26% worse than not
  // overlapping at all. Sized from the invariant it measured 32.0.
  //
  // ~124 MiB of pinned host memory per peer that actually stages. The pool is
  // created on first use, so DRAM-only peers still pay nothing for it.
  static constexpr size_t kStagingSlabCount =
      (kMaxPutWavesInFlight - 1) * kMaxPutWaveChunks +
      kMaxOutQueueBytes / kMaxChunkSize + kMaxPutWaveChunks +
      kStagingSlabsReservedForReader;
  static_assert(
      kStagingSlabCount >= kMaxPutWavesInFlight * kMaxPutWaveChunks +
              kStagingSlabsReservedForReader,
      "the pool must hold every wave the window keeps in flight at once, or "
      "put() deadlocks against slabs only it can release");
  // Two full wire frames can remain pinned while Phase 3d retires H2D copies;
  // exhaustion falls back to the reusable vector instead of blocking receive.
  static constexpr size_t kReceiveSlabCount = 2;
  // Per-lane queue cap, so the aggregate bound stays kMaxOutQueueBytes however
  // many lanes are configured. An empty lane admits any frame regardless (see
  // enqueueFrame), so dividing cannot make an oversized frame undrainable.
  size_t laneCapBytes() const;
  // Round-robins a lane for one frame. put/get frames each carry their own
  // reqId/segId/offset, so the peer reassembles them without ordering help and
  // any lane will do.
  size_t pickLane();

  // Cap on bytes buffered in unmatchedSends_. The reader thread never stops
  // draining the socket -- that is what avoids the mutual-READ deadlock -- so
  // the socket buffer applies no backpressure and a peer sending faster than
  // the local side posts recvs would grow this deque without limit, each entry
  // up to the wire-frame cap. One max-size frame may sit buffered; beyond that
  // the connection is refused rather than absorbed.
  static constexpr size_t kMaxUnmatchedSendBytes = 64UL * 1024 * 1024;
  // Total bytes currently held in unmatchedSends_. Guarded by recvMu_.
  size_t unmatchedBytes_{0};

  std::mutex lifecycleMu_;
  // Set once by shutdown() under lifecycleMu_. Makes teardown one-shot and
  // stops a later bind()/connect() from resurrecting a shut-down transport.
  // Checked under the mutex rather than short-circuiting before it, so a second
  // shutdown() (the normal MultiTransport-then-destructor path) waits for the
  // first to finish rather than returning while teardown is still running.
  std::atomic<bool> shutdown_{false};

  // Round-robin cursor for lane selection. Relaxed: an occasional duplicate or
  // skipped index only perturbs balance, and nothing reads it for correctness.
  std::atomic<uint64_t> nextLane_{0};
  std::atomic<bool> running_{false};
  std::atomic<bool> connBroken_{false};

  // Guards the staging queue, which the reader thread appends to and the
  // EventBase drains.
  std::mutex stagingMu_;
  std::deque<PendingReadReply> pendingReplies_;
  // Retired strictly front-first. Copies for one device go on that device's
  // stream and so signal in issue order; across devices they are independent,
  // so a reply that is ready can wait behind an older one still running.
  // Ordered retirement is still correct -- offsets make out-of-order replies
  // safe on the wire, this only costs latency -- and a transport serves one
  // peer, which in practice means one device.
  bool replyPollScheduled_{false};
  // VRAM reads that arrived with the pool exhausted, oldest first. Bounded by
  // kMaxInflightRequests for the same reason inflight_ is: it grows on
  // peer-supplied frames, and the reader will not stop reading them.
  std::deque<DeferredReadReply> deferredReplies_;

  // Created on first VRAM staging need, then never replaced. Its own mutex
  // rather than stagingMu_: acquiring a slab can wait (put(), from a caller
  // thread), and the staging queue must stay available to the reader while it
  // does.
  std::mutex poolMu_;
  std::shared_ptr<TcpPinnedSlabPool> slabPool_;

  // Separate from the outbound/responder staging pool: receive slabs must hold
  // any legal wire frame, not just one kMaxChunkSize payload.
  std::mutex receivePoolCreateMu_;
  // Accessed with the std::atomic_load/store free functions, which support
  // shared_ptr before std::atomic<shared_ptr> is available in the toolchain.
  std::shared_ptr<TcpPinnedSlabPool> receiveSlabPool_;
  bool receivePoolUnavailable_{false};

  std::mutex inflightMu_;
  std::unordered_map<uint64_t, TcpInflight> inflight_;
  std::atomic<uint64_t> nextReqId_{1};

  // Two-sided send/recv rendezvous (FIFO-matched on the single stream):
  // inbound SEND frames waiting for a posted recv, and posted recvs waiting
  // for an inbound SEND frame.
  std::mutex recvMu_;
  std::deque<TcpPendingRecv> pendingRecvs_;
  std::deque<std::vector<uint8_t>> unmatchedSends_;
};

class TcpTransportFactory : public TransportFactory {
 public:
  static Status supported();

  // @param host  Local IP the transport binds to and advertises to peers.
  //              Defaults to loopback; pass a routable IP (e.g. an eth2 IPv6)
  //              for cross-host transfers.
  explicit TcpTransportFactory(
      int deviceId,
      EventBase* evb,
      TcpTransportConfig config = {},
      std::string host = "127.0.0.1",
      std::shared_ptr<CudaApi> cudaApi = nullptr);

  Result<std::unique_ptr<RegistrationHandle>> registerSegment(
      Segment& segment) override;

  Result<std::unique_ptr<RemoteRegistrationHandle>> importSegment(
      size_t segmentLength,
      std::span<const uint8_t> payload) override;

  Result<std::unique_ptr<Transport>> createTransport(
      std::span<const uint8_t> peerTopology) override;

  std::vector<uint8_t> getTopology() override;

  Status canConnect(std::span<const uint8_t> peerTopology) override;

 private:
  int deviceId_{-1};
  EventBase* evb_{nullptr};
  TcpTransportConfig config_;
  std::string host_{"127.0.0.1"};
  std::shared_ptr<CudaApi> cudaApi_;
  std::shared_ptr<TcpSegmentRegistry> registry_{
      std::make_shared<TcpSegmentRegistry>()};
  std::atomic<uint64_t> nextSegId_{1};
};

} // namespace uniflow
