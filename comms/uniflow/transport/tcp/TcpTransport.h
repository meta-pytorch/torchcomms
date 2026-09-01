// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <string>
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

  /// Copies a reply payload into the caller's destination buffer (`write`) and
  /// records that chunk's completion as one atomic step.
  ///
  /// Resolving this op's promise is what releases the caller to free or reuse
  /// the destination, so a write into it may only start while the promise is
  /// unresolved and must keep it unresolved until the write finishes. Holding
  /// `mu` across `write` gives both: a concurrent fail() (send error, peer
  /// disconnect, shutdown) either resolves the op first, in which case `write`
  /// is skipped because the destination may already be gone, or it blocks
  /// until the write is done.
  ///
  /// `mu` is a leaf lock: nothing in this struct acquires another mutex while
  /// holding it, and no caller may hold one of the transport's container
  /// mutexes (`inflightMu_`, `recvMu_`, `outMu_`) when calling in. `write`
  /// therefore must only touch the destination buffer -- a memcpy or a device
  /// copy -- and never reach back into the transport's queues.
  template <typename Fn>
  void writeAndComplete(Fn&& write) {
    std::lock_guard<std::mutex> lk(mu);
    if (done) {
      return;
    }
    Status status = write();
    if (status.hasError()) {
      failLocked(std::move(status));
    } else {
      completeOneLocked();
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
    if (remaining == 0) {
      done = true;
      promise.set_value(Ok());
    }
  }

  void failLocked(Status status) {
    if (done) {
      return;
    }
    done = true;
    promise.set_value(std::move(status));
  }
};

struct TcpTransportInfo {
  struct __attribute__((packed)) Header {
    uint16_t port{0};
    uint16_t hostLen{0};
  };

  std::string host{"127.0.0.1"};
  uint16_t port{0};

  TransportInfo serialize() const;
  static Result<TcpTransportInfo> deserialize(std::span<const uint8_t> data);
};

class CudaApi;

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
  /// VRAM path that is not bounded by the transport. The reader queues its Ack
  /// and reply frames without blocking, but it holds its lease across the whole
  /// host-staging copy, and that copy ends in a synchronize on the null stream
  /// (handleFrame passes no stream), which waits on every blocking stream on
  /// the device -- i.e. on unrelated application GPU work. So a caller here can
  /// wait for an application step, and deadlocks outright if that GPU work is
  /// itself gated on a later inbound frame, since the parked reader is the
  /// thread that would deliver it. Giving the staging copies a dedicated
  /// non-blocking stream removes both; until then a DRAM-only deployment is the
  /// safe case, because there the copy under lease is a plain memcpy.
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

/// A ReadReply whose payload is still being copied out of VRAM. The frame is
/// already built, header and all; only the staging copy into its payload is
/// outstanding.
///
/// The lease lives here rather than on the reader thread. It has to outlive the
/// copy, because it is what stops the owner deregistering and freeing the
/// source buffer underneath the GPU, but nothing requires the reader to be the
/// thread holding it. Moving it here is what lets the reader return to the
/// socket immediately while the segment stays protected.
struct PendingReadReply {
  std::vector<uint8_t> frame;
  TcpSegmentRegistry::Lease lease;
  /// cudaEvent_t, kept opaque so this header does not need the CUDA runtime.
  void* event{nullptr};
  uint64_t reqId{0};
  /// Needed at teardown to wait on the right device's stream.
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
/// Establishes one full-duplex data connection per peer (deterministic
/// listener/dialer role by host:port ordering). Three threads run per
/// connection:
///  - reader: blocking recv + demultiplex; never blocks on a send, so it always
///    drains inbound (avoids the mutual-READ deadlock).
///  - sender: drains an outbound frame queue and performs all socket sends
///    (requests, ACKs, READ replies), serialized by construction.
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
      controller::TcpSocketConfig config = {},
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

  TransportInfo bind() override;
  Status connect(std::span<const uint8_t> remoteInfo) override;

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

  Result<const TcpRemoteRegistrationHandle*> findRemoteHandle(
      const RemoteRegisteredSegment::Span& span) const;

  // Reader thread: blocking recv + demultiplex until the connection closes.
  void readerLoop() noexcept;
  // Sender thread: drains outQueue_ and performs all socket sends.
  void senderLoop() noexcept;
  // Handles one fully-received inbound frame (reader thread).
  void handleFrame(std::span<const uint8_t> frame) noexcept;
  // The body of handleFrame(), allowed to throw. Peer-supplied lengths size
  // allocations in here and the VRAM staging path throws on an invalid
  // deviceId, so handleFrame() is the boundary that stops either reaching the
  // top of the reader thread.
  void handleFrameImpl(std::span<const uint8_t> frame);
  // Queues one fire-and-forget framed message for the sender thread.
  // `mayBlock` must be true only for caller threads. Returns false if the frame
  // was not queued (transport closing, or the reader hit the cap and refused
  // the connection).
  [[nodiscard]] bool enqueueFrame(TcpFrame frame, bool mayBlock);
  // Queues a run of frames as one indivisible step: either every frame is in
  // outQueue_ or none is. Enqueuing them one at a time would let the sender
  // thread start transmitting a partially built group, which for a multi-chunk
  // put means the peer applies a prefix of a transfer this side may still fail
  // to finish staging.
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
  // Starts the D2H copy for a ReadReply and hands the frame to the staging
  // queue, so the reader thread can go back to draining the socket instead of
  // waiting on the device. Consumes the lease. Returns an error only if the
  // copy could not be started, in which case nothing was queued.
  Status stageReadReply(
      std::vector<uint8_t> frame,
      TcpSegmentRegistry::Lease lease,
      const void* src,
      size_t len,
      int deviceId,
      uint64_t reqId);
  // Retires staged replies whose copy has finished, oldest first. Runs on the
  // EventBase, never on the reader thread: polling from the reader would put
  // the head-of-line block straight back.
  void pollPendingReadReplies();
  // Kicks the poll loop if it is not already running. Safe from any thread.
  void schedulePendingReplyPoll();
  // Waits for outstanding staging copies and drops the frames. Called during
  // teardown, because the GPU may still be writing into a pending frame's
  // payload and that memory must not be freed underneath it.
  /// Waits for a staged D2H copy on `deviceId` whose completion is otherwise
  /// unknown, so the frame it targets can be released. Used on the paths where
  /// a copy was issued but the event never became a usable completion signal.
  void waitForStagedCopy(int deviceId) noexcept;

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
  // Closes the data connection, at most once for the lifetime of the transport.
  // Both the reader thread and shutdown() refuse the connection, so this has
  // two concurrent callers.
  void closeDataConnOnce();

  [[maybe_unused]] int deviceId_{-1};
  EventBase* evb_{nullptr};
  std::shared_ptr<TcpSegmentRegistry> registry_;
  std::shared_ptr<CudaApi> cudaApi_;
  controller::TcpSocketConfig config_;
  std::string name_{"tcp"};
  // Atomic: written by bind()/connect()/shutdown() and read by the
  // put/get/send/recv guards and state(), which callers may invoke from any
  // thread (e.g. a shutdown racing an in-flight transfer). A plain member here
  // is a data race, matching why running_/connBroken_ below are atomic too.
  std::atomic<TransportState> state_{TransportState::Disconnected};

  std::string host_{"127.0.0.1"};
  uint16_t port_{0};

  std::unique_ptr<controller::AsyncTcpServer> server_;
  std::unique_ptr<controller::Conn> dataConn_;

  // Serialises bind()/connect()/shutdown(), which together own server_,
  // dataConn_, reader_ and sender_. Without it a shutdown() concurrent with an
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
  // hold across the reader_/sender_ joins in shutdown(): neither loop, nor
  // failAllPending(), ever acquires it.
  // Bounds on the outbound queue and the in-flight map, for work this side
  // originates: outQueue_ holds whole payloads, and inflight_ one entry per
  // outstanding chunk, so an application issuing put/get faster than the link
  // drains would otherwise grow both without limit. Caller threads wait for
  // room, which is real backpressure -- the application slows down and nothing
  // breaks.
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
  // Bytes currently queued in outQueue_. Guarded by outMu_.
  size_t outBytes_{0};

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

  std::thread reader_;
  std::thread sender_;
  std::atomic<bool> running_{false};
  std::atomic<bool> connBroken_{false};
  // Gates closeDataConnOnce(). Conn::close() is a check-then-act on a bare fd,
  // so two callers seeing it open both close it; the second reaps a descriptor
  // some unrelated thread has since been handed.
  std::atomic<bool> dataConnClosed_{false};

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

  // Outbound frame queue drained by the sender thread. Decoupling all sends
  // from the reader thread is what prevents a mutual-READ deadlock (two peers'
  // readers both blocked mid-send while neither drains its recv).
  std::mutex outMu_;
  std::condition_variable outCv_;
  std::deque<TcpOutItem> outQueue_;
  bool outClosed_{false};

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
      controller::TcpSocketConfig config = {},
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
  controller::TcpSocketConfig config_;
  std::string host_{"127.0.0.1"};
  std::shared_ptr<CudaApi> cudaApi_;
  std::shared_ptr<TcpSegmentRegistry> registry_{
      std::make_shared<TcpSegmentRegistry>()};
  std::atomic<uint64_t> nextSegId_{1};
};

} // namespace uniflow
