// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/transport/ib/ChunkHooks.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/commSpecs.h"

namespace ctran::transport {

// Sentinel staging-slot index used by zero-copy chunks (no staging slot
// is consumed). CB chunks must pass a real index in [0, pipelineDepth_).
constexpr int kNoStagingSlot = -1;

// Backend-agnostic mode used by callers to ask CtranMapper for the
// matching per-peer transport. Each peer is bound to exactly one mode
// for the lifetime of its cached transport.
enum class HostTransportMode {
  kZeroCopy,
  kCopyBased,
};

// Per-peer remote endpoint info, produced on the receiver side by
// calling ctran::transport::ib::impl::importRemoteInfo() on the
// ControlMsg the caller received via iRecvCtrlMsg. Currently IB-specific
// (embeds CtranIbRemoteAccessKey); future backends will need a
// tagged-union or per-backend variant.
struct RemotePeerInfo {
  bool isZeroCopy{false};
  void* memHdl{nullptr};
  CtranIbRemoteAccessKey remoteKey{};
};

// Direction of a ChunkRequest.
enum class ChunkKind : uint8_t {
  kInvalid = 0,
  kSend,
  kRecv,
};

// Inspectable handle returned (via output pointer in *ChunkArgs) by every
// iSendChunk / iRecvChunk call. Caller-owned: must outlive the iput's
// CQE / notify.
//
//   ZC send : completion observed via `ibReq.isComplete()` (transport
//             gives the caller's `ibReq` directly to the IB backend).
//   ZC recv : completion observed via per-VC counter compare against
//             `mySeq`.
//   CB send / CB recv : completion observed via per-slot counter compare
//                       against `mySeq` (the per-slot state machine bumps
//                       the counter on each chunk's final transition).
struct ChunkRequest {
  ChunkKind kind{ChunkKind::kInvalid};
  int16_t vcIdx{-1};
  int16_t stagingSlot{kNoStagingSlot};
  uint64_t mySeq{0};
  // Used only by ZC send: CtranIb signals iput completion through it.
  // Default-constructed for CB and ZC recv.
  CtranIbRequest ibReq;
};

// Forward declaration for friend list of CtrlRequest. The two
// concrete IB host transports need access to CtrlRequest's private
// storage to drive the IB ctrl-msg exchange without exposing that
// machinery in the public interface.
namespace ib {
class HostZcTransport;
class HostCbTransport;
} // namespace ib

// Caller-owned ctrl-exchange poll handle. Pure inflight-request tracker
// for one iSendCtrlMsg / iRecvCtrlMsg. The caller-owned payload buffer
// passed to iSendCtrlMsg / iRecvCtrlMsg is what holds the wire bytes — the
// transport just retains a pointer to it for the duration of the IB
// exchange. CtrlRequest itself only carries the underlying IB request
// state needed to poll for completion.
class CtrlRequest {
 public:
  CtrlRequest() = default;
  bool isComplete() const {
    return complete_;
  }

 private:
  bool complete_{false};
  CtranIbRequest ctrlReq_;

  friend class ctran::transport::ib::HostZcTransport;
  friend class ctran::transport::ib::HostCbTransport;
};

// Send-chunk args. Mode-agnostic — fields irrelevant to the active
// transport are silently ignored (debug-build asserts catch
// mode/field misuse).
//
// Field groups:
//   - Common         : userBuf, offset, len, vcIdx, req
//   - CB-specific    : stagingSlot, round, hooks
//                      (ZC asserts stagingSlot == kNoStagingSlot,
//                       round == 0, hooks empty.)
//   - ZC-send only   : localMrHdl, remoteMrHdl, remoteKey
//                      (CB asserts these are nullptr.)
struct SendChunkArgs {
  // Common
  const void* userBuf{nullptr};
  size_t offset{0};
  size_t len{0};
  int vcIdx{0};
  // Optional output. If non-null, transport writes a polling handle
  // here. Caller owns *req and must keep it alive until
  // testChunkDone(*req) returns true.
  ChunkRequest* req{nullptr};

  // CB-specific
  int stagingSlot{kNoStagingSlot};
  int round{0};
  SendChunkHooks hooks{};

  // ZC-specific
  void* localMrHdl{nullptr};
  void* remoteMrHdl{nullptr};
  const CtranIbRemoteAccessKey* remoteKey{nullptr};
};

// Recv-chunk args. Same shape as SendChunkArgs minus the ZC-specific
// remote-info group (recv-side ZC has nothing per-chunk to address —
// data lands in the buffer the receiver previously exported via the
// iSendCtrlMsg / impl::exportRecvBuf handshake).
struct RecvChunkArgs {
  void* userBuf{nullptr};
  size_t offset{0};
  size_t len{0};
  int vcIdx{0};
  ChunkRequest* req{nullptr};

  // CB-specific
  int stagingSlot{kNoStagingSlot};
  int round{0};
  RecvChunkHooks hooks{};
};

// Abstract per-peer host-side P2P transport interface.
//
// Concrete implementations split on transport mode:
//   - ctran::transport::ib::HostZcTransport — pure zero-copy IB transport.
//   - ctran::transport::ib::HostCbTransport — pure copy-based IB transport
//     with per-slot staging buffer + GpeKernelSync flow control.
//
// Each peer is bound to exactly one mode for the lifetime of its
// cached transport (see CtranMapper::getP2pTransport(peer, mode)).
class IP2pHostTransport {
 public:
  virtual ~IP2pHostTransport() = default;

  // === Accessors ===
  virtual int peerRank() const = 0;
  virtual HostTransportMode mode() const = 0;
  // ZC: returns 0 (no staging). CB: pipelineDepth_ from ctor.
  virtual int pipelineDepth() const = 0;
  // ZC: returns 0 (no chunking; one chunk covers the full slice).
  // CB: chunkSize_ from ctor.
  virtual size_t chunkSize() const = 0;

  // === Resource lifecycle ===
  virtual commResult_t progress() = 0;

  // === Per-operation ctrl exchange ===
  //
  // Pure ctrl-msg primitives — no mem-export/import is bundled in.
  // The caller owns the payload buffer and (for ZC) is responsible
  // for filling it via ctran::transport::ib::impl::exportRecvBuf (or
  // for CB) zeroing it as a SYNC.
  //
  // iSendCtrlMsg: posts `len` bytes from `payload` on vcs[0] with wire
  // header `type`. The caller-owned `payload` must outlive the
  // request (the IB backend retains a pointer to it until completion).
  virtual commResult_t iSendCtrlMsg(
      ControlMsgType type,
      const void* payload,
      size_t len,
      CtrlRequest* out) = 0;

  // iRecvCtrlMsg: posts an irecv into the caller-owned `payload` buffer
  // (capacity `len`). After completion, the bytes are available in
  // that buffer. For the ZC sender path, the caller then typically
  // calls ctran::transport::ib::impl::importRemoteInfo on the
  // received ControlMsg to populate a RemotePeerInfo. The
  // caller-owned `payload` must outlive the request.
  virtual commResult_t
  iRecvCtrlMsg(void* payload, size_t len, CtrlRequest* out) = 0;

  // === Chunk addressing ===
  //
  // ZC: returns single-chunk values (1, 0, totalSize).
  // CB: returns the equivalent of ceil(totalSize/chunkSize_) etc.
  virtual int computeTotalChunks(size_t totalSize) const = 0;
  virtual size_t computeChunkOffset(int chunkIdx, size_t totalSize) const = 0;
  virtual size_t computeChunkLen(int chunkIdx, size_t totalSize) const = 0;

  // === Issue-readiness ===
  //
  // Validation arguments (vcIdx, stagingSlot) are aborted on
  // invalidity — caller bugs are programming errors, not
  // return-false-able states. May trigger lazy VC rendezvous on
  // first call (one-time per peer; subsequent calls are O(1) reads).
  // Returns true iff the slot is in a state that can accept a new
  // chunk:
  //   ZC : always true after validation succeeds.
  //   CB : the staging slot is IDLE.
  virtual bool isReadyForSend(int vcIdx, int stagingSlot = kNoStagingSlot) = 0;
  virtual bool isReadyForRecv(int vcIdx, int stagingSlot = kNoStagingSlot) = 0;

  // === Chunk-level send/recv ===
  //
  // Precondition: isReadyForSend/Recv(vcIdx, stagingSlot) was true
  // immediately before this call.
  //
  // Mode-agnostic: callers fill the SendChunkArgs/RecvChunkArgs struct
  // with whatever fields apply to their mode. The active transport
  // uses what its mode needs and asserts that the wrong-mode fields
  // are at their defaults.
  virtual commResult_t iSendChunk(const SendChunkArgs& args) = 0;
  virtual commResult_t iRecvChunk(const RecvChunkArgs& args) = 0;

  // Pumps progress() once, then polls completion. Idempotent. Writes
  // the observed completion (true if the chunk is fully complete) into
  // *done. *done is left untouched if the call returns an error.
  virtual commResult_t testChunkDone(const ChunkRequest& req, bool* done) = 0;

  // === Ctrl-request poll/wait ===
  // `testCtrlMsgDone` pumps progress() once and writes the observed
  // completion into *done. `waitCtrlMsgDone` blocks until the ctrl
  // request completes.
  virtual commResult_t testCtrlMsgDone(CtrlRequest& req, bool* done) = 0;
  virtual commResult_t waitCtrlMsgDone(CtrlRequest& req) = 0;

  // === Per-transport caller-must-lock contract ===
  //
  // Every hot-path method on a concrete IP2pHostTransport asserts that
  // the calling thread first acquired this transport's mutex via
  // lock() (or its RAII guard P2pTransportLockGuard). The check is
  // always-on (unconditional FB_CHECKABORT) — see free function
  // checkLocked() below. Trivial accessors (peerRank, mode,
  // pipelineDepth, chunkSize, the compute* overrides) are exempt.
  //
  // Concrete implementations back this with a per-instance std::mutex
  // plus a thread_local flag.
  virtual void lock() = 0;
  virtual void unlock() = 0;
};

// Asserts the calling thread holds `t`'s per-transport lock (acquired
// via t->lock() or a P2pTransportLockGuard). Aborts via FB_CHECKABORT
// when not locked. Unlike checkEpochLock(), this check is always
// enabled — there is no cvar gate.
//
// Implementation lives next door in the `impl::` namespace; defined
// inline here so every callsite sees the body and we don't need a
// separate .cc file.
namespace impl {

// Per-thread map from transport instance pointer to "this thread
// currently holds the per-transport lock" flag.
//
// `inline thread_local` (C++17) gives one instance per thread,
// shared across all TUs that include this header.
inline thread_local std::unordered_map<void*, std::atomic_bool>
    p2pTransportLockedFlags;

// Helpers used by concrete IP2pHostTransport implementations to flip
// the thread_local flag from inside their lock()/unlock() bodies.
inline bool p2pTransportLockFlagIsSet(void* t) {
  return p2pTransportLockedFlags[t].load();
}
inline void p2pTransportLockFlagSet(void* t, bool value) {
  p2pTransportLockedFlags[t].store(value);
}

} // namespace impl

inline void checkLocked(IP2pHostTransport* t) {
  FB_CHECKABORT(
      t != nullptr,
      "CTRAN-TRANSPORT: checkLocked called with null IP2pHostTransport*");
  FB_CHECKABORT(
      impl::p2pTransportLockedFlags[t].load(),
      "CTRAN-TRANSPORT: per-transport lock not held by the calling thread. "
      "Caller must wrap the access in a P2pTransportLockGuard (or call "
      "t->lock() / t->unlock() explicitly). This is a caller bug.");
}

// RAII guard for a per-transport lock.
class P2pTransportLockGuard {
 public:
  explicit P2pTransportLockGuard(IP2pHostTransport* t) : t_(t) {
    if (t_ != nullptr) {
      t_->lock();
    }
  }
  ~P2pTransportLockGuard() {
    if (t_ != nullptr) {
      t_->unlock();
    }
  }

  P2pTransportLockGuard(const P2pTransportLockGuard&) = delete;
  P2pTransportLockGuard& operator=(const P2pTransportLockGuard&) = delete;
  P2pTransportLockGuard(P2pTransportLockGuard&&) = delete;
  P2pTransportLockGuard& operator=(P2pTransportLockGuard&&) = delete;

 private:
  IP2pHostTransport* t_{nullptr};
};

} // namespace ctran::transport
