// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "comms/ctran/transport/IP2pHostTransport.h"
#include "comms/ctran/transport/ib/HostTransportImpl.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/commSpecs.h"

class CtranIb;
class CtranIbVirtualConn;
struct CommLogData;

namespace ctran::transport::ib {

// Pure zero-copy host-side IB transport.
//
// One chunk per iSendChunk/iRecvChunk call covers the full slice — there
// is no chunking and no staging buffer. Each iSendChunk posts a single
// iput on the chosen VC; per-VC FIFO completion order matches issue
// order, and the caller receives the IB request directly inside its
// caller-owned ChunkRequest. The transport keeps no per-VC send state.
//
// On the recv side, the only completion signal CtranIbVc gives is
// "consume one notify" (no read-only notify counter), so the transport
// keeps per-VC `(recvIssued, recvCompleted)` counters and drains in
// `progress()`.
//
// Shared plumbing (lazy connectVcs, ctrl-msg primitives, mem
// export/import) lives in HostTransportImpl.{h,cc} as
// ctran::transport::ib::impl::* free functions — this class delegates
// to them rather than inheriting.
class HostZcTransport : public IP2pHostTransport {
 public:
  HostZcTransport(
      int peerRank,
      CtranIb* ctranIb,
      int myRank,
      int cudaDev,
      const CommLogData* logMetaData);

  ~HostZcTransport() override = default;

  inline HostTransportMode mode() const override {
    return HostTransportMode::kZeroCopy;
  }

  inline int peerRank() const override {
    return peerRank_;
  }
  // Number of per-peer VCs.
  inline int numVcs() const {
    return static_cast<int>(vcs_.size());
  }
  inline bool vcsReady() const {
    return !vcs_.empty();
  }
  inline int myRank() const {
    return myRank_;
  }
  inline int cudaDev() const {
    return cudaDev_;
  }

  // Pure-ZC: no staging.
  inline int pipelineDepth() const override {
    return 0;
  }
  inline size_t chunkSize() const override {
    return 0;
  }

  // Single-chunk addressing.
  inline int computeTotalChunks(size_t /*totalSize*/) const override {
    return 1;
  }
  inline size_t computeChunkOffset(int /*chunkIdx*/, size_t /*totalSize*/)
      const override {
    return 0;
  }
  inline size_t computeChunkLen(int /*chunkIdx*/, size_t totalSize)
      const override {
    return totalSize;
  }

  // === Resource lifecycle ===
  inline commResult_t progress() override {
    ::ctran::transport::checkLocked(this);
    // progressImpl → ctranIb_->progress() internally invokes
    // checkEpochLock(ctranIb_).
    FB_COMMCHECK(impl::progressImpl(ctranIb_));
    FB_COMMCHECK(pollRecvNotifications());
    return commSuccess;
  }

  // === Per-operation ctrl exchange ===
  // Pure ctrl-msg primitives — the caller owns the payload buffer.
  // The receiver builds an IB_EXPORT_MEM ControlMsg via
  // ctran::transport::ib::impl::exportRecvBuf() and hands its address
  // to iSendCtrlMsg; the sender provides its own buffer to iRecvCtrlMsg,
  // waits, then resolves the bytes via
  // ctran::transport::ib::impl::importRemoteInfo().
  commResult_t iSendCtrlMsg(
      ControlMsgType type,
      const void* payload,
      size_t len,
      CtrlRequest* out) override;
  commResult_t iRecvCtrlMsg(void* payload, size_t len, CtrlRequest* out)
      override;

  // === Ctrl-request poll/wait ===
  commResult_t testCtrlMsgDone(CtrlRequest& req, bool* done) override;
  commResult_t waitCtrlMsgDone(CtrlRequest& req) override;

  // ZC is always issue-ready: vcIdx must be in range and stagingSlot
  // must be the sentinel.
  inline bool isReadyForSend(int vcIdx, int /*stagingSlot*/ = kNoStagingSlot)
      override {
    ::ctran::transport::checkLocked(this);
    impl::checkValidVc(vcs_, vcIdx);
    return true;
  }
  inline bool isReadyForRecv(int vcIdx, int /*stagingSlot*/ = kNoStagingSlot)
      override {
    ::ctran::transport::checkLocked(this);
    impl::checkValidVc(vcs_, vcIdx);
    return true;
  }

  // ZC iSendChunk: thin wrapper on vcs_[vcIdx]->iput.
  // - Asserts CB-only fields are at defaults
  //   (stagingSlot==kNoStagingSlot, round==0).
  // - Requires localMrHdl, remoteMrHdl, remoteKey to be non-null.
  // - If args.req != nullptr: writes ChunkRequest{kSend, vcIdx,
  //   kNoStagingSlot, mySeq=0, ibReq=...}; the caller polls
  //   `ibReq.isComplete()` (or testChunkDone) for completion.
  // - If args.req == nullptr: fire-and-forget. The transport posts
  //   the iput without an attached completion request.
  commResult_t iSendChunk(const SendChunkArgs& args) override;

  // ZC iRecvChunk: bumps vcRecvIssued_[vcIdx]. Even when args.req is
  // null, the issued counter is bumped — the QP delivers notifies in
  // FIFO order, so a later tracked recv's mySeq depends on earlier
  // notifies being drained by progress().
  commResult_t iRecvChunk(const RecvChunkArgs& args) override;

  commResult_t testChunkDone(const ChunkRequest& req, bool* done) override;

  // === Per-transport caller-lock (see IP2pHostTransport.h) ===
  //
  // Aborts on double-lock-from-same-thread and unlock-without-lock
  // (mirrors CtranIb::epochLock / epochUnlock at
  // fbcode/comms/ctran/backends/ib/CtranIb.cc:741-794).
  inline void lock() override {
    FB_CHECKABORT(
        !::ctran::transport::impl::p2pTransportLockFlagIsSet(this),
        "CTRAN-IB: HostZcTransport::lock() called twice on the same thread "
        "without an intervening unlock(). Likely a missing "
        "P2pTransportLockGuard scope.");
    transportMutex_.lock();
    ::ctran::transport::impl::p2pTransportLockFlagSet(this, true);
  }
  inline void unlock() override {
    FB_CHECKABORT(
        ::ctran::transport::impl::p2pTransportLockFlagIsSet(this),
        "CTRAN-IB: HostZcTransport::unlock() called without a matching "
        "lock() on this thread.");
    ::ctran::transport::impl::p2pTransportLockFlagSet(this, false);
    transportMutex_.unlock();
  }

 private:
  commResult_t pollRecvNotifications();

  int peerRank_;
  int myRank_;
  int cudaDev_;
  CtranIb* ctranIb_;

  std::vector<std::shared_ptr<CtranIbVirtualConn>> vcs_;

  // Per-VC recv counters. Sized in the ctor to numVcs() once
  // connectVcs() has populated vcs_.
  std::vector<uint64_t> vcRecvIssued_;
  std::vector<uint64_t> vcRecvCompleted_;

  // Per-transport caller-must-lock mutex. Algorithms must hold this
  // (typically via P2pTransportLockGuard) for the duration of every
  // hot-path call into the transport — see checkLocked() in
  // IP2pHostTransport.h. Read-only trivial accessors do NOT require
  // this lock.
  std::mutex transportMutex_;

  const CommLogData* logMetaData_{nullptr};
};

// ─────────────────────────────────────────────────────────────────────────────
// Inline definitions for the critical-path overrides.
// ─────────────────────────────────────────────────────────────────────────────

inline commResult_t HostZcTransport::iSendCtrlMsg(
    ControlMsgType type,
    const void* payload,
    size_t len,
    CtrlRequest* out) {
  FB_CHECKABORT(
      out != nullptr, "CTRAN-IB: iSendCtrlMsg requires non-null out pointer");
  ::ctran::transport::checkLocked(this);
  FB_COMMCHECK(checkEpochLock(ctranIb_));
  impl::checkValidVc(vcs_, impl::kCtrlMsgVc);
  return impl::iSendCtrlMsgImpl(vcs_, type, payload, len, out->ctrlReq_);
}

inline commResult_t
HostZcTransport::iRecvCtrlMsg(void* payload, size_t len, CtrlRequest* out) {
  FB_CHECKABORT(
      out != nullptr, "CTRAN-IB: iRecvCtrlMsg requires non-null out pointer");
  ::ctran::transport::checkLocked(this);
  FB_COMMCHECK(checkEpochLock(ctranIb_));
  impl::checkValidVc(vcs_, impl::kCtrlMsgVc);
  return impl::iRecvCtrlMsgImpl(vcs_, payload, len, out->ctrlReq_);
}

inline commResult_t HostZcTransport::testCtrlMsgDone(
    CtrlRequest& req,
    bool* done) {
  FB_CHECKABORT(
      done != nullptr, "CTRAN-IB: testCtrlMsgDone requires non-null done");
  ::ctran::transport::checkLocked(this);
  return impl::testCtrlMsgDoneImpl(
      req.complete_, req.ctrlReq_, [this]() { return progress(); }, done);
}

inline commResult_t HostZcTransport::waitCtrlMsgDone(CtrlRequest& req) {
  ::ctran::transport::checkLocked(this);
  return impl::waitCtrlMsgDoneImpl(
      req.complete_, req.ctrlReq_, [this]() { return progress(); });
}

inline commResult_t HostZcTransport::iSendChunk(const SendChunkArgs& args) {
  ::ctran::transport::checkLocked(this);
  FB_COMMCHECK(checkEpochLock(ctranIb_));
  impl::checkValidVc(vcs_, args.vcIdx);

  const void* src = static_cast<const char*>(args.userBuf) + args.offset;
  void* dst = static_cast<char*>(args.remoteMrHdl) + args.offset;
  void* regElem = impl::toIbRegElem(args.localMrHdl);

  CtranIbRequest* ibReqPtr = nullptr;
  if (args.req != nullptr) {
    *args.req = ChunkRequest{
        .kind = ChunkKind::kSend,
        .vcIdx = static_cast<int16_t>(args.vcIdx),
        .stagingSlot = kNoStagingSlot,
    };
    args.req->ibReq.repost(1);
    ibReqPtr = &args.req->ibReq;
  }

  CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs_[args.vcIdx]->mutex, {
    FB_COMMCHECK(
        vcs_[args.vcIdx]->iput(
            src,
            dst,
            args.len,
            regElem,
            *args.remoteKey,
            /*notify=*/true,
            /*config=*/nullptr,
            ibReqPtr,
            /*fast=*/false));
  });
  return commSuccess;
}

inline commResult_t HostZcTransport::iRecvChunk(const RecvChunkArgs& args) {
  ::ctran::transport::checkLocked(this);
  impl::checkValidVc(vcs_, args.vcIdx);

  // Always bump issued — even when args.req is null — because peer
  // notify CQEs are FIFO-delivered on the VC. A later tracked recv's
  // mySeq depends on earlier notifies being drained.
  const uint64_t mySeq = ++vcRecvIssued_[args.vcIdx];
  if (args.req != nullptr) {
    *args.req = ChunkRequest{
        .kind = ChunkKind::kRecv,
        .vcIdx = static_cast<int16_t>(args.vcIdx),
        .stagingSlot = kNoStagingSlot,
        .mySeq = mySeq,
    };
  }
  return commSuccess;
}

inline commResult_t HostZcTransport::testChunkDone(
    const ChunkRequest& req,
    bool* done) {
  FB_CHECKABORT(
      done != nullptr, "CTRAN-IB: testChunkDone requires non-null done");
  ::ctran::transport::checkLocked(this);
  impl::checkValidVc(vcs_, req.vcIdx);
  FB_COMMCHECK(progress());

  switch (req.kind) {
    case ChunkKind::kSend:
      // Source of truth: caller-owned IB request. Cast away const because
      // CtranIbRequest::isComplete is not const-qualified.
      *done = const_cast<CtranIbRequest&>(req.ibReq).isComplete();
      return commSuccess;
    case ChunkKind::kRecv:
      *done = vcRecvCompleted_[req.vcIdx] >= req.mySeq;
      return commSuccess;
    case ChunkKind::kInvalid:
      break;
  }
  FB_CHECKABORT(
      false, "CTRAN-IB: testChunkDone called with kInvalid ChunkRequest");
  return commInternalError;
}

inline commResult_t HostZcTransport::pollRecvNotifications() {
  // impl::progressImpl(ctranIb_) drains every notify CQE and increments
  // vcs_[i]->notifyCount_. Our job here is purely software: drain
  // pending notifies on each VC and bump vcRecvCompleted_[i].
  const int n = static_cast<int>(vcRecvCompleted_.size());
  for (int i = 0; i < n; ++i) {
    while (vcRecvCompleted_[i] < vcRecvIssued_[i]) {
      bool notified = false;
      CTRAN_IB_PER_OBJ_LOCK_GUARD(
          vcs_[i]->mutex, { FB_COMMCHECK(vcs_[i]->checkNotify(&notified)); });
      if (!notified) {
        break;
      }
      ++vcRecvCompleted_[i];
    }
  }
  return commSuccess;
}

} // namespace ctran::transport::ib
