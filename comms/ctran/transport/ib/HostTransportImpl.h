// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/backends/ib/CtranIbImpl.h"
#include "comms/ctran/backends/ib/CtranIbVc.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/transport/IP2pHostTransport.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/commSpecs.h"

struct CommLogData;

namespace ctran::transport::ib::impl {

// Index of the per-peer VC used for ctrl-msg exchange. Both ctrl-plane
// primitives (iSendCtrlMsg / iRecvCtrlMsg) and the CB resource exchange post
// on this fixed VC. The data plane (iSendChunk / iRecvChunk) picks
// vcIdx ∈ [0, numVcs) per chunk independently.
constexpr int kCtrlMsgVc = 0;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers shared by the IB host transport. All live in
// ctran::transport::ib::impl and are defined `inline` in this header so
// every callsite sees the body.
//
// The ctrl-plane helpers take primitive references (CtranIbRequest&,
// bool&) rather than CtrlRequest&. The concrete transports (which are
// friends of CtrlRequest) pluck the fields out at the callsite and
// pass them in, so these helpers never have to know about
// CtrlRequest's shape.
// ─────────────────────────────────────────────────────────────────────────────

// Cast an opaque RegElem* (ctran::regcache::RegElem*) to the IB-backend
// ibRegElem pointer. Single place where the layered registration
// abstraction is unwrapped.
inline void* toIbRegElem(void* memHdl) {
  auto* regElem = static_cast<ctran::regcache::RegElem*>(memHdl);
  return regElem->ibRegElem;
}

// Build an IB_EXPORT_MEM ControlMsg describing the local recv buffer.
// ZC receivers call this and then hand &msg + sizeof(msg) to iSendCtrlMsg
// so the peer learns where to iput.
inline commResult_t
exportRecvBuf(void* localMrHdl, void* recvBuf, ControlMsg& outMsg) {
  return CtranIb::exportMem(recvBuf, toIbRegElem(localMrHdl), outMsg);
}

// Resolve a received ControlMsg into a RemotePeerInfo
// (EXPORT_MEM ⇒ ZC peer, fills mr+key; SYNC ⇒ CB peer, all-zeros).
// ZC senders call this on the caller-owned ControlMsg buffer that
// they passed to iRecvCtrlMsg, AFTER waitCtrlMsgDone returns.
inline commResult_t importRemoteInfo(
    const ControlMsg& msg,
    RemotePeerInfo* out) {
  if (out == nullptr) {
    CLOGF(ERR, "CTRAN-IB: importRemoteInfo: out is null");
    return commInvalidArgument;
  }
  if (msg.type == ControlMsgType::IB_EXPORT_MEM) {
    out->isZeroCopy = true;
    FB_COMMCHECK(CtranIb::importMem(&out->memHdl, &out->remoteKey, msg));
    return commSuccess;
  }
  if (msg.type == ControlMsgType::SYNC) {
    out->isZeroCopy = false;
    out->memHdl = nullptr;
    out->remoteKey = CtranIbRemoteAccessKey{};
    return commSuccess;
  }
  CLOGF(ERR, "CTRAN-IB: importRemoteInfo unhandled msg type {}", msg.type);
  return commInternalError;
}

// One-shot VC rendezvous + storage. Called exactly once per
// transport instance, from the concrete transport's constructor.
inline void connectAndPopulateVcs(
    CtranIb* ctranIb,
    int peerRank,
    int myRank,
    std::vector<std::shared_ptr<CtranIbVirtualConn>>& vcs) {
  FB_CHECKABORT(
      vcs.empty(),
      "CTRAN-IB: connectAndPopulateVcs called with already-populated vcs "
      "(peerRank=" +
          std::to_string(peerRank) +
          "). Must be invoked exactly once from the transport ctor.");
  vcs = ctranIb->connectVcs(peerRank);
  FB_CHECKABORT(
      !vcs.empty(),
      "CTRAN-IB: connectAndPopulateVcs got 0 VCs for peerRank=" +
          std::to_string(peerRank));
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: host transport VCs ready, peerRank={} myRank={} numVcs={}",
      peerRank,
      myRank,
      vcs.size());
}

// Defense-in-depth precondition for every hot-path access to vcs_.
inline void checkValidVc(
    const std::vector<std::shared_ptr<CtranIbVirtualConn>>& vcs,
    int vcIdx) {
  FB_CHECKABORT(
      !vcs.empty(),
      "CTRAN-IB: host transport hot-path called with empty vcs_ — "
      "connectAndPopulateVcs() should have run in the constructor.");
  FB_CHECKABORT(
      vcIdx >= 0 && vcIdx < static_cast<int>(vcs.size()),
      "CTRAN-IB: vcIdx=" + std::to_string(vcIdx) + " out of range [0, " +
          std::to_string(vcs.size()) + ")");
}

// Post `len` bytes from `payload` via vcs[kCtrlMsgVc]->isendCtrlMsg
// with wire header `type`. CtranIb retains a pointer to the
// caller-owned `payload` until the exchange completes; the caller
// must keep that buffer alive. The caller-owned `ctrlReq` is the
// inflight-IB-request slot; reposted on entry.
inline commResult_t iSendCtrlMsgImpl(
    std::vector<std::shared_ptr<CtranIbVirtualConn>>& vcs,
    ControlMsgType type,
    const void* payload,
    size_t len,
    CtranIbRequest& ctrlReq) {
  ctrlReq.repost(1);
  CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs[kCtrlMsgVc]->mutex, {
    FB_COMMCHECK(
        vcs[kCtrlMsgVc]->isendCtrlMsg(
            type, const_cast<void*>(payload), len, ctrlReq));
  });
  return commSuccess;
}

// Post a matching irecvCtrlMsg via vcs[kCtrlMsgVc] into the
// caller-owned `payload` buffer of capacity `len`. After completion,
// the caller reads the bytes from its own buffer. The caller-owned
// `ctrlReq` is the inflight-IB-request slot; reposted on entry.
inline commResult_t iRecvCtrlMsgImpl(
    std::vector<std::shared_ptr<CtranIbVirtualConn>>& vcs,
    void* payload,
    size_t len,
    CtranIbRequest& ctrlReq) {
  ctrlReq.repost(1);
  CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs[kCtrlMsgVc]->mutex, {
    FB_COMMCHECK(vcs[kCtrlMsgVc]->irecvCtrlMsg(payload, len, ctrlReq));
  });
  return commSuccess;
}

// Pump `progressFn` once, then poll ctrlReq.isComplete(). Writes the
// observed completion into `complete` (sticky once set) and into
// *done. Pure ctrl-plane primitive — no remote-info resolution
// happens here; the caller does that explicitly with importRemoteInfo
// if needed.
inline commResult_t testCtrlMsgDoneImpl(
    bool& complete,
    CtranIbRequest& ctrlReq,
    const std::function<commResult_t()>& progressFn,
    bool* done) {
  if (complete) {
    *done = true;
    return commSuccess;
  }
  FB_COMMCHECK(progressFn());
  complete = ctrlReq.isComplete();
  *done = complete;
  return commSuccess;
}

// Block-spin testCtrlMsgDoneImpl.
inline commResult_t waitCtrlMsgDoneImpl(
    bool& complete,
    CtranIbRequest& ctrlReq,
    const std::function<commResult_t()>& progressFn) {
  bool done = false;
  while (!done) {
    FB_COMMCHECK(testCtrlMsgDoneImpl(complete, ctrlReq, progressFn, &done));
  }
  return commSuccess;
}

// Wraps ctranIb->progress() or the NIC-affine ctranIb->progress(device)
// overload. deviceIdx populated ⇒ drain just that NIC's CQ.
inline commResult_t progressImpl(
    CtranIb* ctranIb,
    std::optional<int> deviceIdx = std::nullopt) {
  if (deviceIdx.has_value()) {
    return ctranIb->progress(*deviceIdx);
  }
  return ctranIb->progress();
}

} // namespace ctran::transport::ib::impl
