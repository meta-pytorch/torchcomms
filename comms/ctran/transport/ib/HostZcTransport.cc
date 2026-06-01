// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/transport/ib/HostZcTransport.h"

#include "comms/ctran/utils/Exception.h"

namespace ctran::transport::ib {

HostZcTransport::HostZcTransport(
    int peerRank,
    CtranIb* ctranIb,
    int myRank,
    int cudaDev,
    const CommLogData* logMetaData)
    : peerRank_(peerRank),
      myRank_(myRank),
      cudaDev_(cudaDev),
      ctranIb_(ctranIb),
      logMetaData_(logMetaData) {
  if (ctranIb_ == nullptr) {
    throw ctran::utils::Exception(
        "HostZcTransport: ctranIb must not be null", commInternalError);
  }
  // Eagerly drive the per-peer VC rendezvous so every subsequent
  // hot-path call (iSendCtrlMsg, iRecvCtrlMsg, iSendChunk, iRecvChunk,
  // progress()/pollRecvNotifications) can rely on `vcs_` being
  // non-empty without paying a per-call "have we connected yet?"
  // branch.
  //
  // Caller-blocking note: ctranIb_->connectVcs(peer) is two-sided
  // (CtranIb.h:514-534). The smaller-rank side initiates and returns
  // immediately; the larger-rank side spins until the peer also calls
  // connectVcs back. Algorithms calling
  // CtranMapper::getP2pTransport(peer, kZeroCopy) must be aware that
  // the construction (on first cache miss) may block on the
  // larger-rank side. Deadlock-freedom still holds — for any pair the
  // smaller rank's initiator is non-blocking, so single-pair ordering
  // never deadlocks.
  impl::connectAndPopulateVcs(ctranIb_, peerRank_, myRank_, vcs_);
  // Size the per-VC recv counters for the lifetime of the transport
  // (replaces the lazy ensureZcRecvVecs() helper).
  vcRecvIssued_.assign(numVcs(), 0);
  vcRecvCompleted_.assign(numVcs(), 0);
}

} // namespace ctran::transport::ib
