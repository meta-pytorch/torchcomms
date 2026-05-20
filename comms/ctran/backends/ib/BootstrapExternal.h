// Copyright (c) Meta Platforms, Inc. and affiliates.

// Owner-managed external bootstrap surface for CtranIb. The whole external
// path (BootstrapMode::kExternal + BootstrapExternal class + RdmaTransport
// caller) is scheduled for deletion once RdmaTransport moves off it.
#ifndef CTRAN_IB_BOOTSTRAP_EXTERNAL_H_
#define CTRAN_IB_BOOTSTRAP_EXTERNAL_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/utils/commSpecs.h"

class CtranComm;
class CtranIbVirtualConn;

namespace ctran::ib {

class VcState;

// BootstrapExternal is the callsite-managed handshake driver used when
// CtranIb is constructed with BootstrapMode::kExternal. The caller (today
// only RdmaTransport) ships the local bus card returned by getLocalVcId()
// to the peer over its own channel, then drives the two-step handshake by
// calling connectVc() with the remote bus card.
//
// Lifetime: held by std::unique_ptr<BootstrapExternal> externalBootstrap_
// on CtranIb. Allocated only when bootstrapMode == kExternal. Must be
// destroyed before VcState (connectVc() publishes into vcState_).
class BootstrapExternal {
 public:
  BootstrapExternal(
      VcState& vcState,
      std::vector<CtranIbDevice>& devices,
      CtranComm* comm,
      int cudaDev,
      uint64_t commHash,
      const std::string& commDesc,
      const CommLogData& logData,
      uint32_t trafficClass);
  ~BootstrapExternal() = default;

  BootstrapExternal(const BootstrapExternal&) = delete;
  BootstrapExternal& operator=(const BootstrapExternal&) = delete;
  BootstrapExternal(BootstrapExternal&&) = delete;
  BootstrapExternal& operator=(BootstrapExternal&&) = delete;

  // Create a fresh VC for the given peer, stash it in pendingVcs_, and
  // return the serialized local bus card for the caller to ship to the
  // peer. Must be paired with a subsequent connectVc() for the same
  // peerRank.
  std::string getLocalVcId(int peerRank);

  // Complete the handshake for a peer previously prepared with
  // getLocalVcId(): pop the pending VC, run setupVc() with the remote
  // bus card, and publish it into vcState_.
  commResult_t connectVc(const std::string& remoteVcIdentifier, int peerRank);

 private:
  VcState& vcState_;
  std::vector<CtranIbDevice>& devices_;
  CtranComm* comm_{nullptr};
  int cudaDev_{-1};
  uint64_t commHash_{0};
  std::string commDesc_;
  const CommLogData& logData_;
  uint32_t trafficClass_{0};

  // VCs created by getLocalVcId() awaiting a matching connectVc() call.
  std::unordered_map<int, std::shared_ptr<CtranIbVirtualConn>> pendingVcs_;
  std::mutex mutex_;
};

} // namespace ctran::ib

#endif // CTRAN_IB_BOOTSTRAP_EXTERNAL_H_
