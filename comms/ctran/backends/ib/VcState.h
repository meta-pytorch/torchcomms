// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_IB_VC_STATE_H_
#define CTRAN_IB_VC_STATE_H_

#include <memory>
#include <string>
#include <vector>

#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>

#include "comms/ctran/backends/ib/CtranIbVc.h"
#include "comms/ctran/utils/CtranPerf.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/LogUtils.h"

namespace ctran::ib {

// Virtual connection status for all peers.
// NOTE: when using a VC, additional per-VC lock is required.
struct VcStateMaps {
  folly::F14FastMap<QpUniqueId, std::shared_ptr<CtranIbVirtualConn>> qpToVcMap;
  folly::F14FastMap<int, std::shared_ptr<CtranIbVirtualConn>> rankToVcMap;
};

// VcState owns all per-peer virtual-connection state for a single CtranIb
// instance. Both the internal bootstrap path (ctran::ib::Bootstrap) and the
// external bootstrap path (BootstrapExternal.cc) publish established VCs
// through setupAndPublishVc(); critical-path callers look them up through
// getVc/getVcByQp. The class is intentionally not thread-safe at the
// construction / init / releaseAll boundary; all mutating operations after
// init() are themselves thread-safe (vcStateMaps_ is folly::Synchronized).
class VcState {
 public:
  VcState() = default;
  ~VcState() = default;

  VcState(const VcState&) = delete;
  VcState& operator=(const VcState&) = delete;
  VcState(VcState&&) = delete;
  VcState& operator=(VcState&&) = delete;

  // Late init. Called by CtranIb::init() after ncclLogData is populated.
  //   - owner: opaque pointer used as "pimpl" in log messages.
  //   - logData: reference to the owner's CommLogData; must outlive *this.
  //   - nRanks: number of ranks. Pass 0 to skip connectedPeerMap setup
  //             (used when there is no associated CtranComm).
  void init(void* owner, const CommLogData& logData, int nRanks);

  // Release every established VC, clear connectedPeerMap, and reset the
  // lock-free fast-path pointer. Callers must invoke this before tearing
  // down resources referenced by the VCs (e.g., the shared CQ).
  void releaseAll();

  // Setup a freshly-constructed VC with the remote bus card, publish it into
  // vcStateMaps_, and log the established connection. Shared between the
  // internal bootstrap path (ctran::ib::Bootstrap) and the external
  // bootstrap path (BootstrapExternal.cc). The VC must not already be
  // present.
  commResult_t setupAndPublishVc(
      std::shared_ptr<CtranIbVirtualConn> vc,
      const std::string& remoteVcIdentifier,
      int peerRank);

  // Update vcStateMaps_ to include the established VC.
  // Thread-safe; callable concurrently by server and client threads.
  commResult_t updateVcState(
      std::shared_ptr<CtranIbVirtualConn> vc,
      int peerRank);

  // Returns true if the peer has been pre-connected (preConnect path).
  inline bool isPeerConnected(int peerRank) const {
    return (
        connectedPeerMap_.size() > peerRank && connectedPeerMap_.at(peerRank));
  }

  // Mark a peer as connected. Used by preConnect() once it observes that
  // the connection has been established on the listen thread.
  inline void markPeerConnected(int peerRank) {
    connectedPeerMap_.at(peerRank) = true;
  }

  // Whether connectedPeerMap_ has been sized (non-empty means nRanks is
  // known). Used by preConnect() to early-out when no comm is associated.
  inline bool hasConnectedPeerMap() const {
    return !connectedPeerMap_.empty();
  }

  // Get a virtual connection for a given peer.
  // If the peer is not yet connected, return nullptr.
  // For a returned VC, it is guaranteed to be ready to use.
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline std::shared_ptr<CtranIbVirtualConn> getVc(int peerRank) {
    if (PerfConfig::skipVcConnectionCheck ||
        (vcStateMapsPtr_ && isPeerConnected(peerRank))) {
      return vcStateMapsPtr_->rankToVcMap.at(peerRank);
    } else {
      auto locked = vcStateMaps_.rlock();

      auto it = locked->rankToVcMap.find(peerRank);
      if (it == locked->rankToVcMap.end()) {
        return nullptr;
      }

      return it->second;
    }
  }

  // Get the virtual connection associated with the given QP.
  // Returns nullptr (and logs) when the QP is unknown.
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline std::shared_ptr<CtranIbVirtualConn> getVcByQp(QpUniqueId qpUniqueId) {
    auto getVcFrom = [&](const VcStateMaps* maybeLockedVcStateMapsPtr)
        -> std::shared_ptr<CtranIbVirtualConn> {
      auto it = maybeLockedVcStateMapsPtr->qpToVcMap.find(qpUniqueId);
      // VC should be already created and added to vcStateMaps_. If not
      // found, it likely indicates a COMM bug, e.g., not using CtranIb
      // epoch lock.
      if (it == maybeLockedVcStateMapsPtr->qpToVcMap.end()) {
        CLOGF(
            ERR,
            "CTRAN-IB: Received unknown QP number {} on IB device {} in pimpl {} commHash {:x}, commDesc {}. Known QPs: {}. It likely indicates a COMM bug.",
            qpUniqueId.first,
            qpUniqueId.second,
            owner_,
            logData_ ? logData_->commHash : 0,
            logData_ ? logData_->commDesc : std::string{},
            f14FastMapToStr(maybeLockedVcStateMapsPtr->qpToVcMap));
        return nullptr;
      }
      return it->second;
    };

    auto vcstatemaps = PerfConfig::skipVcConnectionCheck
        ? vcStateMapsPtr_
        : &(*vcStateMaps_.rlock());
    return getVcFrom(vcstatemaps);
  }

 private:
  commResult_t checkAndInsertQpToVcMap(
      folly::F14FastMap<QpUniqueId, std::shared_ptr<CtranIbVirtualConn>>& map,
      QpUniqueId& qpId,
      std::shared_ptr<CtranIbVirtualConn>& vc);

  // Opaque "pimpl" pointer (the owning CtranIb*), used only for log output
  // so we can match the existing error messages.
  void* owner_{nullptr};
  // Non-owning pointer to the owner's CommLogData. Set in init().
  const CommLogData* logData_{nullptr};

  folly::Synchronized<VcStateMaps> vcStateMaps_;
  // Lock-free struct to be used in eager connect mode which guarantees
  // single-threaded access.
  VcStateMaps* vcStateMapsPtr_{nullptr};

  // bitmap to indicate whether a peer is connected.
  // note that only one thread accesses it (e.g., GPE thread) or the epoch
  // lock needs to be acquired.
  std::vector<bool> connectedPeerMap_;
};

} // namespace ctran::ib

#endif // CTRAN_IB_VC_STATE_H_
