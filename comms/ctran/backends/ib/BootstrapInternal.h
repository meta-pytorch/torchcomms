// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_IB_BOOTSTRAP_INTERNAL_H_
#define CTRAN_IB_BOOTSTRAP_INTERNAL_H_

#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <folly/Expected.h>
#include <folly/SocketAddress.h>

#include <sys/socket.h>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/backends/ib/VcLayout.h"
#include "comms/ctran/bootstrap/ISocketFactory.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/commSpecs.h"

class CtranComm;
class CtranIbVirtualConn;

namespace ctran::ib {

class VcState;

// Bootstrap is the internal handshake driver for one CtranIb instance. It
// owns the listen socket and accept thread, knows how to client-connect to
// a peer, and publishes every freshly-handshaken VC (or vector of VCs in
// multi-VC mode) into the supplied VcState. CtranIb only ever calls
// start() / connect() / shutdown() / getListenAddress() on this object;
// the rest of the bootstrap surface is private to BootstrapInternal.cc.
//
// Call hierarchy for a single peer connection (server or client side):
//   acceptLoop / connect
//     -> exchangeAndPublish
//          for vcIdx in [0, vcLayout_.maxVcsPerPeer):
//              build vc, swap bus cards (under vc->mutex)
//          vcState_.setupAndPublishVc(vcs, remoteBusCards, peerRank)
//          ack handshake
//
// Lifetime: held by std::unique_ptr<Bootstrap> bootstrap_ on CtranIb.
// Allocated only when bootstrapMode != kExternal. Must be destroyed before
// VcState (its accept thread calls vcState_.setupAndPublishVc).
class Bootstrap {
 public:
  Bootstrap(
      VcState& vcState,
      const VcLayout& vcLayout,
      std::shared_ptr<::ctran::bootstrap::ISocketFactory> socketFactory,
      std::shared_ptr<::comms::fault_tolerance::Abort> abortCtrl,
      const CommLogData& logData,
      CtranComm* comm,
      std::vector<CtranIbDevice>& devices,
      uint32_t trafficClass,
      int cudaDev,
      int rank,
      uint64_t commHash,
      const std::string& commDesc);
  ~Bootstrap();

  Bootstrap(const Bootstrap&) = delete;
  Bootstrap& operator=(const Bootstrap&) = delete;
  Bootstrap(Bootstrap&&) = delete;
  Bootstrap& operator=(Bootstrap&&) = delete;

  // Bind+listen on either the default interface (when qpServerAddr is
  // absent — equivalent to BootstrapMode::kDefaultServer) or on the
  // provided address (equivalent to BootstrapMode::kSpecifiedServer),
  // optionally allGather listen addrs across the communicator, and spawn
  // the accept thread.
  void start(std::optional<const SocketServerAddr*> qpServerAddr);

  // Slow path: open a client socket to the peer, exchange business cards
  // for all per-peer VCs, and hand the resulting VC vector to
  // vcState_.setupAndPublishVc(...).
  commResult_t connect(
      int peerRank,
      std::optional<const SocketServerAddr*> peerAddr);

  // Shut down the listen socket and join the accept thread. Idempotent;
  // safe to call from CtranIb's destructor or from ~Bootstrap.
  void shutdown();

  // Returns the bound listen address, or an error code if the socket has
  // not been bound yet.
  folly::Expected<folly::SocketAddress, int> getListenAddress() const;

 private:
  static void acceptLoop(Bootstrap* self);

  // Shared post-handshake path used by both the server accept side and the
  // client connect side. Loops over vcLayout_.maxVcsPerPeer VCs: build +
  // swap bus cards under vc->mutex; then hands the resulting vector +
  // remote bus cards to vcState_.setupAndPublishVc and exchanges the
  // final ack.
  commResult_t exchangeAndPublish(
      std::unique_ptr<::ctran::bootstrap::ISocket> sock,
      bool isServer,
      int peerRank);

  VcState& vcState_;
  const VcLayout& vcLayout_;
  std::shared_ptr<::ctran::bootstrap::ISocketFactory> socketFactory_;
  std::shared_ptr<::comms::fault_tolerance::Abort> abortCtrl_;
  const CommLogData& logData_;
  CtranComm* comm_{nullptr};
  std::vector<CtranIbDevice>& devices_;
  uint32_t trafficClass_{0};
  int cudaDev_{-1};
  int rank_{-1};
  uint64_t commHash_{0};
  std::string commDesc_;

  std::unique_ptr<::ctran::bootstrap::IServerSocket> listenSocket_;
  std::vector<sockaddr_storage> allListenSocketAddrs_;
  std::thread listenThread_;
  // Guards shutdown() so that listenSocket_->shutdown() and
  // listenThread_.join() each happen at most once even though shutdown()
  // is called both explicitly by CtranIb::~CtranIb() and again by
  // ~Bootstrap() as a safety net.
  bool shutdownCalled_{false};
};

} // namespace ctran::ib

#endif // CTRAN_IB_BOOTSTRAP_INTERNAL_H_
