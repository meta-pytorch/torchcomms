// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/backends/ib/BootstrapInternal.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <folly/ScopeGuard.h>
#include <folly/SocketAddress.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/ib/CtranIbVc.h"
#include "comms/ctran/backends/ib/VcState.h"
#include "comms/ctran/bootstrap/Socket.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/Debug.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/ScubaLogger.h"

namespace {
const std::string kCtranIbLogEventName{"CtranIb-QpExchange"};

const uint64_t kBootstrapMagic = 0xfaceb00cdeadbeef;
} // namespace

// TODO: We may want to retry if err is ECONNRESET,
// ETIMEDOUT, or ECONNRESET. For other errors, we
// may still want to throw an ctran::utils::Exception,
// like what would happen if FT is disabled (via
// the FB_SYSCHECKTHROW_EX macro).
#define HANDLE_SOCKET_ERROR(cmd, self)                                       \
  if (!self->abortCtrl_->Enabled()) {                                        \
    FB_SYSCHECKTHROW_EX(cmd, self->rank_, self->commHash_, self->commDesc_); \
  } else {                                                                   \
    int errCode = cmd;                                                       \
    if (errCode || self->abortCtrl_->Test()) {                               \
      CLOGF(ERR, "Socket error encountered: {}. Aborting.", errCode);        \
      self->abortCtrl_->Set(); /* Ensure remote is notified */               \
      break;                                                                 \
    }                                                                        \
  }

namespace ctran::ib {

Bootstrap::Bootstrap(
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
    const std::string& commDesc)
    : vcState_(vcState),
      vcLayout_(vcLayout),
      socketFactory_(std::move(socketFactory)),
      abortCtrl_(std::move(abortCtrl)),
      logData_(logData),
      comm_(comm),
      devices_(devices),
      trafficClass_(trafficClass),
      cudaDev_(cudaDev),
      rank_(rank),
      commHash_(commHash),
      commDesc_(commDesc) {
  listenSocket_ = socketFactory_->createServerSocket(
      static_cast<int>(NCCL_SOCKET_RETRY_CNT), abortCtrl_);
}

Bootstrap::~Bootstrap() {
  // Idempotent: explicit shutdown from CtranIb::~CtranIb runs first, but
  // we call shutdown() again here as a safety net.
  shutdown();
}

void Bootstrap::shutdown() {
  if (std::exchange(shutdownCalled_, true)) {
    return;
  }
  if (listenSocket_) {
    listenSocket_->shutdown();
  }
  if (listenThread_.joinable()) {
    listenThread_.join();
  }
}

folly::Expected<folly::SocketAddress, int> Bootstrap::getListenAddress() const {
  return listenSocket_->getListenAddress();
}

void Bootstrap::start(std::optional<const SocketServerAddr*> qpServerAddr) {
  // Setup the listen socket
  std::string resolvedIfName;
  folly::SocketAddress addrSockAddr;
  if (!qpServerAddr.has_value()) {
    // Use default NCCL socket ifname
    auto maybeAddr = ::ctran::bootstrap::getInterfaceAddress(
        NCCL_SOCKET_IFNAME, NCCL_SOCKET_IPADDR_PREFIX, true, &resolvedIfName);
    if (maybeAddr.hasError()) {
      CLOGF(WARN, "CTRAN-IB: No socket interfaces found");
      throw ::ctran::utils::Exception(
          "CTRAN-IB : No socket interfaces found",
          commSystemError,
          rank_,
          commHash_,
          commDesc_);
    }

    addrSockAddr = folly::SocketAddress(maybeAddr.value(), 0 /* port */);
  } else {
    auto qpServerAddrPtr = qpServerAddr.value();
    // use provided addr(i.e. ip, port, host) to initialize ctranIB
    addrSockAddr = toSocketAddress(*qpServerAddrPtr);
    resolvedIfName = qpServerAddrPtr->ifName;
  }

  FB_SYSCHECKTHROW_EX(
      listenSocket_->bindAndListen(addrSockAddr, resolvedIfName),
      rank_,
      commHash_,
      commDesc_);
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: Rank {} created listen socket with {} listenAddr {} ifname {}",
      rank_,
      qpServerAddr.has_value() ? "specified" : "self-finding",
      listenSocket_->getListenAddress()->describe().c_str(),
      resolvedIfName);

  // Exchange listen sock address among all ranks
  if (comm_) {
    allListenSocketAddrs_.resize(comm_->statex_->nRanks());
    auto maybeListenAddr = listenSocket_->getListenAddress();
    if (maybeListenAddr.hasError()) {
      FB_SYSCHECKTHROW_EX(maybeListenAddr.error(), rank_, commHash_, commDesc_);
    }
    maybeListenAddr->getAddress(&allListenSocketAddrs_[rank_]);

    auto resFuture = comm_->bootstrap_->allGather(
        allListenSocketAddrs_.data(),
        sizeof(allListenSocketAddrs_.at(0)),
        comm_->statex_->rank(),
        comm_->statex_->nRanks());
    FB_COMMCHECKTHROW_EX(
        static_cast<commResult_t>(std::move(resFuture).get()), logData_);
  }

  listenThread_ = std::thread{acceptLoop, this};
}

commResult_t Bootstrap::connect(
    int peerRank,
    std::optional<const SocketServerAddr*> peerAddr) {
  folly::SocketAddress peerSockAddr;
  const std::string* clientIfName;
  // When peer server address is passed, connect to it directly.
  // Otherwise, use the pre-exchanged listen socket address which requires an
  // associated communicator.
  if (peerAddr.has_value()) {
    auto peerAddrPtr = peerAddr.value();
    peerSockAddr = toSocketAddress(*peerAddrPtr);
    // always use the same ifname as remote server
    clientIfName = &peerAddrPtr->ifName;
  } else {
    FB_CHECKABORT(
        allListenSocketAddrs_.size() > 0,
        "Peer address is not specified, but pre-exchanged listen sockets is empty. It indicates a COMM internal bug.");
    peerSockAddr = toSocketAddress(allListenSocketAddrs_[peerRank]);
    clientIfName = &NCCL_CLIENT_SOCKET_IFNAME;
  }

  NcclScubaEvent scubaEvent(kCtranIbLogEventName, &logData_);
  scubaEvent.startAndRecord();
  SCOPE_EXIT {
    scubaEvent.stopAndRecord();
  };

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: Establishing connection: commHash {:x}, commDesc {}, rank {}, peer {}, peer listenAddr {} clientIfName {}",
      commHash_,
      commDesc_,
      rank_,
      peerRank,
      peerSockAddr.describe(),
      *clientIfName);

  // Send SETUP command to remote listenThread
  std::unique_ptr<::ctran::bootstrap::ISocket> sock =
      socketFactory_->createClientSocket(abortCtrl_);
  FB_SYSCHECKRETURN(
      sock->connect(
          peerSockAddr,
          *clientIfName,
          std::chrono::milliseconds(NCCL_SOCKET_RETRY_SLEEP_MSEC),
          NCCL_SOCKET_RETRY_CNT),
      commRemoteError);
  FB_SYSCHECKRETURN(
      sock->send(&kBootstrapMagic, sizeof(uint64_t)), commRemoteError);
  FB_SYSCHECKRETURN(sock->send(&rank_, sizeof(int)), commRemoteError);
  return exchangeAndPublish(std::move(sock), /*isServer=*/false, peerRank);
}

commResult_t Bootstrap::exchangeAndPublish(
    std::unique_ptr<::ctran::bootstrap::ISocket> sock,
    bool isServer,
    int peerRank) {
  if (peerRank < 0 || (comm_ && peerRank >= comm_->statex_->nRanks())) {
    CLOGF(
        ERR,
        "invalid peerRank ({}) < 0 or >= nRanks {}",
        peerRank,
        comm_ ? comm_->statex_->nRanks() : -1);
    return commInternalError;
  }

  // Whether to exchange a single VC (legacy) or multiple VCs is a static
  // property of this CtranIb instance, so the same loop handles both. The
  // two ends always agree on the count via cvar (vcLayout_.maxVcsPerPeer).
  const int numVcs = vcLayout_.maxVcsPerPeer;
  std::vector<std::shared_ptr<CtranIbVirtualConn>> vcs;
  std::vector<std::string> remoteBusCards;
  vcs.reserve(numVcs);
  remoteBusCards.reserve(numVcs);

  // Resolve the per-VC MAX_QPS slice for this peer (cvar/configList /
  // numVcs). Different peers may resolve to different values depending
  // on their connection class.
  int maxQpsPerVc =
      CtranIbVirtualConn::computeMaxQpsPerVc(comm_, peerRank, numVcs);

  for (int vcIdx = 0; vcIdx < numVcs; ++vcIdx) {
    // Create a new VC for the peer
    auto vc = std::make_shared<CtranIbVirtualConn>(
        devices_,
        peerRank,
        comm_,
        trafficClass_,
        cudaDev_,
        vcLayout_.vcToActiveDevices[vcIdx],
        maxQpsPerVc);

    std::string localBusCard, remoteBusCard;
    {
      // No need to lock since VC is not yet exposed to local rank. Lock to
      // simply follow VC thread-safety semantics.
      const std::lock_guard<std::mutex> lock(vc->mutex);

      /* exchange business cards */
      std::size_t size = vc->getBusCardSize();
      localBusCard.resize(size);
      remoteBusCard.resize(size);
      FB_COMMCHECK(vc->getLocalBusCard(localBusCard.data()));
      if (isServer) {
        FB_SYSCHECKRETURN(
            sock->recv(remoteBusCard.data(), size), commRemoteError);
        FB_SYSCHECKRETURN(
            sock->send(localBusCard.data(), size), commRemoteError);
      } else {
        FB_SYSCHECKRETURN(
            sock->send(localBusCard.data(), size), commRemoteError);
        FB_SYSCHECKRETURN(
            sock->recv(remoteBusCard.data(), size), commRemoteError);
      }
    }

    vcs.push_back(std::move(vc));
    remoteBusCards.push_back(std::move(remoteBusCard));
  }

  FB_COMMCHECK(
      vcState_.setupAndPublishVc(std::move(vcs), remoteBusCards, peerRank));

  // Ack that the connection is fully established.
  // Ensure remote rank don't use the VC before local setupVc and
  // vcStateMaps update.
  int ack{0};
  if (isServer) {
    FB_SYSCHECKRETURN(sock->send(&ack, sizeof(int)), commRemoteError);
    FB_SYSCHECKRETURN(sock->recv(&ack, sizeof(int)), commRemoteError);
  } else {
    FB_SYSCHECKRETURN(sock->recv(&ack, sizeof(int)), commRemoteError);
    FB_SYSCHECKRETURN(sock->send(&ack, sizeof(int)), commRemoteError);
  }
  return commSuccess;
}

void Bootstrap::acceptLoop(Bootstrap* self) {
  // Set cudaDev for logging
  FB_CUDACHECKTHROW_EX(
      cudaSetDevice(self->cudaDev_),
      self->rank_,
      self->commHash_,
      self->commDesc_);
  commNamedThreadStart(
      "CTranIbListen",
      self->rank_,
      self->commHash_,
      self->commDesc_.c_str(),
      __func__);
  while (1) {
    // Accept a connection from a peer. Socket will automatically closed when
    // it'll go out of scope (part of its destructor).
    auto maybeSocket = self->listenSocket_->acceptSocket();
    if (maybeSocket.hasError()) {
      if (self->listenSocket_->hasShutDown()) {
        break; // listen socket is closed or the CtranIb instance was aborted
      }
      HANDLE_SOCKET_ERROR(maybeSocket.error(), self);
    }

    std::unique_ptr<::ctran::bootstrap::ISocket> socket =
        std::move(maybeSocket.value());

    uint64_t magic{0};
    HANDLE_SOCKET_ERROR(socket->recv(&magic, sizeof(uint64_t)), self);
    if (magic != kBootstrapMagic) {
      CLOGF(
          WARN,
          "CTRAN-IB: Invalid magic - received {:x} but expected {:x} for commHash {:x} commDesc {}. "
          "Likely unexpected connection attempt. Ignoring. Local Addr: {},  Peer Addr: {}",
          magic,
          kBootstrapMagic,
          self->commHash_,
          self->commDesc_,
          socket->getLocalAddress().describe(),
          socket->getPeerAddress().describe());
      continue;
    }

    int peerRank;
    HANDLE_SOCKET_ERROR(socket->recv(&peerRank, sizeof(int)), self);
    const auto err = self->exchangeAndPublish(
        std::move(socket), /*isServer=*/true, peerRank);
    if (err != 0) { // TODO: We may want to handle certain errors differently?
      CLOGF(
          ERR,
          "CTRAN-IB: Failed to establish connection with peer rank {} for commHash {:x} commDesc {}, err={}",
          peerRank,
          self->commHash_,
          self->commDesc_,
          err);
      continue;
    }
  }

  CLOGF(
      INFO,
      "CTRAN-IB: Listen thread terminating, rank {} commHash {:x} commDesc {}",
      self->rank_,
      self->commHash_,
      self->commDesc_);
  return;
}

} // namespace ctran::ib
