// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <sys/socket.h>
#include <thread>

#include "comms/ctran/backends/tcpdevmem/CtranTcpDm.h"
#include "comms/ctran/backends/tcpdevmem/CtranTcpDmSingleton.h"
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/Debug.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/LogUtils.h"
#include "folly/SocketAddress.h"
#include "folly/synchronization/CallOnce.h"

#include <cerrno>

namespace ctran {

#define COMMCHECK_TCP(cmd)                                            \
  do {                                                                \
    ::comms::tcp_devmem::Status RES = cmd;                            \
    if (RES == ::comms::tcp_devmem::Status::InternalError) {          \
      return commInternalError;                                       \
    } else if (RES == ::comms::tcp_devmem::Status::RemoteError) {     \
      return commRemoteError;                                         \
    } else if (RES == ::comms::tcp_devmem::Status::InvalidArgument) { \
      return commInvalidArgument;                                     \
    }                                                                 \
  } while (0)

namespace {

bool isPeerDisconnectErrno(int err) {
  switch (err) {
    case ECONNABORTED:
    case ECONNREFUSED:
    case ECONNRESET:
    case ENOTCONN:
    case EPIPE:
    case ETIMEDOUT:
      return true;
    default:
      return false;
  }
}

commResult_t mapBootstrapSocketError(
    int err,
    const char* op,
    int rank,
    int peerRank,
    uint64_t commHash,
    const std::string& commDesc) {
  if (isPeerDisconnectErrno(err)) {
    CLOGF(
        WARN,
        "CTRAN-TCPDM: bootstrap {} failed with peer {} on rank {} commHash {:x} commDesc {} errno={} (treating as remote error)",
        op,
        peerRank,
        rank,
        commHash,
        commDesc,
        err);
    return commRemoteError;
  }

  CLOGF(
      ERR,
      "CTRAN-TCPDM: bootstrap {} failed with peer {} on rank {} commHash {:x} commDesc {} errno={}",
      op,
      peerRank,
      rank,
      commHash,
      commDesc,
      err);
  return commInternalError;
}

} // namespace

void CtranTcpDm::bootstrapPrepare(meta::comms::IBootstrap* bootstrap) {
  folly::SocketAddress ifAddrSockAddr;
  sockaddr_in6 sin6{};
  auto dev = netdev_->bootstrapIface();
  sin6.sin6_family = AF_INET6;
  sin6.sin6_addr = dev->addr;
  ifAddrSockAddr.setFromSockaddr(&sin6);
  FB_SYSCHECKTHROW_EX(
      listenSocket_.bindAndListen(ifAddrSockAddr, dev->name.c_str()),
      rank_,
      commHash_,
      commDesc_);

  std::string line =
      ::comms::tcp_devmem::addrToString(&dev->addr, 0, dev->name.c_str());
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-TCPDM: Rank {} created listen socket based on a self-finding address {} for cuda device {}",
      rank_,
      line.c_str(),
      cudaDev_);

  allListenSocketAddrs_.resize(nRanks_);
  auto maybeListenAddr = listenSocket_.getListenAddress();
  if (maybeListenAddr.hasError()) {
    FB_SYSCHECKTHROW_EX(maybeListenAddr.error(), rank_, commHash_, commDesc_);
  }
  maybeListenAddr->getAddress(&allListenSocketAddrs_[rank_]);

  auto resFuture = bootstrap->allGather(
      allListenSocketAddrs_.data(),
      sizeof(allListenSocketAddrs_.at(0)),
      rank_,
      nRanks_);
  FB_COMMCHECKTHROW_EX(
      static_cast<commResult_t>(std::move(resFuture).get()),
      rank_,
      commHash_,
      commDesc_);

  for (int i = 0; i < nRanks_; i++) {
    sockaddr_in6* sin =
        reinterpret_cast<sockaddr_in6*>(&allListenSocketAddrs_[i]);

    std::string line = ::comms::tcp_devmem::addrToString(
        &sin->sin6_addr, sin->sin6_port, nullptr);
    CLOGF_SUBSYS(
        INFO, INIT, "CTRAN-TCPDM: Rank {} bootstrap address {}", i, line);
  }

  listenThread_ = std::thread([this]() { bootstrapAccept(); });
}

void CtranTcpDm::bootstrapAddRecvPeer(
    int peerRank,
    ::comms::tcp_devmem::CommunicatorInterface* comm) {
  std::lock_guard lock(mutex_);
  recvComms_[peerRank] = comm;
}

void CtranTcpDm::bootstrapAccept() {
  // Set cudaDev for logging
  FB_CUDACHECKTHROW_EX(cudaSetDevice(cudaDev_), rank_, commHash_, commDesc_);
  commNamedThreadStart(
      "CTranTcpListen", rank_, commHash_, commDesc_.c_str(), __func__);

  while (1) {
    int peerRank;

    // Accept a connection from a peer. Socket will automatically closed when
    // it'll go out of scope (part of its destructor)
    auto maybeSocket = listenSocket_.accept();
    if (maybeSocket.hasError()) {
      if (maybeSocket.error() == EBADF || maybeSocket.error() == EINVAL) {
        break; // listen socket is closed
      }
      FB_SYSCHECKTHROW_EX(maybeSocket.error(), rank_, commHash_, commDesc_);
    }
    auto& socket = maybeSocket.value();
    FB_SYSCHECKTHROW_EX(
        socket.recv(&peerRank, sizeof(int)), rank_, commHash_, commDesc_);

    auto transport = CtranTcpDmSingleton::getTransport();

    ::comms::tcp_devmem::Handle handle{};
    ::comms::tcp_devmem::ListenerInterface* listenComm{};
    COMMCHECKTHROW(transport->listen(netdev_, &handle, &listenComm));

    FB_SYSCHECKTHROW_EX(
        socket.send(&handle, sizeof(handle)), rank_, commHash_, commDesc_);

    ::comms::tcp_devmem::CommunicatorInterface* recvComm;
    COMMCHECKTHROW(transport->accept(listenComm, &recvComm));
    COMMCHECKTHROW(transport->closeListen(listenComm));
    recvComm->setCommStats(&profilerCtx_.commStats);

    bootstrapAddRecvPeer(peerRank, recvComm);

    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-TCPDM: Established data connection: commHash {:x}, "
        "commDesc {}, rank {}, peer {}",
        commHash_,
        commDesc_,
        rank_,
        peerRank);
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-TCPDM: Accept thread terminating for commHash {:x}, commDesc {}, rank {}",
      commHash_,
      commDesc_,
      rank_);
}

void CtranTcpDm::bootstrapAddSendPeer(
    int peerRank,
    ::comms::tcp_devmem::CommunicatorInterface* comm) {
  std::lock_guard lock(mutex_);
  sendComms_[peerRank] = comm;
}

commResult_t CtranTcpDm::bootstrapConnect(
    int peerRank,
    const folly::SocketAddress& peerSockAddr) {
  commResult_t res = commSuccess;

  ctran::bootstrap::Socket sock;
  int err = sock.connect(
      peerSockAddr,
      NCCL_CLIENT_SOCKET_IFNAME,
      std::chrono::milliseconds(NCCL_SOCKET_RETRY_SLEEP_MSEC),
      NCCL_SOCKET_RETRY_CNT);
  if (err != 0) {
    return mapBootstrapSocketError(
        err, "connect", rank_, peerRank, commHash_, commDesc_);
  }

  err = sock.send(&rank_, sizeof(int));
  if (err != 0) {
    return mapBootstrapSocketError(
        err, "send", rank_, peerRank, commHash_, commDesc_);
  }

  ::comms::tcp_devmem::Handle handle{};
  err = sock.recv(&handle, sizeof(handle));
  if (err != 0) {
    return mapBootstrapSocketError(
        err, "recv", rank_, peerRank, commHash_, commDesc_);
  }

  ::comms::tcp_devmem::CommunicatorInterface* sendComm{};
  COMMCHECKTHROW(
      CtranTcpDmSingleton::getTransport()->connect(
          netdev_, &handle, &sendComm));
  sendComm->setCommStats(&profilerCtx_.commStats);

  bootstrapAddSendPeer(peerRank, sendComm);

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-TCPDM: Established data connection: commHash {:x}, "
      "commDesc {}, pimpl {}, rank {}, peer {}",
      commHash_,
      commDesc_,
      (void*)this,
      rank_,
      peerRank);

  return res;
}

CtranTcpDm::CtranTcpDm(CtranComm* comm, ctran::Profiler* profiler) {
  comm_ = comm;
  cudaDev_ = comm->statex_->cudaDev();
  rank_ = comm->statex_->rank();
  nRanks_ = comm->statex_->nRanks();
  commHash_ = comm->statex_->commHash();
  commDesc_ = comm->statex_->commDesc();

  auto transport = CtranTcpDmSingleton::getTransport();
  netdev_ = transport->getDeviceFor(cudaDev_);
  transport->open(netdev_);

  profilerCtx_.cuDev = cudaDev_;
  profilerCtx_.rank = rank_;
  profilerCtx_.nRanks = nRanks_;
  profilerCtx_.commHash = commHash_;

  bootstrapPrepare(comm->bootstrap_.get());

  registerProfilerHooks(profiler);

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-TCPDM: created TCPDM backend {} for commHash {:x} commDesc {}",
      (void*)this,
      commHash_,
      commDesc_);
}

CtranTcpDm::~CtranTcpDm() {
  listenSocket_.shutdown();
  listenThread_.join();

  const uint32_t closeFlags = comm_ != nullptr && comm_->testAbort()
      ? ::comms::tcp_devmem::kCloseFlagForce
      : 0;
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-TCPDM: destroying backend {} commHash {:x} commDesc {} aborted {} closeFlags {:#x}",
      (void*)this,
      commHash_,
      commDesc_,
      aborted_.load(),
      closeFlags);
  closeComms("backend destruction", closeFlags);
  CtranTcpDmSingleton::getTransport()->shutdown(false);
}

void CtranTcpDm::closeComms(const char* reason, uint32_t closeFlags) {
  std::unordered_map<int, ::comms::tcp_devmem::CommunicatorInterface*>
      sendComms;
  std::unordered_map<int, ::comms::tcp_devmem::CommunicatorInterface*>
      recvComms;
  size_t queuedRecvCount = 0;
  size_t cancelledQueuedRecvCount = 0;
  {
    std::lock_guard lock(mutex_);
    queuedRecvCount = queuedRecv_.size();
    for (auto& recvReq : queuedRecv_) {
      if (recvReq->req != nullptr) {
        recvReq->req->complete(::comms::tcp_devmem::Status::RemoteError);
        ++cancelledQueuedRecvCount;
      }
    }
    for (auto& ctrlReq : queuedCtrlRecv_) {
      if (ctrlReq->req != nullptr) {
        ctrlReq->req->complete(::comms::tcp_devmem::Status::RemoteError);
        ++cancelledQueuedRecvCount;
      }
    }
    queuedRecv_.clear();
    queuedCtrlRecv_.clear();
    sendComms.swap(sendComms_);
    recvComms.swap(recvComms_);
  }

  const bool forceClose = closeFlags & ::comms::tcp_devmem::kCloseFlagForce;
  if (forceClose) {
    CLOGF(
        WARN,
        "CTRAN-TCPDM: closing backend {} rank {} commHash {:x} commDesc {} reason {} flags {:#x} sendComms {} recvComms {} queuedRecvs {} cancelledQueuedRecvs {}",
        (void*)this,
        rank_,
        commHash_,
        commDesc_,
        reason == nullptr ? "unknown" : reason,
        closeFlags,
        sendComms.size(),
        recvComms.size(),
        queuedRecvCount,
        cancelledQueuedRecvCount);
  } else {
    CLOGF(
        INFO,
        "CTRAN-TCPDM: closing backend {} rank {} commHash {:x} commDesc {} reason {} flags {:#x} sendComms {} recvComms {} queuedRecvs {} cancelledQueuedRecvs {}",
        (void*)this,
        rank_,
        commHash_,
        commDesc_,
        reason == nullptr ? "unknown" : reason,
        closeFlags,
        sendComms.size(),
        recvComms.size(),
        queuedRecvCount,
        cancelledQueuedRecvCount);
  }

  auto transport = CtranTcpDmSingleton::getTransport();
  for (auto& [peerRank, comm] : sendComms) {
    auto status = transport->closeSend(comm, closeFlags);
    if (status != ::comms::tcp_devmem::Status::Ok) {
      CLOGF(
          WARN,
          "CTRAN-TCPDM: closeSend failed for peer {} on rank {} commHash {:x} commDesc {} reason {} flags {:#x} status {}",
          peerRank,
          rank_,
          commHash_,
          commDesc_,
          reason == nullptr ? "unknown" : reason,
          closeFlags,
          static_cast<int>(status));
    }
  }

  for (auto& [peerRank, comm] : recvComms) {
    auto status = transport->closeRecv(comm, closeFlags);
    if (status != ::comms::tcp_devmem::Status::Ok) {
      CLOGF(
          WARN,
          "CTRAN-TCPDM: closeRecv failed for peer {} on rank {} commHash {:x} commDesc {} reason {} flags {:#x} status {}",
          peerRank,
          rank_,
          commHash_,
          commDesc_,
          reason == nullptr ? "unknown" : reason,
          closeFlags,
          static_cast<int>(status));
    }
  }
}

void CtranTcpDm::abortOutstanding(const char* reason) {
  if (aborted_.exchange(true)) {
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-TCPDM: backend {} already aborted for rank {} commHash {:x} commDesc {} reason {}",
        (void*)this,
        rank_,
        commHash_,
        commDesc_,
        reason == nullptr ? "unknown" : reason);
    return;
  }

  closeComms(reason, ::comms::tcp_devmem::kCloseFlagForce);
}

void CtranTcpDm::profilerStart() {
  CtranTcpDmSingleton::getTransport()->profilerStart(profilerCtx_);
}

void CtranTcpDm::profilerEnd() {
  CtranTcpDmSingleton::getTransport()->profilerEnd(profilerCtx_);
}

commResult_t CtranTcpDm::preConnect(const std::unordered_set<int>& peerRanks) {
  if (aborted_.load()) {
    return commRemoteError;
  }
  for (int peerRank : peerRanks) {
    FB_COMMCHECK(connectPeer(peerRank));
  }

  return commSuccess;
}

commResult_t CtranTcpDm::regMem(
    const void* buf,
    const size_t len,
    const int cudaDev,
    void** handle) {
  auto transport = CtranTcpDmSingleton::getTransport();

  auto dev = transport->getDeviceFor(cudaDev);

  int dmabufFd = ctran::utils::getCuMemDmaBufFd(buf, len);
  ::comms::tcp_devmem::MemHandleInterface* mhandle = nullptr;
  if (dmabufFd < 0) {
    COMMCHECK_TCP(transport->regMr(dev, (void*)buf, len, &mhandle));
  } else {
    COMMCHECK_TCP(
        transport->regDmabufMr(dev, (void*)buf, len, dmabufFd, &mhandle));
  }
  *handle = reinterpret_cast<void*>(mhandle);

  return commSuccess;
}

commResult_t CtranTcpDm::deregMem(void* handle) {
  auto transport = CtranTcpDmSingleton::getTransport();
  auto* mhandle =
      reinterpret_cast<::comms::tcp_devmem::MemHandleInterface*>(handle);

  COMMCHECK_TCP(transport->deregMr(mhandle));

  return commSuccess;
}

commResult_t CtranTcpDm::isend(
    int peerRank,
    void* handle,
    void* data,
    size_t size,
    CtranTcpDmRequest& req) {
  const auto connectResult = connectPeer(peerRank);
  if (connectResult != commSuccess) {
    if (connectResult == commRemoteError) {
      req.complete(::comms::tcp_devmem::Status::RemoteError);
    }
    return connectResult;
  }

  ::comms::tcp_devmem::CommunicatorInterface* comm = nullptr;
  {
    std::lock_guard lock(mutex_);
    if (aborted_.load()) {
      req.complete(::comms::tcp_devmem::Status::RemoteError);
      return commRemoteError;
    }
    auto it = sendComms_.find(peerRank);
    if (it == sendComms_.end()) {
      return commInternalError;
    }
    comm = it->second;
  }

  auto transport = CtranTcpDmSingleton::getTransport();
  ::comms::tcp_devmem::RequestInterface* request{nullptr};
  COMMCHECK_TCP(transport->queueRequest(
      comm,
      ::comms::tcp_devmem::Transport::Op::Send,
      data,
      size,
      handle,
      &request));
  req.track(transport.get(), request);

  return commSuccess;
}

commResult_t CtranTcpDm::connectPeer(int peerRank) {
  {
    std::lock_guard lock(mutex_);
    if (aborted_.load()) {
      return commRemoteError;
    }
    if (sendComms_.find(peerRank) != sendComms_.end()) {
      return commSuccess;
    }
  }

  folly::SocketAddress peerAddr;
  peerAddr.setFromSockaddr(
      reinterpret_cast<sockaddr_in6*>(&allListenSocketAddrs_[peerRank]));
  return bootstrapConnect(peerRank, peerAddr);
}

commResult_t CtranTcpDm::irecvCtrlMsgConnected(
    int peerRank,
    std::shared_ptr<std::array<uint8_t, 1>> storage,
    CtranTcpDmRequest& req) {
  ::comms::tcp_devmem::CommunicatorInterface* comm = nullptr;
  {
    std::lock_guard lock(mutex_);
    if (aborted_.load()) {
      req.complete(::comms::tcp_devmem::Status::RemoteError);
      return commRemoteError;
    }
    auto it = recvComms_.find(peerRank);
    if (it == recvComms_.end()) {
      return commInternalError;
    }
    comm = it->second;
  }

  auto transport = CtranTcpDmSingleton::getTransport();
  ::comms::tcp_devmem::RequestInterface* request{nullptr};
  COMMCHECK_TCP(transport->queueRequest(
      comm,
      ::comms::tcp_devmem::Transport::Op::RecvCtrl,
      storage->data(),
      storage->size(),
      nullptr,
      &request));
  req.track(transport.get(), request, std::move(storage));
  return commSuccess;
}

void CtranTcpDm::ctrlRecvProgress() {
  while (true) {
    std::unique_ptr<CtrlRecvRequest> ctrlReq;
    {
      std::unique_lock lock(mutex_);
      for (auto it = queuedCtrlRecv_.begin(); it != queuedCtrlRecv_.end();
           ++it) {
        if (recvComms_.find((*it)->peerRank) == recvComms_.end()) {
          continue;
        }
        ctrlReq = std::move(*it);
        queuedCtrlRecv_.erase(it);
        break;
      }
    }

    if (ctrlReq == nullptr) {
      return;
    }

    auto result = irecvCtrlMsgConnected(
        ctrlReq->peerRank, std::move(ctrlReq->storage), *ctrlReq->req);
    if (result != commSuccess) {
      ctrlReq->req->complete(::comms::tcp_devmem::Status::RemoteError);
      return;
    }
  }
}

commResult_t CtranTcpDm::isendCtrlMsg(
    const ControlMsg& msg,
    int peerRank,
    CtranTcpDmRequest& req) {
  if (msg.type != ControlMsgType::SYNC) {
    req.complete();
    return commSuccess;
  }

  const auto connectResult = connectPeer(peerRank);
  if (connectResult != commSuccess) {
    if (connectResult == commRemoteError) {
      req.complete(::comms::tcp_devmem::Status::RemoteError);
    }
    return connectResult;
  }

  ::comms::tcp_devmem::CommunicatorInterface* comm = nullptr;
  {
    std::lock_guard lock(mutex_);
    if (aborted_.load()) {
      req.complete(::comms::tcp_devmem::Status::RemoteError);
      return commRemoteError;
    }
    auto it = sendComms_.find(peerRank);
    if (it == sendComms_.end()) {
      return commInternalError;
    }
    comm = it->second;
  }

  auto storage = std::make_shared<std::array<uint8_t, 1>>();
  (*storage)[0] = 1;
  auto transport = CtranTcpDmSingleton::getTransport();
  ::comms::tcp_devmem::RequestInterface* request{nullptr};
  COMMCHECK_TCP(transport->queueRequest(
      comm,
      ::comms::tcp_devmem::Transport::Op::SendCtrl,
      storage->data(),
      storage->size(),
      nullptr,
      &request));
  req.track(transport.get(), request, std::move(storage));
  return commSuccess;
}

commResult_t CtranTcpDm::irecvCtrlMsg(
    ControlMsg& msg,
    int peerRank,
    CtranTcpDmRequest& req) {
  if (msg.type != ControlMsgType::SYNC) {
    req.complete();
    return commSuccess;
  }

  auto storage = std::make_shared<std::array<uint8_t, 1>>();
  {
    std::unique_lock lock(mutex_);
    if (aborted_.load()) {
      req.complete(::comms::tcp_devmem::Status::RemoteError);
      return commRemoteError;
    }

    if (recvComms_.find(peerRank) == recvComms_.end()) {
      auto ctrlReq = std::make_unique<CtrlRecvRequest>();
      ctrlReq->peerRank = peerRank;
      ctrlReq->storage = storage;
      ctrlReq->req = &req;
      req.markQueuedRecv(storage);
      queuedCtrlRecv_.push_back(std::move(ctrlReq));
      return commSuccess;
    }
  }

  return irecvCtrlMsgConnected(peerRank, std::move(storage), req);
}

commResult_t CtranTcpDm::progress() {
  ctrlRecvProgress();
  recvNotifyProgress();

  while (true) {
    std::unique_ptr<RecvRequest> recvReq;
    {
      std::unique_lock lock(mutex_);
      if (aborted_.load()) {
        return commRemoteError;
      }

      for (auto it = queuedRecv_.begin(); it != queuedRecv_.end(); ++it) {
        if (recvComms_.find((*it)->peerRank) == recvComms_.end()) {
          continue;
        }
        recvReq = std::move(*it);
        queuedRecv_.erase(it);
        break;
      }
    }

    if (recvReq == nullptr) {
      return commSuccess;
    }

    auto result = irecvConnected(
        recvReq->peerRank,
        recvReq->handle,
        recvReq->data,
        recvReq->size,
        *recvReq->req,
        recvReq->unpackPool);
    if (result != commSuccess) {
      recvReq->req->complete(::comms::tcp_devmem::Status::RemoteError);
      return result;
    }
  }
}

void CtranTcpDm::cancelQueuedRecv(CtranTcpDmRequest* req) {
  std::unique_lock lock(mutex_);

  for (auto it = queuedRecv_.begin(); it != queuedRecv_.end();) {
    if ((*it)->req == req) {
      if (req != nullptr) {
        req->complete(::comms::tcp_devmem::Status::RemoteError);
      }
      it = queuedRecv_.erase(it);
    } else {
      ++it;
    }
  }
}

commResult_t CtranTcpDm::irecv(
    int peerRank,
    void* handle,
    void* data,
    size_t size,
    CtranTcpDmRequest& req,
    void* unpackPool) {
  {
    std::unique_lock lock(mutex_);
    if (aborted_.load()) {
      req.complete(::comms::tcp_devmem::Status::RemoteError);
      return commRemoteError;
    }

    // Peer is not connected, queue this operation. We can't block
    // the irecv callers. progress() should be called periodically to
    // attempt to post these requests again.
    if (recvComms_.find(peerRank) == recvComms_.end()) {
      auto recvReq = std::make_unique<RecvRequest>();
      recvReq->peerRank = peerRank;
      recvReq->handle = handle;
      recvReq->data = data;
      recvReq->size = size;
      recvReq->req = &req;
      recvReq->unpackPool = unpackPool;
      req.markQueuedRecv();
      queuedRecv_.push_back(std::move(recvReq));
      return commSuccess;
    }
  }

  return irecvConnected(peerRank, handle, data, size, req, unpackPool);
}

commResult_t CtranTcpDm::irecvConnected(
    int peerRank,
    void* handle,
    void* data,
    size_t size,
    CtranTcpDmRequest& req,
    void* unpackPool) {
  ::comms::tcp_devmem::CommunicatorInterface* comm = nullptr;
  {
    std::lock_guard lock(mutex_);
    if (aborted_.load()) {
      req.complete(::comms::tcp_devmem::Status::RemoteError);
      return commRemoteError;
    }
    auto it = recvComms_.find(peerRank);
    if (it == recvComms_.end()) {
      return commInternalError;
    }
    comm = it->second;
  }
  if (!comm) {
    return commInternalError;
  }

  auto transport = CtranTcpDmSingleton::getTransport();
  ::comms::tcp_devmem::RequestInterface* request{nullptr};
  COMMCHECK_TCP(transport->queueRequest(
      comm,
      ::comms::tcp_devmem::Transport::Op::Recv,
      data,
      size,
      handle,
      &request,
      unpackPool));

  req.track(transport.get(), request);

  return commSuccess;
}

commResult_t CtranTcpDm::irecvCounted(
    int peerRank,
    void* handle,
    void* data,
    size_t size,
    void* unpackPool) {
  auto req = std::make_unique<CtranTcpDmRequest>();
  auto* rawReq = req.get();
  pendingRecvNotifies_[peerRank].push_back(std::move(req));

  {
    std::unique_lock lock(mutex_);
    if (recvComms_.find(peerRank) == recvComms_.end()) {
      auto recvReq = std::make_unique<RecvRequest>();
      recvReq->peerRank = peerRank;
      recvReq->handle = handle;
      recvReq->data = data;
      recvReq->size = size;
      recvReq->req = rawReq;
      recvReq->unpackPool = unpackPool;
      rawReq->markQueuedRecv();
      queuedRecv_.push_back(std::move(recvReq));
      return commSuccess;
    }
  }

  return irecvConnected(peerRank, handle, data, size, *rawReq, unpackPool);
}

void CtranTcpDm::recvNotifyProgress() {
  for (auto& [peerRank, pending] : pendingRecvNotifies_) {
    while (!pending.empty() && pending.front()->isComplete()) {
      pending.pop_front();
      recvNotifyCount_[peerRank]++;
    }
  }
}

commResult_t CtranTcpDm::checkNotify(int peerRank, bool* done) {
  recvNotifyProgress();
  auto it = recvNotifyCount_.find(peerRank);
  if (it != recvNotifyCount_.end() && it->second > 0) {
    it->second--;
    *done = true;
  } else {
    *done = false;
  }
  return commSuccess;
}

commResult_t
CtranTcpDm::prepareUnpackConsumer(SQueues* sqs, size_t blocks, void** pool) {
  COMMCHECK_TCP(
      CtranTcpDmSingleton::getTransport()->prepareUnpackConsumer(
          netdev_, sqs, blocks, pool));
  return commSuccess;
}

commResult_t CtranTcpDm::teardownUnpackConsumer(void* pool) {
  COMMCHECK_TCP(
      CtranTcpDmSingleton::getTransport()->teardownUnpackConsumer(
          netdev_, pool));
  return commSuccess;
}

void CtranTcpDm::registerProfilerHooks(ctran::Profiler* profiler) {
  if (!profiler) {
    return;
  }
  profiler->registerStartHook(
      ProfilerEvent::ALGO_TOTAL, [this]() { profilerStart(); });
  profiler->registerEndHook(
      ProfilerEvent::ALGO_TOTAL, [this]() { profilerEnd(); });
}

} // namespace ctran
