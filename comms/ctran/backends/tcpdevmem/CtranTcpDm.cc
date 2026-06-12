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

    // Negative rank = ctrl socket connection (bufSync)
    if (peerRank < 0) {
      int actualPeer = -(peerRank + 1);
      int ctrlFd = socket.getFd();
      std::lock_guard lock(mutex_);
      ctrlSocks_.emplace(actualPeer, std::move(socket));
      CLOGF_SUBSYS(
          INFO,
          INIT,
          "CTRAN-TCPDM: ctrl socket accepted from peer {} fd={}",
          actualPeer,
          ctrlFd);
      continue;
    }

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
  FB_SYSCHECKRETURN(
      sock.connect(
          peerSockAddr,
          NCCL_CLIENT_SOCKET_IFNAME,
          std::chrono::milliseconds(NCCL_SOCKET_RETRY_SLEEP_MSEC),
          NCCL_SOCKET_RETRY_CNT),
      commInternalError);
  FB_SYSCHECKRETURN(sock.send(&rank_, sizeof(int)), commInternalError);

  ::comms::tcp_devmem::Handle handle{};
  FB_SYSCHECKRETURN(sock.recv(&handle, sizeof(handle)), commInternalError);

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

void CtranTcpDm::ensureCtrlSocket(int peerRank) {
  {
    std::lock_guard lock(mutex_);
    if (ctrlSocks_.count(peerRank)) {
      return;
    }
  }

  folly::SocketAddress peerAddr;
  peerAddr.setFromSockaddr(
      reinterpret_cast<sockaddr_in6*>(&allListenSocketAddrs_[peerRank]));
  ctran::bootstrap::Socket sock;
  FB_SYSCHECKTHROW_EX(
      sock.connect(
          peerAddr,
          NCCL_CLIENT_SOCKET_IFNAME,
          std::chrono::milliseconds(NCCL_SOCKET_RETRY_SLEEP_MSEC),
          NCCL_SOCKET_RETRY_CNT),
      rank_,
      commHash_,
      commDesc_);
  int marker = -(rank_ + 1);
  FB_SYSCHECKTHROW_EX(
      sock.send(&marker, sizeof(int)), rank_, commHash_, commDesc_);
  std::lock_guard lock(mutex_);
  ctrlSocks_.emplace(peerRank, std::move(sock));
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
    queuedRecv_.clear();
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

void CtranTcpDm::ctrlSyncProgress() {
  for (auto& [peerRank, sock] : ctrlSocks_) {
    auto& pending = pendingSyncRecvs_[peerRank];
    while (!pending.empty()) {
      uint8_t sync;
      int ret = ::recv(sock.getFd(), &sync, sizeof(sync), MSG_DONTWAIT);
      if (ret <= 0) {
        break;
      }
      syncRecvCount_[peerRank]++;
      pending.front()->complete();
      pending.pop_front();
      CLOGF_SUBSYS(
          INFO,
          COLL,
          "CTRAN-TCPDM: ctrlSyncProgress completed sync from peer {}, total={}, remaining={}, fd={}",
          peerRank,
          syncRecvCount_[peerRank],
          pending.size(),
          sock.getFd());
    }
  }
}

commResult_t CtranTcpDm::isendCtrlMsg(
    const ControlMsg& msg,
    int peerRank,
    CtranTcpDmRequest& req) {
  // only allow sync messages to be sent on the ctrl socket
  if (msg.type != ControlMsgType::SYNC) {
    req.complete();
    return commSuccess;
  }
  // only sendeer can do lazy connect to avoid deadlock.
  ensureCtrlSocket(peerRank);
  std::lock_guard lock(mutex_);
  auto& sock = ctrlSocks_.at(peerRank);
  uint8_t sync = 1;
  sock.send(&sync, sizeof(sync));
  req.complete();
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
  std::lock_guard lock(mutex_);
  pendingSyncRecvs_[peerRank].push_back(&req);
  return commSuccess;
}

commResult_t CtranTcpDm::progress() {
  ctrlSyncProgress();
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
