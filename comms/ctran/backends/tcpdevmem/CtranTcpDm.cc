// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/backends/tcpdevmem/CtranTcpDm.h"
#include "comms/ctran/backends/tcpdevmem/CtranTcpDmSingleton.h"
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

static folly::once_flag enabledFlag;
static bool enabled;

bool CtranTcpDm::isEnabled() {
  folly::call_once(enabledFlag, [&]() {
    auto it = std::find(
        NCCL_CTRAN_BACKENDS.begin(),
        NCCL_CTRAN_BACKENDS.end(),
        NCCL_CTRAN_BACKENDS::tcpdm);
    enabled = it != NCCL_CTRAN_BACKENDS.end();
  });
  return enabled;
}

void CtranTcpDm::bootstrapPrepare(ctran::bootstrap::IBootstrap* bootstrap) {
  folly::SocketAddress ifAddrSockAddr;
  sockaddr_in6 sin6{};
  sin6.sin6_family = AF_INET6;
  sin6.sin6_addr = netdev_->addr;
  ifAddrSockAddr.setFromSockaddr(&sin6);
  FB_SYSCHECKTHROW(listenSocket_.bindAndListen(ifAddrSockAddr, *netdev_->name));

  std::string line =
      ::comms::tcp_devmem::addrToString(&netdev_->addr, 0, *netdev_->name);
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
    FB_SYSCHECKTHROW(maybeListenAddr.error());
  }
  maybeListenAddr->getAddress(&allListenSocketAddrs_[rank_]);

  auto resFuture = bootstrap->allGather(
      allListenSocketAddrs_.data(),
      sizeof(allListenSocketAddrs_.at(0)),
      rank_,
      nRanks_);
  FB_COMMCHECKTHROW(static_cast<commResult_t>(std::move(resFuture).get()));

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
    ::comms::tcp_devmem::Communicator* comm) {
  std::lock_guard lock(mutex_);
  recvComms_[peerRank] = comm;
}

void CtranTcpDm::bootstrapAccept() {
  // Set cudaDev for logging
  FB_CUDACHECKTHROW(cudaSetDevice(cudaDev_));
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
      FB_SYSCHECKTHROW(maybeSocket.error());
    }
    auto& socket = maybeSocket.value();
    FB_SYSCHECKTHROW(socket.recv(&peerRank, sizeof(int)));

    ::comms::tcp_devmem::Handle handle{};
    ::comms::tcp_devmem::CommunicatorListener* listenComm{};
    COMMCHECKTHROW(transport_->listen(netdev_, &handle, &listenComm));

    FB_SYSCHECKTHROW(socket.send(&handle, sizeof(handle)));

    ::comms::tcp_devmem::Communicator* recvComm;
    COMMCHECKTHROW(transport_->accept(listenComm, &recvComm));
    COMMCHECKTHROW(transport_->closeListen(listenComm));

    bootstrapAddRecvPeer(peerRank, recvComm);

    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-TC: Established connection: commHash {:x}, commDesc {}, "
        "rank {}, peer {}",
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
    ::comms::tcp_devmem::Communicator* comm) {
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

  ::comms::tcp_devmem::Communicator* sendComm{};
  COMMCHECKTHROW(transport_->connect(netdev_, &handle, &sendComm));

  bootstrapAddSendPeer(peerRank, sendComm);

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-TCPDM: Established connection: commHash {:x}, commDesc {}, pimpl {}, "
      "rank {}, peer {}",
      commHash_,
      commDesc_,
      (void*)this,
      rank_,
      peerRank);

  return res;
}

CtranTcpDm::CtranTcpDm(
    [[maybe_unused]] CtranComm* comm,
    [[maybe_unused]] CtranCtrlManager* ctrlMgr) {
  transport_ = CtranTcpDmSingleton::getTransport();

  cudaDev_ = comm->statex_->cudaDev();
  rank_ = comm->statex_->rank();
  nRanks_ = comm->statex_->nRanks();
  commHash_ = comm->statex_->commHash();
  commDesc_ = comm->statex_->commDesc();
  netdev_ = transport_->getDeviceFor(cudaDev_);

  bootstrapPrepare(comm->bootstrap_.get());

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

  for (auto comm : sendComms_) {
    transport_->closeSend(comm.second);
  }
  for (auto comm : recvComms_) {
    transport_->closeRecv(comm.second);
  }

  transport_.reset();
}

commResult_t CtranTcpDm::preConnect(const std::unordered_set<int>& peerRanks) {
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

  ::comms::tcp_devmem::NetDev* dev = transport->getDeviceFor(cudaDev);

  int dmabufFd = ctran::utils::getCuMemDmaBufFd(buf, len);
  ::comms::tcp_devmem::MemHandle* mhandle = nullptr;
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
  auto* mhandle = reinterpret_cast<::comms::tcp_devmem::MemHandle*>(handle);

  COMMCHECK_TCP(transport->deregMr(mhandle));

  return commSuccess;
}

commResult_t CtranTcpDm::isend(
    int peerRank,
    void* handle,
    void* data,
    size_t size,
    CtranTcpDmRequest& req) {
  FB_COMMCHECK(connectPeer(peerRank));

  ::comms::tcp_devmem::Communicator* comm = sendComms_.at(peerRank);

  ::comms::tcp_devmem::Request* request{nullptr};
  COMMCHECK_TCP(transport_->queueRequest(
      comm,
      ::comms::tcp_devmem::Transport::Op::Send,
      data,
      size,
      handle,
      &request));
  req.track(transport_, request);

  return commSuccess;
}

commResult_t CtranTcpDm::connectPeer(int peerRank) {
  if (sendComms_.find(peerRank) != sendComms_.end()) {
    return commSuccess;
  }

  folly::SocketAddress peerAddr;
  peerAddr.setFromSockaddr(
      reinterpret_cast<sockaddr_in6*>(&allListenSocketAddrs_[peerRank]));
  return bootstrapConnect(peerRank, peerAddr);
}

commResult_t CtranTcpDm::progress() {
  std::unique_lock lock(mutex_);

  for (auto it = queuedRecv_.begin(); it != queuedRecv_.end();) {
    auto& recvReq = *it;

    if (recvComms_.find(recvReq->peerRank) == recvComms_.end()) {
      ++it;
      continue;
    }

    FB_COMMCHECK(irecvConnected(
        recvReq->peerRank,
        recvReq->handle,
        recvReq->data,
        recvReq->size,
        *recvReq->req));

    it = queuedRecv_.erase(it);
  }

  return commSuccess;
}

commResult_t CtranTcpDm::irecv(
    int peerRank,
    void* handle,
    void* data,
    size_t size,
    CtranTcpDmRequest& req) {
  {
    std::unique_lock lock(mutex_);

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
      queuedRecv_.push_back(std::move(recvReq));
      return commSuccess;
    }
  }

  return irecvConnected(peerRank, handle, data, size, req);
}

commResult_t CtranTcpDm::irecvConnected(
    int peerRank,
    void* handle,
    void* data,
    size_t size,
    CtranTcpDmRequest& req) {
  ::comms::tcp_devmem::Communicator* comm = recvComms_.at(peerRank);
  if (!comm) {
    return commInternalError;
  }

  ::comms::tcp_devmem::Request* request{nullptr};
  COMMCHECK_TCP(transport_->queueRequest(
      comm,
      ::comms::tcp_devmem::Transport::Op::Recv,
      data,
      size,
      handle,
      &request));
  req.track(transport_, request);

  return commSuccess;
}

commResult_t CtranTcpDm::prepareUnpackConsumer(SQueues* sqs, size_t blocks) {
  COMMCHECK_TCP(transport_->prepareUnpackConsumer(netdev_, sqs, blocks));
  return commSuccess;
}

} // namespace ctran
