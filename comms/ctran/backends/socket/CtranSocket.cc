// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/backends/socket/CtranSocket.h"
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "comms/ctran/backends/socket/CtranSocketBase.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CtranLogUtils.h"
#include "comms/ctran/utils/Debug.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"

CtranSocket::CtranSocket(CtranComm* comm)
    : comm(comm),
      rank_(comm->statex_->rank()),
      cudaDev_(comm->statex_->cudaDev()),
      commHash_(comm->statex_->commHash()),
      commDesc_(comm->statex_->commDesc()) {
  init(SocketServerAddr());
  CTRAN_LOG_SUBSYS(
      INFO,
      INIT,
      "CTRAN-SOCKET: Initialized {} from comm {}",
      (void*)this,
      (void*)comm);
}

CtranSocket::CtranSocket(
    int rank,
    int cudaDev,
    uint64_t commHash,
    const std::string& commDesc,
    const SocketServerAddr& serverAddr)
    : comm(nullptr),
      rank_(rank),
      cudaDev_(cudaDev),
      commHash_(commHash),
      commDesc_(commDesc) {
  init(serverAddr);
  CTRAN_LOG_SUBSYS(
      INFO,
      INIT,
      "CTRAN-SOCKET: Initialized {} from rank {} cudaDev {} commHash {:x} commDesc {}",
      (void*)this,
      rank,
      cudaDev,
      commHash,
      commDesc);
}

CtranSocket::~CtranSocket(void) {
  listenSocket_.shutdown();
  listenThread_.join();
  {
    auto locked = socketMaps_.wlock();
    for (auto it = locked->rankToSocket.begin();
         it != locked->rankToSocket.end();
         ++it) {
      it->second->close();
      it->second.reset();
    }
  }

  CTRAN_LOG_SUBSYS(
      INFO,
      INIT,
      "CTRAN-SOCKET: destroyed SOCKET backend {} for commHash {:x} commDesc {}",
      (void*)this,
      commHash_,
      commDesc_);
}

commResult_t CtranSocket::preConnect(const std::unordered_set<int>& peerRanks) {
  bool newConnection = false;
  // if map is empty, it means we don't know comm size and peerAddr
  if (preConnectPeerMap_.empty()) {
    return commSuccess;
  }
  // Actively connect to peers with larger rank number, if not already connected
  for (int peerRank : peerRanks) {
    FB_COMMCHECK(checkValidPeer(peerRank));
    if (rank_ < peerRank && getSocket(peerRank) == nullptr) {
      FB_COMMCHECK(bootstrapConnect(peerRank, SocketServerAddr()));
    }
  }
  // check if all requested peers are connected
  for (int peerRank : peerRanks) {
    if (!preConnectPeerMap_.at(peerRank)) {
      while (getSocket(peerRank) == nullptr) {
        // wait listening thread to establish connections with rest of peers
        // with smaller rank number
      }
      preConnectPeerMap_.at(peerRank) = true;
      newConnection = true;
    }
  }

  if (newConnection) {
    CTRAN_LOG_SUBSYS(
        INFO,
        INIT,
        "CTRAN-SOCKET: Rank-{} Pre-connected peers: [{}]",
        rank_,
        unorderedSetToStr(peerRanks));
  }

  return commSuccess;
}

void CtranSocket::init(const SocketServerAddr& serverAddr) {
  ncclLogData_ = CommLogData{
      .commId = comm ? comm->logMetaData_.commId : 0,
      .commHash = commHash_,
      .commDesc = commDesc_,
      .rank = rank_,
      .nRanks = comm ? comm->statex_->nRanks() : 1};

  if (comm) {
    // initialize pre-connection map
    this->preConnectPeerMap_.resize(comm->statex_->nRanks(), false);
    // only exchange listen sockets with bootstrapAllGather when using comm to
    // initialize CtranSocket
    std::string resolvedIfName;
    auto maybeAddr = ctran::bootstrap::getInterfaceAddress(
        NCCL_SOCKET_IFNAME, NCCL_SOCKET_IPADDR_PREFIX, true, &resolvedIfName);
    if (maybeAddr.hasError()) {
      std::string msg = fmt::format(
          "CTRAN-SOCKET: No socket interfaces found (NCCL_SOCKET_IFNAME={}, NCCL_SOCKET_IPADDR_PREFIX={})",
          NCCL_SOCKET_IFNAME,
          NCCL_SOCKET_IPADDR_PREFIX);
      CTRAN_ERR(commSystemError, "{}", msg);
      throw ctran::utils::Exception(
          msg, commSystemError, rank_, commHash_, commDesc_);
    } else {
      CTRAN_LOG_SUBSYS(
          INFO,
          INIT,
          "CTRAN-SOCKET: socket address set to {} on interface {}",
          maybeAddr->str(),
          resolvedIfName);
    }

    folly::SocketAddress ifAddrSockAddr(maybeAddr.value(), 0 /* port */);
    FB_SYSCHECKTHROW_EX(
        listenSocket_.bindAndListen(ifAddrSockAddr, resolvedIfName),
        ncclLogData_);
    CTRAN_LOG_SUBSYS(
        INFO,
        INIT,
        "CTRAN-IB: Rank {} created listen socket based on a self-finding address {}",
        rank_,
        listenSocket_.getListenAddress()->describe());

    allListenSocketAddrs_.resize(comm->statex_->nRanks());
    auto maybeListenAddr = listenSocket_.getListenAddress();
    if (maybeListenAddr.hasError()) {
      FB_SYSCHECKTHROW_EX(maybeListenAddr.error(), ncclLogData_);
    }
    maybeListenAddr->getAddress(&allListenSocketAddrs_[rank_]);

    // exchange listen socket addresses with peers
    auto resFuture = comm->bootstrap_->allGather(
        allListenSocketAddrs_.data(),
        sizeof(allListenSocketAddrs_[0]),
        comm->statex_->rank(),
        comm->statex_->nRanks());
    FB_COMMCHECKTHROW_EX(
        static_cast<commResult_t>(std::move(resFuture).get()),
        comm->logMetaData_);
  } else {
    // use provided addr(i.e. ip, port, host) to initialize ctranSocket
    auto serverAddrSockAddr = toSocketAddress(serverAddr);
    listenSocket_.bindAndListen(serverAddrSockAddr, serverAddr.ifName);
    CTRAN_LOG_SUBSYS(
        INFO,
        INIT,
        "CTRAN-SOCKET: Rank {} created listen socket based on provided address {} "
        "and will use interface {} to connect",
        rank_,
        serverAddrSockAddr.describe(),
        serverAddr.ifName);
  }
  listenThread_ = std::thread{[this]() { bootstrapAccept(); }};
  CTRAN_LOG_SUBSYS(
      INFO,
      INIT,
      "CTRAN-SOCKET: created SOCKET backend {} for commHash {} commDesc {}",
      (void*)this,
      commHash_,
      commDesc_);
}

void CtranSocket::bootstrapAccept() {
  // Set cudaDev for logging
  FB_CUDACHECKTHROW_EX(cudaSetDevice(cudaDev_), comm->logMetaData_);
  commNamedThreadStart(
      "CTranSocketListen", rank_, commHash_, commDesc_, __func__);
  while (1) {
    int peerRank;
    // Create a new socket for the peer
    auto maybeSocket = listenSocket_.accept(true);
    if (maybeSocket.hasError()) {
      if (maybeSocket.error() == EBADF || maybeSocket.error() == EINVAL) {
        break; // listen socket is closed
      }
      FB_SYSCHECKTHROW_EX(maybeSocket.error(), ncclLogData_);
    }
    auto socket = std::make_unique<ctran::bootstrap::Socket>(
        std::move(maybeSocket.value()));
    FB_SYSCHECKTHROW_EX(socket->recv(&peerRank, sizeof(int)), ncclLogData_);

    CTRAN_LOG_SUBSYS(
        INFO,
        INIT,
        "CTRAN-SOCKET: Established connection: commHash {:x}, commDesc {}, "
        "socket {}, rank {}, peer {}",
        commHash_,
        commDesc_,
        (void*)socket.get(),
        rank_,
        peerRank);

    // Store the nccl socket
    FB_COMMCHECKTHROW_EX(checkValidPeer(peerRank), comm->logMetaData_);
    FB_COMMCHECKTHROW_EX(
        updateSocket(std::move(socket), peerRank), comm->logMetaData_);
  }

  CTRAN_LOG_SUBSYS(
      INFO,
      INIT,
      "CTRAN-SOCKET: Listen thread terminating for commHash {:x}, commDesc {}, rank {}",
      commHash_,
      commDesc_,
      rank_);

  return;
}

commResult_t CtranSocket::bootstrapConnect(
    int peerRank,
    const SocketServerAddr& peerAddr) {
  if (allListenSocketAddrs_.size() > 0) {
    // ctran-socket is initialized with comm
    return bootstrapConnect(
        peerRank, toSocketAddress(allListenSocketAddrs_[peerRank]));
  } else {
    // ctran-socket is initialized without comm
    // we need to use peerAddr since the peer's listening addr is NOT
    // exchanged and stored in allListenSocketAddrs_
    return bootstrapConnect(peerRank, toSocketAddress(peerAddr));
  }
}

commResult_t CtranSocket::bootstrapConnect(
    int peerRank,
    const folly::SocketAddress& peerSockAddr) {
  commResult_t res = commSuccess;

  auto socket = std::make_unique<ctran::bootstrap::Socket>();
  FB_SYSCHECKRETURN(
      socket->connect(
          peerSockAddr,
          NCCL_CLIENT_SOCKET_IFNAME,
          std::chrono::milliseconds(NCCL_SOCKET_RETRY_SLEEP_MSEC),
          NCCL_SOCKET_RETRY_CNT),
      commInternalError);
  FB_SYSCHECKRETURN(socket->send(&rank_, sizeof(int)), commInternalError);

  if (peerRank == rank_) {
    // if send command to self, close the socket and return
    socket->close();
    goto exit;
  }

  CTRAN_LOG_SUBSYS(
      INFO,
      INIT,
      "CTRAN-SOCKET: Established connection: commHash {:x}, commDesc {}, pimpl {}, "
      "sock {}, rank {}, peer {}",
      commHash_,
      commDesc_,
      (void*)this,
      (void*)socket.get(),
      rank_,
      peerRank);

  // Store the nccl socket
  FB_COMMCHECK(checkValidPeer(peerRank));
  FB_COMMCHECKTHROW_EX(
      updateSocket(std::move(socket), peerRank), comm->logMetaData_);

exit:

  return res;
}

commResult_t CtranSocket::updateSocket(
    std::unique_ptr<ctran::bootstrap::Socket> sock,
    int peerRank) {
  auto locked = socketMaps_.wlock();
  if (locked->rankToSocket.find(peerRank) != locked->rankToSocket.end()) {
    CTRAN_ERR(
        commInternalError,
        "CTRAN-SOCKET: socket already exists for peerRank {} in pimpl {} "
        "commHash {:x}, commDesc {}. It likely indicates a NCCL bug.",
        peerRank,
        (void*)this,
        commHash_,
        commDesc_);
    return commInternalError;
  }
  locked->rankToSocket[peerRank] = std::move(sock);
  return commSuccess;
}

bool CtranSocket::addToPendingOpsIfRequired(
    void* payload,
    std::size_t size,
    int peerRank,
    CtranSocketRequest& req,
    SockPendingOp::OpType opType,
    ctran::bootstrap::Socket* sock) {
  bool pending = false;
  {
    auto locked = rankToPendingOpsMap_.wlock();
    auto it = locked->find(peerRank);
    // If socket is not established or there are pending ops, put the op into
    // pending queue, because we must issue all ops in order.
    // Otherwise, ctrlMsg may mismatch.
    if (!sock || it != locked->end()) {
      auto pendingOp =
          std::make_unique<SockPendingOp>(opType, payload, size, peerRank, req);
      CTRAN_LOG_TRACE(
          COLL,
          "CTRAN-SOCKET: enqueue pendingOp {} size {} peer {}",
          opType,
          size,
          peerRank);

      if (it == locked->end()) {
        // if there's no socket and no pendingops queue, create a queue
        auto res = locked->emplace(peerRank, PendingOpQueue());
        res.first->second.q.push_back(std::move(pendingOp));
      } else {
        it->second.q.push_back(std::move(pendingOp));
      }
      pending = true;
    }
  }
  return pending;
}

commResult_t CtranSocket::progressPendingOps(void) {
  // write a loop to process all pending ops
  std::unordered_map<int, std::vector<std::unique_ptr<SockPendingOp>>>
      readyToIssueOps;

  {
    auto locked = rankToPendingOpsMap_.wlock();
    if (locked->empty()) {
      return commSuccess;
    }
    // Try to find ready-to-issue pending ops for each peer
    // NOTE: we move ctrlMsg issue out of the critical section of pendingOps
    // to avoid potential deadlock with lock to Socket. We expect
    // pendingOps only before initial connection, thus the extra move should
    // not cause perf overhead.
    for (auto it = locked->begin(); it != locked->end();) {
      int peerRank = it->first;
      ctran::bootstrap::Socket* sock = getSocket(peerRank);
      // If Socket has established, move the pendingOps list to
      // readyToIssueOps; otherwise, skip and move to next peer.
      if (sock) {
        for (auto& op : it->second.q) {
          readyToIssueOps[peerRank].push_back(std::move(op));
        }
        // Remove the peer entry once all pending ops are issued.
        // We should no longer have any pending ops for this peer, since new
        // entry would be created only when Socket is not established.
        it = locked->erase(it);
      } else {
        it++;
      }
    }
  }

  // Post ready-to-issue pending ops
  for (auto& [peerRank, pendingOps] : readyToIssueOps) {
    ctran::bootstrap::Socket* sock = getSocket(peerRank);
    for (auto& op : pendingOps) {
      if (op->type == SockPendingOp::OpType::ISEND_CTRL) {
        CTRAN_LOG_TRACE(
            COLL, "CTRAN-SOCKET: socket {} send to {}", (void*)sock, peerRank);
        FB_SYSCHECKRETURN(
            sock->send(&op->packet, op->packet.wireSize()), commInternalError);
        op->req.complete();
      } else {
        postRecvOp(peerRank, std::move(op));
      }
    }
  }
  return commSuccess;
}

commResult_t CtranSocket::progressInternal() {
  FB_COMMCHECK(progressPendingOps());
  std::vector<struct pollfd> fds;
  std::vector<int> peerRanks;
  {
    auto locked = socketMaps_.rlock();
    for (auto it = locked->rankToSocket.begin();
         it != locked->rankToSocket.end();
         it++) {
      if (it->first == rank_) {
        continue;
      }
      fds.emplace_back();
      fds.back().fd = it->second->getFd();
      fds.back().events = POLLIN;
      peerRanks.emplace_back(it->first);
    }
  }
  bool continueWhileLoop = true;
  while (continueWhileLoop) {
    continueWhileLoop = false;
    int count = poll(fds.data(), fds.size(), NCCL_CTRAN_SOCKET_POLL_TIMEOUT);
    if (count < 0) {
      CTRAN_ERR(
          commInternalError,
          "CTRAN-SOCKET: polling error, errno {}",
          strerror(errno));
      return commInternalError;
    } else if (count > 0) {
      CTRAN_LOG_TRACE(COLL, "CTRAN-SOCKET: polling returns {} events", count);
      // there's a socket which receives data, let's continue
      continueWhileLoop = true;
    } else {
      break;
    }
    for (int fid = 0; fid < fds.size(); fid++) {
      if (fds[fid].revents & POLLIN) {
        // let's read it, and dequeue a posted recv from the queue
        ctran::bootstrap::Socket* socket = getSocket(peerRanks[fid]);
        auto& recvQueue = getRecvCtrlQueue(peerRanks[fid]);
        std::optional<SocketCtrlPacket> packet;
        bool connectionClosed = false;
        FB_COMMCHECK(
            doRecvMsg(socket, peerRanks[fid], packet, &connectionClosed));
        if (connectionClosed) {
          // If any peer closes the connection and the socket is removed, we
          // will break the outer loop since the vector of pollfds has to be
          // re-constructed. However, we'll continue the inner loop to receive
          // msgs from the other sockets.
          continueWhileLoop = false;
          continue;
        }
        if (!packet) {
          // Frame arrived in pieces; the remainder is picked up on a later
          // poll.
          continue;
        }
        if (recvQueue.postedOps_.empty()) {
          // no posted op, let's read it and store as unexpected msg
          CTRAN_LOG_TRACE(
              COLL,
              "CTRAN-SOCKET: received control packet from peer {}, size {}, add to unexpected msg queue",
              peerRanks[fid],
              packet->payloadSize);
          recvQueue.unexpMsgs_.push_back(
              std::make_unique<SocketCtrlPacket>(std::move(*packet)));
        } else {
          auto op = dequeFront(recvQueue.postedOps_);
          FB_COMMCHECK(copyPacketToRecvOp(*packet, *op, peerRanks[fid]));
          op->req.complete();
          CTRAN_LOG_TRACE(
              COLL,
              "CTRAN-SOCKET: received control packet from peer {}, size {}, complete a posted recv",
              peerRanks[fid],
              packet->payloadSize);
        }
      } else if (fds[fid].revents != 0) {
        CTRAN_ERR(
            commInternalError,
            "CTRAN-SOCKET: unexpected poll event {} rank {}",
            fds[fid].revents,
            peerRanks[fid]);
        return commInternalError;
      }
    }
  }
  return commSuccess;
}

commResult_t CtranSocket::checkCtrlPayload(
    const void* payload,
    std::size_t size,
    const char* dir) {
  if ((size > 0 && payload == nullptr) || size > CTRAN_CTRL_MAX_PAYLOAD_SIZE) {
    CTRAN_ERR(
        commInvalidArgument,
        "CTRAN-SOCKET: invalid {} control payload {} size {} (max {})",
        dir,
        payload,
        size,
        CTRAN_CTRL_MAX_PAYLOAD_SIZE);
    return commInvalidArgument;
  }
  return commSuccess;
}

commResult_t CtranSocket::isendCtrlMsgImpl(
    const void* payload,
    std::size_t size,
    int peerRank,
    const SocketServerAddr& peerServerAddr,
    CtranSocketRequest& req) {
  FB_COMMCHECK(checkValidPeer(peerRank));
  FB_COMMCHECK(checkCtrlPayload(payload, size, "send"));
  ctran::bootstrap::Socket* sock = getSocket(peerRank);

  // nullptr socket indicates not yet established connection; try to connect.
  // For smaller peerRank, wait peerRank connects to local listenThread.
  // The ctrlMsg will be enqueued to pendingOps and sent out
  // when polling progress.
  if (rank_ < peerRank && sock == nullptr) {
    FB_COMMCHECK(bootstrapConnect(peerRank, peerServerAddr));
    // Get socket again after connection is established
    sock = getSocket(peerRank);
  }

  CTRAN_LOG_TRACE(
      COLL,
      "CTRAN-SOCKET: isendCtrlMsgImpl to {} socket: {}",
      peerRank,
      (void*)sock);

  if (!addToPendingOpsIfRequired(
          const_cast<void*>(payload),
          size,
          peerRank,
          req,
          SockPendingOp::OpType::ISEND_CTRL,
          sock)) {
    SocketCtrlPacket packet;
    FB_CHECKABORT(
        packet.copyFrom(payload, size),
        "validated Socket control payload must fit in a packet");
    CTRAN_LOG_TRACE(
        COLL, "CTRAN-SOCKET: socket {} send to {}", (void*)sock, peerRank);
    FB_SYSCHECKRETURN(
        sock->send(&packet, packet.wireSize()), commInternalError);
    req.complete();
  }

  return commSuccess;
}

commResult_t CtranSocket::irecvCtrlMsgImpl(
    void* payload,
    std::size_t size,
    int peerRank,
    const SocketServerAddr& peerServerAddr,
    CtranSocketRequest& req) {
  FB_COMMCHECK(checkValidPeer(peerRank));
  FB_COMMCHECK(checkCtrlPayload(payload, size, "receive"));
  ctran::bootstrap::Socket* sock = getSocket(peerRank);

  // nullptr socket indicates not yet established connection; try to connect.
  // For smaller peerRank, wait peerRank connects to local listenThread.
  // The ctrlMsg will be enqueued to pendingOps and sent out when polling
  // progress.
  if (rank_ < peerRank && sock == nullptr) {
    FB_COMMCHECK(bootstrapConnect(peerRank, peerServerAddr));
    // Get socket again after connection is established
    sock = getSocket(peerRank);
  }

  CTRAN_LOG_TRACE(
      COLL,
      "CTRAN-SOCKET: irecvCtrlMsgImpl from {} socket: {}",
      peerRank,
      (void*)sock);

  if (!addToPendingOpsIfRequired(
          payload,
          size,
          peerRank,
          req,
          SockPendingOp::OpType::IRECV_CTRL,
          sock)) {
    auto recvop = std::make_unique<SockPendingOp>(
        SockPendingOp::OpType::IRECV_CTRL, payload, size, peerRank, req);
    postRecvOp(peerRank, std::move(recvop));
  }

  return commSuccess;
}

commResult_t CtranSocket::doRecvMsg(
    ctran::bootstrap::Socket* socket,
    int peerRank,
    std::optional<SocketCtrlPacket>& packet,
    bool* closed) {
  packet.reset();
  *closed = false;
  auto& state = rankToRecvFrameMap_[peerRank];

  // recvAsync returns -1 both for a real failure and for a would-block, so the
  // two are separated on errno here. A would-block just means the rest of the
  // frame has not arrived; state carries the progress to the next poll.
  auto recvStage = [&](void* dst, std::size_t remaining, std::size_t* done) {
    int bytesRead = socket->recvAsync(dst, remaining);
    if (bytesRead > 0) {
      *done += static_cast<std::size_t>(bytesRead);
      return commSuccess;
    }
    if (bytesRead == 0) {
      *closed = true;
      return commSuccess;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
      return commSuccess;
    }
    CTRAN_ERR(
        commInternalError,
        "CTRAN-SOCKET: peer {} socket recvAsync failure, errno {}",
        peerRank,
        strerror(errno));
    return commInternalError;
  };

  auto finishPartialRead = [&]() {
    if (*closed) {
      CTRAN_LOG_SUBSYS(
          WARN,
          COLL,
          "CTRAN-SOCKET: peer {} closed connection, remove socket",
          peerRank);
      rankToRecvFrameMap_.erase(peerRank);
      removeSocket(peerRank);
    }
    return commSuccess;
  };

  if (state.headerBytes < sizeof(state.packet.payloadSize)) {
    auto* header = reinterpret_cast<char*>(&state.packet.payloadSize);
    FB_COMMCHECK(recvStage(
        header + state.headerBytes,
        sizeof(state.packet.payloadSize) - state.headerBytes,
        &state.headerBytes));
    if (*closed || state.headerBytes < sizeof(state.packet.payloadSize)) {
      return finishPartialRead();
    }
    if (state.packet.payloadSize > CTRAN_CTRL_MAX_PAYLOAD_SIZE) {
      CTRAN_ERR(
          commInternalError,
          "CTRAN-SOCKET: peer {} sent oversized control payload {} (max {})",
          peerRank,
          state.packet.payloadSize,
          CTRAN_CTRL_MAX_PAYLOAD_SIZE);
      return commInternalError;
    }
  }

  if (state.payloadBytes < state.packet.payloadSize) {
    FB_COMMCHECK(recvStage(
        state.packet.payload.data() + state.payloadBytes,
        state.packet.payloadSize - state.payloadBytes,
        &state.payloadBytes));
    if (*closed || state.payloadBytes < state.packet.payloadSize) {
      return finishPartialRead();
    }
  }

  packet = state.packet;
  rankToRecvFrameMap_.erase(peerRank);
  return commSuccess;
}

commResult_t CtranSocket::copyPacketToRecvOp(
    const SocketCtrlPacket& packet,
    SockPendingOp& recvOp,
    int peerRank) {
  if (!packet.copyTo(recvOp.payload, recvOp.size)) {
    CTRAN_ERR(
        commInternalError,
        "CTRAN-SOCKET: control payload size mismatch from peer {}: received {}, expected {}",
        peerRank,
        packet.payloadSize,
        recvOp.size);
    return commInternalError;
  }
  return commSuccess;
}

commResult_t CtranSocket::postRecvOp(
    int peerRank,
    std::unique_ptr<SockPendingOp> recvop) {
  auto& recvQueue = getRecvCtrlQueue(peerRank);

  if (recvQueue.unexpMsgs_.empty()) {
    recvQueue.postedOps_.push_back(std::move(recvop));
  } else {
    auto unexpMsg = dequeFront(recvQueue.unexpMsgs_);
    FB_COMMCHECK(copyPacketToRecvOp(*unexpMsg, *recvop, peerRank));
    FB_COMMCHECK(recvop->req.complete());
    CTRAN_LOG_TRACE(
        COLL,
        "CTRAN-SOCKET: received control packet from peer {}, size {}, complete a posted recv",
        peerRank,
        unexpMsg->payloadSize);
  }
  return commSuccess;
}
