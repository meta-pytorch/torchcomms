// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <fmt/core.h>
#include <folly/SocketAddress.h>
#include <folly/Synchronized.h>
#include <memory>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/socket/CtranSocketBase.h"
#include "comms/ctran/bootstrap/Socket.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/commSpecs.h"

/**
 * CtranSocket class to be used by algorithms and ctranMapper.
 */
class CtranSocket {
 public:
  /**
   * Creates local socket resources for a given communicator.
   * It also launches a listen thread to accept remote connection.
   * The remote connection will happen when the remote peer
   * issues the first message to the local rank. Input arguments:
   *    - comm: the NCCL communicator
   *    - ctrlMgr: the ctranCtrlManager that manages control message callback
   *               functions registered by other modules. Passed to VC for
   *               calling callback when receiving control message.
   */
  CtranSocket(CtranComm* comm, CtranCtrlManager* ctrlMgr);

  CtranSocket(
      int rank,
      int cudaDev,
      uint64_t commHash,
      const std::string& commDesc,
      CtranCtrlManager* ctrlMgr,
      const SocketServerAddr& serverAddr);
  ~CtranSocket();

  commResult_t preConnect(const std::unordered_set<int>& peerRanks);

  // Finish all pending ops
  inline commResult_t progress(void) {
    return progressInternal();
  }

  // Send control message packet over the Socket connection.
  // Only call this function when the peer's ListenSocket address is available
  inline commResult_t
  isendCtrlMsg(const ControlMsg& msg, int peerRank, CtranSocketRequest& req) {
    return isendCtrlMsg(msg, peerRank, SocketServerAddr(), req);
  }

  // Send control message packet to a remote address over the Socket
  // Input arguments:
  //   - msg: the control message to be sent
  //   - peerRank: the rank to send the control message to
  //   - peerServerAddr: the ip/port/hostname info of the server to connect to
  // Output arguments:
  //   - req: the request object to track the progress of the control message
  inline commResult_t isendCtrlMsg(
      const ControlMsg& msg,
      int peerRank,
      const SocketServerAddr& peerServerAddr,
      CtranSocketRequest& req) {
    return isendCtrlMsgImpl(msg, peerRank, peerServerAddr, req);
  }

  // Receive control message packet over the Socket connection.
  // Only call this function when the peer's ListenSocket address is available
  inline commResult_t
  irecvCtrlMsg(ControlMsg& msg, int peerRank, CtranSocketRequest& req) {
    return irecvCtrlMsg(msg, peerRank, SocketServerAddr(), req);
  }

  // Receive control message packet from a remote address over the Socket
  // Input arguments:
  //   - msg: the control message to be received
  //   - peerRank: the rank to receive the control message from
  //   - peerServerAddr: the ip/port/hostname info of the remote server
  // Output arguments:
  //   - req: the request object to track the progress of the control message
  inline commResult_t irecvCtrlMsg(
      ControlMsg& msg,
      int peerRank,
      const SocketServerAddr& peerServerAddr,
      CtranSocketRequest& req) {
    return irecvCtrlMsgImpl(msg, peerRank, peerServerAddr, req);
  }

  CtranComm* comm{nullptr};

 private:
  struct recvCtrlQueue {
    std::deque<std::unique_ptr<SockPendingOp>> postedOps_;
    std::deque<std::unique_ptr<ControlMsg>> unexpMsgs_;
  };

  void init(const SocketServerAddr& serverAddr);
  void bootstrapAccept();
  commResult_t bootstrapConnect(int peerRank, const SocketServerAddr& peerAddr);
  commResult_t bootstrapConnect(
      int peerRank,
      const folly::SocketAddress& peerAddr);
  commResult_t updateSocket(
      std::unique_ptr<ctran::bootstrap::Socket> sock,
      int rank);

  inline commResult_t checkValidPeer(int peerRank) {
    if (peerRank < 0 || (comm && peerRank >= comm->statex_->nRanks())) {
      CLOGF(
          ERR,
          "invalid peerRank ({}) < 0 or >= nRanks {}",
          peerRank,
          comm ? comm->statex_->nRanks() : -1);
      return commInternalError;
    }
    return commSuccess;
  }

  // Check whether vc is ready and no outstanding pending ops
  bool addToPendingOpsIfRequired(
      const ControlMsg& msg,
      int peerRank,
      CtranSocketRequest& req,
      SockPendingOp::OpType opType,
      ctran::bootstrap::Socket* sock);

  commResult_t progressPendingOps(void);

  commResult_t progressInternal();

  commResult_t isendCtrlMsgImpl(
      const ControlMsg& msg,
      int peerRank,
      const SocketServerAddr& peerServerAddr,
      CtranSocketRequest& req);

  commResult_t irecvCtrlMsgImpl(
      ControlMsg& msg,
      int peerRank,
      const SocketServerAddr& peerServerAddr,
      CtranSocketRequest& req);

  int doRecvMsg(
      ctran::bootstrap::Socket* socket,
      int peerRank,
      ControlMsg* msg);

  commResult_t postRecvOp(int peerRank, std::unique_ptr<SockPendingOp> recvop);

  // Get the socket for a given peer.
  // If the peer is not yet connected, return nullptr.
  // For a returned socket, it is guaranteed to be ready to use.
  inline ctran::bootstrap::Socket* getSocket(int peerRank) {
    auto locked = socketMaps_.rlock();
    auto it = locked->rankToSocket.find(peerRank);
    if (it == locked->rankToSocket.end()) {
      return nullptr;
    }
    return it->second.get();
  }

  inline void removeSocket(int peerRank) {
    auto locked = socketMaps_.wlock();
    locked->rankToSocket[peerRank]->close();
    locked->rankToSocket.erase(peerRank);
  }

  inline recvCtrlQueue& getRecvCtrlQueue(int peerRank) {
    auto it = rankToRecvCtrlMap_.find(peerRank);
    if (it == rankToRecvCtrlMap_.end()) {
      rankToRecvCtrlMap_[peerRank] = recvCtrlQueue();
    }
    auto& recvQueue = rankToRecvCtrlMap_[peerRank];
    return recvQueue;
  }

  const int rank_;
  const int cudaDev_;
  const uint64_t commHash_;
  const std::string commDesc_;
  CtranCtrlManager* ctrlMgr_{nullptr};
  CommLogData ncclLogData_;
  std::vector<bool> preConnectPeerMap_;

  ctran::bootstrap::ServerSocket listenSocket_{
      static_cast<int>(NCCL_SOCKET_RETRY_CNT)};
  std::vector<sockaddr_storage> allListenSocketAddrs_{};
  std::thread listenThread_;

  struct SocketMaps {
    folly::F14FastMap<int, std::unique_ptr<ctran::bootstrap::Socket>>
        rankToSocket;
  };
  // the rankToSocket is accessed by both the main thread and the listen thread.
  folly::Synchronized<SocketMaps> socketMaps_;

  folly::Synchronized<
      folly::F14FastMap<int, std::deque<std::unique_ptr<SockPendingOp>>>>
      rankToPendingOpsMap_;

  // every rank maintains a postedrecv queue and unexpected msg queue
  folly::F14FastMap<int, recvCtrlQueue> rankToRecvCtrlMap_;
};
