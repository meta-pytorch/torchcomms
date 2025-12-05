// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <unordered_map>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/tcpdevmem/CtranTcpDmBase.h"
#include "comms/ctran/bootstrap/Socket.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/tcp_devmem/transport.h"
#include "comms/utils/commSpecs.h"

namespace ctran {

class CtranTcpDm {
 public:
  CtranTcpDm(CtranComm* comm, CtranCtrlManager* ctrlMgr);
  ~CtranTcpDm();

  commResult_t preConnect(const std::unordered_set<int>& peerRanks);

  static commResult_t
  regMem(const void* buf, const size_t len, const int cudaDev, void** handle);

  static commResult_t deregMem(void* handle);

  commResult_t isend(
      int peerRank,
      void* handle,
      void* data,
      size_t size,
      CtranTcpDmRequest& req);

  commResult_t irecv(
      int peerRank,
      void* handle,
      void* data,
      size_t size,
      CtranTcpDmRequest& req,
      int unpackPoolId);

  commResult_t iput(
      const void* sbuf,
      void* dbuf,
      std::size_t len,
      int peerRank,
      void* tcpdmRegElem,
      bool notify,
      CtranTcpDmConfig* config,
      CtranTcpDmRequest* req) {
    return isend(peerRank, tcpdmRegElem, (void*)sbuf, len, *req);
  }

  commResult_t irecvCtrlMsg(
      [[maybe_unused]] ControlMsg& msg,
      [[maybe_unused]] int peerRank,
      CtranTcpDmRequest& req) {
    // Don't share receiver's control information with the sender. Rely
    // on the receiver buffering (TCP window) instead of explicit
    // synchronization.
    req.complete();
    return commSuccess;
  }

  commResult_t isendCtrlMsg(
      [[maybe_unused]] const ControlMsg& msg,
      [[maybe_unused]] int peerRank,
      CtranTcpDmRequest& req) {
    // Don't share receiver's control information with the sender. Rely
    // on the receiver buffering (TCP window) instead of explicit
    // synchronization.
    req.complete();
    return commSuccess;
  }

  // irecv operations can not proceed unless the peer has been connected.
  // When there is no peer, irecv operations are queued and progress()
  // has to be called to make progress on them.
  commResult_t progress();

  // Export the location of GPU kernel consumer queues.
  // Returns the allocated pool index via the out parameter poolIndex.
  commResult_t
  prepareUnpackConsumer(SQueues* sqs, size_t blocks, int* poolIndex = nullptr);

  // Return GPU kernel consumer queues to the pool.
  // poolIndex: the pool index returned by prepareUnpackConsumer.
  commResult_t teardownUnpackConsumer(int poolIndex);

 private:
  std::shared_ptr<::comms::tcp_devmem::Transport> transport_{nullptr};
  ctran::bootstrap::ServerSocket listenSocket_{
      static_cast<int>(NCCL_SOCKET_RETRY_CNT)};
  std::vector<sockaddr_storage> allListenSocketAddrs_{};
  std::thread listenThread_;

  int cudaDev_{-1};
  int rank_{-1};
  int nRanks_{-1};
  uint64_t commHash_{0};
  std::string commDesc_;
  ::comms::tcp_devmem::NetDev* netdev_{nullptr};

  std::mutex mutex_;
  std::unordered_map<int, ::comms::tcp_devmem::Communicator*> recvComms_;
  std::unordered_map<int, ::comms::tcp_devmem::Communicator*> sendComms_;

  struct RecvRequest {
    int peerRank{-1};
    void* handle{nullptr};
    void* data{nullptr};
    size_t size{0};
    CtranTcpDmRequest* req{nullptr};
    int unpackPoolId{-1};
  };
  std::list<std::unique_ptr<RecvRequest>> queuedRecv_;

  commResult_t connectPeer(int peerRank);

  void bootstrapPrepare(ctran::bootstrap::IBootstrap* bootstrap);
  void bootstrapAddRecvPeer(
      int peerRank,
      ::comms::tcp_devmem::Communicator* comm);
  void bootstrapAccept();
  void bootstrapAddSendPeer(
      int peerRank,
      ::comms::tcp_devmem::Communicator* comm);
  commResult_t bootstrapConnect(
      int peerRank,
      const folly::SocketAddress& peerSockAddr);

  commResult_t irecvConnected(
      int peerRank,
      void* handle,
      void* data,
      size_t size,
      CtranTcpDmRequest& req,
      int unpackPoolId);
};

} // namespace ctran
