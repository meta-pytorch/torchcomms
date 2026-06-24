// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <atomic>
#include <deque>
#include <list>
#include <memory>
#include <unordered_map>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/tcpdevmem/CtranTcpDmBase.h"
#include "comms/ctran/bootstrap/Socket.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/tcp_devmem/transport.h"
#include "comms/utils/commSpecs.h"

namespace ctran {

class Profiler;

class CtranTcpDm {
 public:
  // `profiler` may be null when the comm has profiling disabled. When
  // non-null, the transport registers its ALGO_TOTAL start/end hooks
  // directly in the constructor.
  explicit CtranTcpDm(CtranComm* comm, ctran::Profiler* profiler = nullptr);
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
      void* unpackPool);

  // Counter-based irecv: request is managed internally, completion
  // increments a per-peer counter checked by checkNotify.
  commResult_t irecvCounted(
      int peerRank,
      void* handle,
      void* data,
      size_t size,
      void* unpackPool);

  // Check if a data recv has completed for peerRank (counter-based,
  // analogous to IB's notifyCount_ approach).
  commResult_t checkNotify(int peerRank, bool* done);

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

  commResult_t
  irecvCtrlMsg(ControlMsg& msg, int peerRank, CtranTcpDmRequest& req);

  commResult_t
  isendCtrlMsg(const ControlMsg& msg, int peerRank, CtranTcpDmRequest& req);

  void profilerStart();
  void profilerEnd();

  // irecv operations can not proceed unless the peer has been connected.
  // When there is no peer, irecv operations are queued and progress()
  // has to be called to make progress on them.
  commResult_t progress();

  void cancelQueuedRecv(CtranTcpDmRequest* req);

  void abortOutstanding(const char* reason);

  // Export the location of GPU kernel consumer queues.
  // Returns the allocated pool via the out parameter pool.
  commResult_t
  prepareUnpackConsumer(SQueues* sqs, size_t blocks, void** pool = nullptr);

  // Return GPU kernel consumer queues to the pool.
  // pool: the pool returned by prepareUnpackConsumer.
  commResult_t teardownUnpackConsumer(void* pool);

 private:
  // Called from the constructor when a non-null Profiler is supplied;
  // attaches ALGO_TOTAL start/end hooks. Kept private now that the
  // mapper no longer needs to drive registration post-construction.
  void registerProfilerHooks(ctran::Profiler* profiler);

  ::comms::tcp_devmem::ProfilerContext profilerCtx_{};

  // The Transport singleton is fetched via CtranTcpDmSingleton::getTransport()
  // at each use site rather than held as a shared_ptr member. Holding it as a
  // member kept the singleton ref-count elevated for the lifetime of every
  // CtranTcpDm and prevented folly::Singleton's destroyInstances from running
  // cleanly at process exit (FATAL "living references"). Mirrors what
  // CtranIb does with CtranIbSingleton::getInstance().
  ctran::bootstrap::ServerSocket listenSocket_{
      static_cast<int>(NCCL_SOCKET_RETRY_CNT)};
  std::vector<sockaddr_storage> allListenSocketAddrs_{};
  std::thread listenThread_;

  CtranComm* comm_{nullptr};
  int cudaDev_{-1};
  int rank_{-1};
  int nRanks_{-1};
  uint64_t commHash_{0};
  std::string commDesc_;
  ::comms::tcp_devmem::NetDevInterface* netdev_{nullptr};
  std::atomic<bool> aborted_{false};

  std::mutex mutex_;
  std::unordered_map<int, ::comms::tcp_devmem::CommunicatorInterface*>
      recvComms_;
  std::unordered_map<int, ::comms::tcp_devmem::CommunicatorInterface*>
      sendComms_;

  struct RecvRequest {
    int peerRank{-1};
    void* handle{nullptr};
    void* data{nullptr};
    size_t size{0};
    CtranTcpDmRequest* req{nullptr};
    void* unpackPool{nullptr};
  };
  std::list<std::unique_ptr<RecvRequest>> queuedRecv_;

  struct CtrlRecvRequest {
    int peerRank{-1};
    std::shared_ptr<std::array<uint8_t, 1>> storage;
    CtranTcpDmRequest* req{nullptr};
  };
  std::list<std::unique_ptr<CtrlRecvRequest>> queuedCtrlRecv_;

  // Counter-based recv notification (analogous to IB's notifyCount_).
  // Internally-owned requests for irecvCounted; completed in FIFO order.
  std::unordered_map<int, std::deque<std::unique_ptr<CtranTcpDmRequest>>>
      pendingRecvNotifies_;
  // Per-peer count of completed data recvs, decremented by checkNotify.
  std::unordered_map<int, int> recvNotifyCount_;

  void recvNotifyProgress();
  void ctrlRecvProgress();

  commResult_t connectPeer(int peerRank);
  void closeComms(const char* reason, uint32_t closeFlags);

  void bootstrapPrepare(meta::comms::IBootstrap* bootstrap);
  void bootstrapAddRecvPeer(
      int peerRank,
      ::comms::tcp_devmem::CommunicatorInterface* comm);
  void bootstrapAccept();
  void bootstrapAddSendPeer(
      int peerRank,
      ::comms::tcp_devmem::CommunicatorInterface* comm);
  commResult_t bootstrapConnect(
      int peerRank,
      const folly::SocketAddress& peerSockAddr);

  commResult_t irecvConnected(
      int peerRank,
      void* handle,
      void* data,
      size_t size,
      CtranTcpDmRequest& req,
      void* unpackPool);

  commResult_t irecvCtrlMsgConnected(
      int peerRank,
      std::shared_ptr<std::array<uint8_t, 1>> storage,
      CtranTcpDmRequest& req);
};

} // namespace ctran
