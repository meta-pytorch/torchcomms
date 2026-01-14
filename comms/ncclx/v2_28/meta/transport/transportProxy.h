// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

#include <folly/MapUtil.h>

#include "comm.h"
#include "nccl_common.h"

#include "comms/utils/commSpecs.h"

namespace ncclx::transport {

constexpr size_t kDefaultSyncPoolSize = 1 << 21; // 2MiB

enum class TransportRequestType : int {
  UNSET = 0,
  PRECONNECT_P2P,
  PRECONNECT_COLL,
  PREP_RESOURCES,
  TERNIMATE,
};

struct TransportRequest {
  TransportRequestType type{TransportRequestType::UNSET};
  ncclComm* comm{nullptr};
  uint64_t channelMask{0};
  uint64_t opCount{0};
  std::shared_ptr<void> peerReconnInfoMap{nullptr};
  std::vector<std::string> bufKeys;
  uint64_t* channelsReadyPtr{nullptr};
  commResult_t state{commInProgress};
  // Following fields are only used for p2p/coll preconnect
  std::array<bool, NCCL_NUM_ALGORITHMS> algoNeedConnect{false};

  // For TERNIMATE request
  TransportRequest(TransportRequestType type) : type(type) {}
  // For PREP_RESOURCES request
  TransportRequest(
      TransportRequestType type,
      ncclComm* comm,
      uint64_t channelMask,
      uint64_t opCount,
      std::shared_ptr<void> peerReconnInfoMap,
      uint64_t* channelsReadyPtr,
      commResult_t initState)
      : type(type),
        comm(comm),
        channelMask(channelMask),
        opCount(opCount),
        peerReconnInfoMap(std::move(peerReconnInfoMap)),
        channelsReadyPtr(channelsReadyPtr),
        state(initState) {}

  // For PRECONNECT_P2P request
  TransportRequest(TransportRequestType type, ncclComm* comm)
      : type(type), comm(comm) {}
  // For PRECONNECT_COLL request
  TransportRequest(
      TransportRequestType type,
      ncclComm* comm,
      bool* algoNeedConnect)
      : type(type), comm(comm) {
    for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
      this->algoNeedConnect[i] = algoNeedConnect[i];
    }
  }
};

inline bool useTransportProxy(ncclComm* comm) {
  return (comm && comm->transportProxy_);
}

class TransportProxy {
 public:
  TransportProxy(ncclComm* comm);
  ~TransportProxy();

  // Get the next available sync flag to synchronize with NCCL kernels
  commResult_t getNextChannelsReadyPtr(uint64_t** channelsReadyPtr);
  // Enqueue a request to the transport proxy for preparing transport resources,
  // if needed, for the current kernel plan.
  //  @param[IN] comm: communicator, the ncclComm struct
  //  @param[IN] channelMask: the bit mask of channels to prepare resources for
  //  @param[IN] peerReconnInfoMap: the map of peer ranks to their transport
  //  connection state
  //  @param[IN] channelsReadyPtr: pointer to the channels ready flag to
  //  synchronize
  commResult_t enqueuePrepRequest(
      ncclComm* comm,
      uint64_t channelMask,
      std::shared_ptr<void> peerReconnInfoMap,
      uint64_t* channelsReadyPtr);
  // Enqueue a request to the transport proxy for pre-connecting to p2p peers
  //  @param[IN] comm: communicator, the ncclComm struct
  commResult_t enqueueP2pPreconnect(ncclComm* comm);
  // Enqueue a request to the transport proxy for pre-connecting to collective
  // peers
  //  @param[IN] comm: communicator, the ncclComm struct
  //  @param[IN] algoNeedConnect: indicate which algorithm needs to be connected
  commResult_t enqueueCollPreconnect(ncclComm* comm, bool* algoNeedConnect);
  commResult_t waitPreconnect();
  void workerThreadFn();
  void shutdown();
  void waitAll();
  void progress();

 private:
  // test to see any request is finished and garbage collect the resource, i.e.,
  // sync flag
  void testAny();
  void prepResources(std::shared_ptr<TransportRequest> req);
  // check if we can skip the prepResources call
  bool canSkipPrepBufs(ncclComm* comm, uint64_t opCount);
  inline uint64_t incrOpCount(uint64_t commHash) {
    auto opCountPtr = folly::get_ptr(opCountMap_, commHash);
    if (opCountPtr) {
      (*opCountPtr)++;
    } else {
      opCountMap_[commHash] = 0;
    }
    return folly::get_default(opCountMap_, commHash);
  }

  // request queue for worker thread to process
  std::deque<std::shared_ptr<TransportRequest>> reqQueue_;
  // tracking the active operation to perform garbage collection once they are
  // done
  std::deque<std::shared_ptr<TransportRequest>> activeOps_;
  // Track the pre-connect requests separately
  std::deque<std::shared_ptr<TransportRequest>> preconnReqQueue_;
  std::condition_variable cv_;
  std::condition_variable preconnCv_;
  mutable std::mutex mutex_;
  bool initialized_{false};
  std::thread workerThread_;
  ncclComm* parentComm_{nullptr};

  // maintain a pool of sync flags for worker thread to sync with NCCL kernels
  uint64_t* syncPoolPtr_{nullptr};
  std::deque<uint64_t*> syncFlagPool_;
  // use a map to record opCount of each communicator hash
  std::unordered_map<uint64_t, uint64_t> opCountMap_;
};

commResult_t tranportProxyInit(struct ncclComm* comm, struct ncclComm* parent);
commResult_t tranportProxyShutdown(struct ncclComm* comm);

} // namespace ncclx::transport
