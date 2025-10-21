// Copyright (c) Meta Platforms, Inc. and affiliates.

// TODO: Migrate to comms/ctran/utils/Alloc.h once we implement
// "ncclCuMemHostAlloc" equivalent
#include "alloc.h"

#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/Utils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/trainer/TrainerContext.h"
#include "meta/transport/transportConnect.h"
#include "meta/transport/transportExt.h"
#include "meta/transport/transportProxy.h"
#include "meta/wrapper/MetaFactory.h"

namespace ncclx::transport {

namespace {
inline enum NCCL_USE_TRANSPORT_PROXY getTransportProxyMode() {
  auto val = NCCL_USE_TRANSPORT_PROXY;
  if (NCCL_USE_TRANSPORT_PROXY == NCCL_USE_TRANSPORT_PROXY::unset) {
    val = (NCCL_USE_MEM_CACHE) ? NCCL_USE_TRANSPORT_PROXY::shared
                               : NCCL_USE_TRANSPORT_PROXY::none;
  }
  CLOGF_SUBSYS(INFO, ENV, "NCCL_USE_TRANSPORT_PROXY={}", static_cast<int>(val));
  return val;
}
}; // namespace

commResult_t tranportProxyInit(struct ncclComm* comm, struct ncclComm* parent) {
  auto mode = getTransportProxyMode();

  if (mode == NCCL_USE_TRANSPORT_PROXY::none) {
    return commSuccess;
  } else if (
      mode != NCCL_USE_TRANSPORT_PROXY::none &&
      ncclx::getChannelMetadataLoc() ==
          NCCL_CHANNEL_METADATA_LOCATION::device) {
    FB_ERRORRETURN(
        commInternalError,
        "NCCL_USE_TRANSPORT_PROXY is not supported with NCCL_CHANNEL_METADATA_LOCATION=device");
  }

  if (mode == NCCL_USE_TRANSPORT_PROXY::shared && parent != nullptr) {
    comm->transportProxy_ = parent->transportProxy_;
  } else {
    comm->transportProxy_ = std::make_shared<TransportProxy>(comm);
  }
  return commSuccess;
}

commResult_t tranportProxyShutdown(struct ncclComm* comm) {
  auto mode = getTransportProxyMode();
  if (mode != NCCL_USE_TRANSPORT_PROXY::none) {
    comm->transportProxy_.reset();
  }

  return commSuccess;
}

TransportProxy::TransportProxy(struct ncclComm* comm) : parentComm_(comm) {
  NCCLCHECKIGNORE(
      ncclCuMemHostAlloc((void**)&syncPoolPtr_, nullptr, kDefaultSyncPoolSize));
  size_t nFlags = kDefaultSyncPoolSize / sizeof(uint64_t);
  for (int elemOffset = 0; elemOffset < nFlags; elemOffset++) {
    syncFlagPool_.push_back(syncPoolPtr_ + elemOffset);
  }
  if (getTransportProxyMode() != NCCL_USE_TRANSPORT_PROXY::none) {
    workerThread_ = std::thread(&TransportProxy::workerThreadFn, this);
  }
  initialized_ = true;
}

TransportProxy::~TransportProxy() {
  if (initialized_) {
    shutdown();
  }
}

void TransportProxy::shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    reqQueue_.push_back(
        std::make_shared<TransportRequest>(TransportRequestType::TERNIMATE));
    cv_.notify_one();
  }
  if (getTransportProxyMode() != NCCL_USE_TRANSPORT_PROXY::none) {
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "Shutting down NCCLX transport worker: join worker thread...");
    workerThread_.join();
  }
  while (!activeOps_.empty()) {
    testAny();
  }
  // free sync flag pool
  if (syncPoolPtr_) {
    activeOps_.clear();
    syncFlagPool_.clear();
    NCCLCHECKIGNORE(ncclCuMemHostFree((void*)syncPoolPtr_));
  }
  initialized_ = false;
}

inline bool TransportProxy::canSkipPrepBufs(ncclComm* comm, uint64_t opCount) {
  // fast path without the need of reconnection
  bool skipReconnect =
      (NCCL_TRANSPORT_PREP_TRAINER_ITERATION_LIMIT != 0 &&
       ncclxGetIteration() > NCCL_TRANSPORT_PREP_TRAINER_ITERATION_LIMIT) ||
      (NCCL_TRANSPORT_RECONNECT_OPCOUNT_LIMIT != 0 &&
       opCount > NCCL_TRANSPORT_RECONNECT_OPCOUNT_LIMIT);

  // Early return if buffers are not shared, or we reach a pre-defined limit to
  // skip exchanging info
  if (!NCCL_USE_SHARED_BUFFER_POOL || skipReconnect) {
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "Rank-{} skip re-connect check: NCCL_TRANSPORT_PREP_TRAINER_ITERATION_LIMIT={} and ncclxGetIteration={} "
        " NCCL_TRANSPORT_RECONNECT_OPCOUNT_LIMIT={} vs. opCount={}",
        comm->rank,
        NCCL_TRANSPORT_PREP_TRAINER_ITERATION_LIMIT,
        ncclxGetIteration(),
        NCCL_TRANSPORT_RECONNECT_OPCOUNT_LIMIT,
        opCount);
    return true;
  }

  return false;
}

commResult_t TransportProxy::getNextChannelsReadyPtr(
    uint64_t** channelsReadyPtr) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (syncFlagPool_.empty()) {
    FB_ERRORTHROW(
        commInternalError, "No available sync flag in transport worker thread");
  }
  auto ptr = syncFlagPool_.front();
  syncFlagPool_.pop_front();

  *channelsReadyPtr = ptr;

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "Transport proxy thread: get next sync flag pointer {:x}",
      (uintptr_t)ptr);

  return commSuccess;
}

commResult_t TransportProxy::enqueuePrepRequest(
    ncclComm* comm,
    uint64_t channelMask,
    std::shared_ptr<void> peerReconnInfoMap,
    uint64_t* channelsReadyPtr) {
  std::unique_lock<std::mutex> lock(mutex_);

  auto opCount = incrOpCount(comm->commHash);

  auto req = std::make_shared<TransportRequest>(
      TransportRequestType::PREP_RESOURCES,
      comm,
      channelMask,
      opCount,
      peerReconnInfoMap,
      channelsReadyPtr,
      (canSkipPrepBufs(comm, opCount)) ? commSuccess : commInProgress);

  reqQueue_.push_back(req);
  cv_.notify_one();

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "{}: Enqueued request to prepare resources for current kernel plan: "
      "opCount={} (comm->opCount={}),channelMask={:x}, channelsReadyPtr={}({:#x})",
      comm->config.commDesc,
      opCount,
      comm->opCount,
      channelMask,
      *channelsReadyPtr,
      (uintptr_t)channelsReadyPtr);

  return commSuccess;
}

commResult_t TransportProxy::enqueueP2pPreconnect(ncclComm* comm) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto req = std::make_shared<TransportRequest>(
      TransportRequestType::PRECONNECT_P2P, comm);

  reqQueue_.push_back(req);
  cv_.notify_one();
  preconnReqQueue_.push_back(req);

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "Enqueue p2p preconnect request ({} in queue) for comm {:x}",
      preconnReqQueue_.size(),
      (uintptr_t)comm);

  return commSuccess;
}

commResult_t TransportProxy::enqueueCollPreconnect(
    ncclComm* comm,
    bool* algoNeedConnect) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto req = std::make_shared<TransportRequest>(
      TransportRequestType::PRECONNECT_COLL, comm, algoNeedConnect);
  reqQueue_.push_back(req);
  cv_.notify_one();

  preconnReqQueue_.push_back(req);

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "Enqueue coll preconnect request ({} in queue) for comm {:x}",
      preconnReqQueue_.size(),
      (uintptr_t)comm);

  return commSuccess;
}

commResult_t TransportProxy::waitPreconnect() {
  if (preconnReqQueue_.empty()) {
    return commSuccess;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  for (const auto& req : preconnReqQueue_) {
    preconnCv_.wait(lock, [&] { return req->state != commInProgress; });
    FB_COMMCHECK(req->state);
  }
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "Waited for {} preconnect requests to complete",
      preconnReqQueue_.size());
  preconnReqQueue_.clear();
  return commSuccess;
}

void TransportProxy::testAny() {
  // Once collective kernel complete the work and reset the flag, release the
  // buffers and move the flag back to the pool to reuse.
  for (const auto& req : activeOps_) {
    auto ptr = req->channelsReadyPtr;
    if (*ptr == 0) {
      FB_COMMCHECKTHROW(req->comm->memCache->release(req->bufKeys));
      CLOGF_SUBSYS(
          INFO,
          COLL,
          "Releasing {} bufKeys for comm {}",
          req->bufKeys.size(),
          ctran::utils::parseCommDesc(req->comm->config.commDesc));
      syncFlagPool_.push_back(ptr);
      req->state = commSuccess;
      CLOGF_SUBSYS(
          INFO,
          COLL,
          "Garbage collect sync flag pointer {:x}, {} collectives in progress",
          (uintptr_t)ptr,
          activeOps_.size());
    }
  }
  // remove any completed requests
  activeOps_.erase(
      std::remove_if(
          activeOps_.begin(),
          activeOps_.end(),
          [](std::shared_ptr<TransportRequest> req) {
            return req->state == commSuccess;
          }),
      activeOps_.end());
}

void TransportProxy::waitAll() {
  while (true) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (activeOps_.empty()) {
      break;
    }
    testAny();
  }
}

void TransportProxy::progress() {
  std::lock_guard<std::mutex> lock(mutex_);
  testAny();
}

void TransportProxy::prepResources(std::shared_ptr<TransportRequest> req) {
  bool skipReconnect = req->state == commSuccess;
  req->state = ncclToMetaComm(ncclx::transportReConnect(
      req->comm,
      req->opCount,
      req->peerReconnInfoMap,
      req->bufKeys,
      skipReconnect));
  if (req->state != commSuccess) {
    FB_ERRORTHROW(
        commInternalError,
        "Failed to reconnect to peers in transport worker thread");
  }
  // Update the mask to signal the NCCL kernel that transport resources are
  // ready to start collectives
  *req->channelsReadyPtr = req->channelMask;

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "{}: Transport is ready for reqCount={}, req->channelMask={:x}, req->channelsReadyPtr={:x} ({:#x})",
      req->comm->config.commDesc,
      req->opCount,
      req->channelMask,
      *req->channelsReadyPtr,
      (uintptr_t)req->channelsReadyPtr);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // reset the state to indicate in-progress of collective kernel
    req->state = commInProgress;
    activeOps_.push_back(req);
  }
}

void TransportProxy::workerThreadFn() {
  NCCL_NAMED_THREAD_START_EXT(
      "TransportProxy",
      parentComm_->rank,
      parentComm_->commHash,
      parentComm_->logMetaData.commDesc);

  FB_CUDACHECKTHROW(cudaSetDevice(parentComm_->cudaDev));

  while (true) {
    // test if any collective is complected, and release the resources
    // until any new requests are enqueued
    bool keepTest = true;
    while (keepTest) {
      std::lock_guard<std::mutex> lock(mutex_);
      testAny();
      keepTest = reqQueue_.empty() && !activeOps_.empty();
    }
    // work on new requests
    std::shared_ptr<TransportRequest> req;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return !reqQueue_.empty(); });
      req = reqQueue_.front();
      reqQueue_.pop_front();
    }

    if (LIKELY(req->type == TransportRequestType::PREP_RESOURCES)) {
      prepResources(req);
    } else if (req->type == TransportRequestType::PRECONNECT_P2P) {
      while (req->comm->initState != ncclSuccess) {
        std::this_thread::yield();
      }
      auto res = ncclx::p2pPreconnect(req->comm);
      std::lock_guard<std::mutex> lock(mutex_);
      req->state = ncclToMetaComm(res);
      preconnCv_.notify_one();
    } else if (req->type == TransportRequestType::PRECONNECT_COLL) {
      while (req->comm->initState != ncclSuccess) {
        std::this_thread::yield();
      }
      auto res = ncclx::collPreconnect(req->comm, req->algoNeedConnect);
      std::lock_guard<std::mutex> lock(mutex_);
      req->state = ncclToMetaComm(res);
      preconnCv_.notify_one();
    } else if (req->type == TransportRequestType::TERNIMATE) {
      break;
    } else {
      CLOGF(
          ERR,
          "Unknown request type {} in transport worker thread",
          static_cast<int>(req->type));
    }
  }
}
}; // namespace ncclx::transport
