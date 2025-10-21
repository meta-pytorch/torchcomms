// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/profiler/CtranProfiler.h"
#include "comms/ctran/commstate/CommStateX.h"
#include "comms/ctran/profiler/AlgoProfilerModule.h"
#include "comms/ctran/profiler/CtranProfilerSlowRankModule.h"
#include "comms/ctran/profiler/QueuePairProfilerModule.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

QueuePair::QueuePair() {
  sendQueue = std::deque<Wqe>();
  bytesInFlight = 0;
}

CtranProfiler::CtranProfiler(CtranComm* comm) : comm_(comm) {
  // install modules if any
  CLOGF_SUBSYS(INFO, INIT, "Initializing Ctran Profiler with comm");
  const auto statex = comm_->statex_.get();
  localRankInfo_.localRank = statex->localRank();
  localRankInfo_.globalRank = statex->rank();
  localRankInfo_.hostName = statex->host();
  CtranProfilerInit();
}

CtranProfiler::CtranProfiler(int rank, const std::string& hostname) {
  // install modules if any
  CLOGF_SUBSYS(INFO, INIT, "Initializing Ctran Profiler without comm");
  localRankInfo_.globalRank = rank;
  localRankInfo_.hostName = hostname;
  CtranProfilerInit();
}

void CtranProfiler::CtranProfilerInit() {
  if (NCCL_SLOW_RANK_ENABLE) {
    installModule<CtranProfilerSlowRankModule>();
  }
  if (NCCL_CTRAN_ALGO_PROFILING_ENABLE) {
    if (!NCCL_FILTER_ALGO_LOGGING_BY_RANKS.empty()) {
      // TODO: use commstatex when `commRanksToWorldRanks` map is available
      for (const auto& rank : NCCL_FILTER_ALGO_LOGGING_BY_RANKS) {
        if (std::stoi(rank) == localRankInfo_.globalRank) {
          installModule<AlgoProfilerModule>();
          break;
        }
      }
    } else {
      installModule<AlgoProfilerModule>();
    }

    if (NCCL_CTRAN_ALGO_PROFILING_SAMPLING_MODE == "collective") {
      profileModuleLoggingConfig_.mt.seed(comm_->statex_->commHash());
    }
  }
  if (NCCL_CTRAN_QP_PROFILING_ENABLE) {
    installModule<QueuePairProfilerModule>();
  }
}

CtranProfiler::~CtranProfiler() {}

/*
 * For every CtranTransportEvent, the profiler does the following:
 * 1. updates its state, if any (e.g., statistics on wqes, queuepairs, data
 * transfers)
 * 2. forwards the event to each installed module that is interested in it
 */
void CtranProfiler::handleTransportEvent(const CtranTransportEvent& event) {
  switch (event.type) {
    case CtranTransportEvent::Type::IRECVCTRL_ISSUED: {
      for (const auto& module : modules_) {
        module->onReadyToSend(event);
      }
      break;
    }
    case CtranTransportEvent::Type::ISENDCTRL_ISSUED: {
      for (const auto& module : modules_) {
        module->onReadyToReceive(event);
      }
      break;
    }
    case CtranTransportEvent::Type::IRECVCTRL_COMPLETE: {
      for (const auto& module : modules_) {
        module->onCtrlReceived(event);
      }
      break;
    }
    case CtranTransportEvent::Type::ISENDCTRL_COMPLETE: {
      for (const auto& module : modules_) {
        module->onCtrlComplete(event);
      }
      break;
    }
    case CtranTransportEvent::Type::PUT_ISSUED: {
      for (const auto& module : modules_) {
        module->onPutIssued(event);
      }
      break;
    }
    case CtranTransportEvent::Type::PUT_COMPLETE: {
      for (const auto& module : modules_) {
        module->onPutComplete(event);
      }
      break;
    }
    case CtranTransportEvent::Type::RECV_STARTED: {
      for (const auto& module : modules_) {
        module->onRecvStarted(event);
      }
      break;
    }
    case CtranTransportEvent::Type::RECV_COMPLETE: {
      for (const auto& module : modules_) {
        module->onRecvComplete(event);
      }
      break;
    }
    default:
      CLOGF(WARN, "CTranProfiler: Unknown transport event type");
  }
}

void CtranProfiler::handleRegistrationEvent(
    const CtranRegistrationEvent& event) {
  switch (event.type) {
    case CtranRegistrationEvent::Type::BUFFER_REGISTRATION_START: {
      for (const auto& module : modules_) {
        module->onBufferRegistrationStart(event);
      }
      break;
    }
    case CtranRegistrationEvent::Type::BUFFER_REGISTRATION_COMPLETE: {
      for (const auto& module : modules_) {
        module->onBufferRegistrationComplete(event);
      }
      break;
    }
    default:
      CLOGF(WARN, "CTranProfiler: Unknown buffer event type");
  }
}

/*
 * For every RDMA transport event (post or complete), the profiler does the
 * following:
 * 1. updates its state, if any; for example, it stores information about
 * pending requests until their completion
 * 2. forwards the event to each installed module that is interested in it
 */
void CtranProfiler::handleRdmaEvent(const CtranRdmaEvent& event) {
  switch (event.type) {
    case CtranRdmaEvent::Type::WR_POSTED: {
      // We consider only write events for now
      // TODO(lume): add recv requests
      if (event.op != CtranRdmaEvent::Operation::SEND) {
        CLOGF(WARN, "CTranProfiler: Unsupported RDMA operation");
        break;
      }

      auto& queue = pendingWqes_[event.queuePair].sendQueue;

      // If the wqe queue reached its limit, we remove the oldest entry.
      // For now we use a define statement to set the limit of a wqe queue. In
      // the future, we will use `ibv_query_qp` to get the max outstanding WR
      // for each queue pair.
      if (queue.size() == MAX_PENDING_WQES) {
        queue.pop_front();
      }

      // the time in microseconds from when the previous WR on the same queue
      // pair was posted if there is no previous WR, the value is 0
      auto timeFromPreviousWRPostUs = [&]() -> std::chrono::microseconds {
        if (pendingWqes_[event.queuePair].lastPostTs.time_since_epoch() ==
            std::chrono::microseconds::zero()) {
          return std::chrono::microseconds::zero();
        } else {
          return std::chrono::duration_cast<std::chrono::microseconds>(
              event.timestamp - pendingWqes_[event.queuePair].lastPostTs);
        }
      }();

      // the time in microseconds from when the previous WR on the same queue
      // pair was completed if there is no previous WR or the previous WR is
      // still outstanding, the value is 0
      auto timeFromPreviousWRCompletionUs = [&]() -> std::chrono::microseconds {
        if (pendingWqes_[event.queuePair].bytesInFlight > 0 ||
            pendingWqes_[event.queuePair].lastCompletionTs.time_since_epoch() ==
                std::chrono::microseconds::zero()) {
          return std::chrono::microseconds::zero();
        } else {
          return std::chrono::duration_cast<std::chrono::microseconds>(
              event.timestamp - pendingWqes_[event.queuePair].lastCompletionTs);
        }
      }();

      auto wqe = Wqe{
          .id = event.id,
          .postTs = event.timestamp,
          .timeFromPreviousWRPostUs = timeFromPreviousWRPostUs,
          .timeFromPreviousWRCompletionUs = timeFromPreviousWRCompletionUs,
          .localRank = localRankInfo_.localRank,
          .globalRank = localRankInfo_.globalRank,
          .remoteRank = event.remoteRank,
          .queuePair = event.queuePair,
          .deviceName = event.deviceName,
          .hostName = localRankInfo_.hostName,
          .totalBytes = event.totalBytes,
          .bytesInFlightOnPost = pendingWqes_[event.queuePair].bytesInFlight,
          .deviceByteOffsetAfterPost = event.deviceByteOffset,
          .putSize = event.putSize,
          .opCode = event.opCode,
      };
      if (comm_ && comm_->statex_) {
        const auto statex = comm_->statex_.get();
        auto scope = [&]() -> Wqe::Scope {
          auto myRank = statex->rank();
          auto peer = event.remoteRank;
          if (statex->isSameNode(myRank, peer))
            return Wqe::Scope::NODE;
          else if (statex->isSameRack(myRank, peer))
            return Wqe::Scope::RACK;
          else if (statex->isSameZone(myRank, peer))
            return Wqe::Scope::ZONE;
          else if (statex->isSameDc(myRank, peer))
            return Wqe::Scope::XZONE;
          else
            return Wqe::Scope::UNKNOWN;
        }();
        wqe.rtsw = statex->rtsw(),
        wqe.remoteHostName = statex->host(event.remoteRank);
        wqe.remoteRtsw = statex->rtsw(event.remoteRank);
        wqe.scope = scope;
      }
      queue.push_back(wqe);
      pendingWqes_[event.queuePair].bytesInFlight += event.totalBytes;
      pendingWqes_[event.queuePair].lastPostTs = event.timestamp;

      break;
    }
    case CtranRdmaEvent::Type::WQE_COMPLETE: {
      // We consider only write events for now
      // TODO(lume): add recv requests
      if (event.op != CtranRdmaEvent::Operation::SEND) {
        CLOGF(WARN, "CTranProfiler: Unsupported RDMA operation");
        break;
      }
      if (pendingWqes_.find(event.queuePair) == pendingWqes_.end()) {
        CLOGF(
            WARN, "CTranProfiler: WQE complete event received for unknown QP");
        break;
      }

      auto& queue = pendingWqes_[event.queuePair].sendQueue;

      // When receiving an RDMA WQE completion event, we scan the queue of
      // pending requests, looking for the request whose id matches the event:
      // - we remove all pending requests whose id is lower than that of the
      // event: these are likely requests posted with send_flags = 0 for which
      // no completion event is generated (TODO(lume): consider reporting them
      // separately in the future)
      // - we stop when the request id matches the current event: this is the
      // request we are looking for; we notify all subscribed modules and exit
      while (!queue.empty()) {
        auto& currentWqe = queue.front();
        // We received a WQE complete event whose id is greater than the one we
        // are expecting. This is possible when the previous WQE were posted
        // with send_flags = 0. Ignore for now, but consider reporting
        // separately.
        if (currentWqe.id < event.id) {
          pendingWqes_[event.queuePair].bytesInFlight -= currentWqe.totalBytes;
          pendingWqes_[event.queuePair].lastCompletionTs = event.timestamp;
          queue.pop_front();
        } else if (currentWqe.id == event.id) {
          pendingWqes_[event.queuePair].bytesInFlight -= currentWqe.totalBytes;
          pendingWqes_[event.queuePair].lastCompletionTs = event.timestamp;
          currentWqe.completionTs = event.timestamp;
          currentWqe.algorithmName = getAlgorithmName();
          currentWqe.messageSize = getSendMessageSize(event.remoteRank);
          currentWqe.bytesInFlightOnComplete =
              pendingWqes_[event.queuePair].bytesInFlight;
          for (const auto& module : modules_) {
            module->onWqeComplete(currentWqe);
          }
          queue.pop_front();
          break;
        }
        // We received a WQE complete event whose id is lower than the one we
        // are expecting. This means the WQEs were completed out-of-order or
        // that we did not save posted WQEs in the correct order.
        else {
          CLOGF(
              WARN, "CTranProfiler: WQE complete event received out of order");
          break;
        }
        if (pendingWqes_[event.queuePair].bytesInFlight < 0) {
          CLOGF(WARN, "CTranProfiler: negative bytes in flight");
          break;
        }
      }
      break;
    }
    default:
      CLOGF(WARN, "CTranProfiler: Unknown RDMA event type");
  }
}

void CtranProfiler::handleAlgoStarted(AlgoContext context) {
  algoContext_ = context;

  // decide whether to log or not log the collective based on the sampling mode
  setLoggingConfig(algoContext_.opCount);

  for (const auto& module : modules_) {
    module->onAlgoStarted(algoContext_);
  }
}

void CtranProfiler::handleAlgoCompleted() {
  for (const auto& module : modules_) {
    module->onAlgoCompleted();
  }
}

bool CtranProfiler::shouldHandleRdmaEvent() {
  if (!NCCL_CTRAN_QP_PROFILING_ENABLE && !NCCL_SLOW_RANK_ENABLE) {
    return false;
  }
  return profileModuleLoggingConfig_.shouldLogCollective_;
}

bool CtranProfiler::shouldHandleTransportEvent() {
  if (!NCCL_CTRAN_ALGO_PROFILING_ENABLE) {
    return false;
  }
  return profileModuleLoggingConfig_.shouldLogCollective_;
}

void CtranProfiler::genBufferRegEvent(
    CtranRegistrationEvent::Operation op,
    CtranRegistrationEvent::Type type) {
  if (shouldHandleTransportEvent()) {
    const CtranRegistrationEvent event{
        .timestamp = std::chrono::high_resolution_clock::now(),
        .type = type,
        .op = op,
    };
    handleRegistrationEvent(event);
  }
}

void CtranProfiler::setLoggingConfig(int opCount) {
  if (NCCL_CTRAN_ALGO_PROFILING_ENABLE) {
    if (NCCL_CTRAN_ALGO_PROFILING_SAMPLING_MODE == "opcount") {
      if (opCount % NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT == 0) {
        profileModuleLoggingConfig_.shouldLogCollective_ = true;
        return;
      }
    } else if (NCCL_CTRAN_ALGO_PROFILING_SAMPLING_MODE == "collective") {
      double r = profileModuleLoggingConfig_.getRandomNumber();
      if (r < 1.0 / NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT) {
        profileModuleLoggingConfig_.shouldLogCollective_ = true;
        return;
      }
    }
  }

  if (NCCL_CTRAN_QP_PROFILING_ENABLE) {
    profileModuleLoggingConfig_.shouldLogCollective_ = true;
    return;
  }

  if (NCCL_SLOW_RANK_ENABLE) {
    if (opCount % NCCL_CTRAN_DEVICE_TRAFFIC_SAMPLING_WEIGHT == 0) {
      profileModuleLoggingConfig_.shouldLogCollective_ = true;
      return;
    }
  }

  profileModuleLoggingConfig_.shouldLogCollective_ = false;
}
