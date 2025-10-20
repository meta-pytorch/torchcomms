// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "comms/ctran/commstate/CommStateX.h"
#include "comms/ctran/profiler/QueuePairProfilerModule.h"
#include "comms/utils/cvars/nccl_cvars.h"

QueuePairProfilerModule::QueuePairProfilerModule(CtranProfiler* profiler)
    : profiler_(profiler) {}

void QueuePairProfilerModule::onWqeComplete(const Wqe& wqe) {
  double r = (double)rand() / (double)RAND_MAX;
  if (r > 1.0 / NCCL_CTRAN_QP_PROFILING_SAMPLING_WEIGHT)
    return;

  auto dataTransfer = RdmaDataTransfer{
      .rank = wqe.globalRank,
      .remoteRank = wqe.remoteRank,
      .deviceName = wqe.deviceName,
      .hostName = wqe.hostName,
      .remoteHostName = wqe.remoteHostName,
      .bytes = wqe.totalBytes,
      .bytesInFlightOnPost = wqe.bytesInFlightOnPost,
      .bytesInFlightOnComplete = wqe.bytesInFlightOnComplete,
      .scope = wqe.scope,
      .putSize = wqe.putSize,
      .opCode = wqe.opCode,
      .messageSize = wqe.messageSize,
      .algorithmName = wqe.algorithmName,
      .durationUs = std::chrono::duration_cast<std::chrono::microseconds>(
          wqe.completionTs - wqe.postTs),
      .idleTimeBeforeUs = wqe.timeFromPreviousWRCompletionUs,
      .timeFromPreviousWRPostUs = wqe.timeFromPreviousWRPostUs,
      .postTs = wqe.postTs,
      .completionTs = wqe.completionTs};

  queuePairs_[wqe.queuePair].push_back(dataTransfer);
}

std::vector<uint32_t> QueuePairProfilerModule::getQueuePairsProfiled() const {
  std::vector<uint32_t> qpNums;
  for (auto& [qpNum, _] : queuePairs_) {
    qpNums.push_back(qpNum);
  }
  return qpNums;
}
std::unordered_map<
    uint32_t,
    std::deque<QueuePairProfilerModule::RdmaDataTransfer>> const&
QueuePairProfilerModule::getQueuePairs() const {
  return queuePairs_;
}

QueuePairProfilerModule::~QueuePairProfilerModule() {
  std::stringstream stream;

  stream << "[\n";
  for (auto& [qp, dataTransfers] : queuePairs_) {
    for (auto& dataTransfer : dataTransfers) {
      stream << "{\"queuePair\": " << qp << ", \"rank\": " << dataTransfer.rank
             << ", \"remoteRank\": " << dataTransfer.remoteRank
             << ", \"deviceName\": " << "\"" << dataTransfer.deviceName << "\""
             << ", \"hostName\": " << "\"" << dataTransfer.hostName << "\""
             << ", \"remoteHostName\": " << "\"" << dataTransfer.remoteHostName
             << "\"" << ", \"bytes\": " << dataTransfer.bytes
             << ", \"bytesInFlightOnPost\": "
             << dataTransfer.bytesInFlightOnPost
             << ", \"bytesInFlightOnComplete\": "
             << dataTransfer.bytesInFlightOnComplete
             << ", \"putSize\": " << dataTransfer.putSize
             << ", \"messageSize\": " << dataTransfer.messageSize
             << ", \"algorithmName\": " << "\"" << dataTransfer.algorithmName
             << "\"" << ", \"scope\": " << dataTransfer.scope
             << ", \"durationUs\": " << dataTransfer.durationUs.count()
             << ", \"idleTimeBeforeUs\": "
             << dataTransfer.idleTimeBeforeUs.count()
             << ", \"timeFromPreviousWRPostUs\": "
             << dataTransfer.timeFromPreviousWRPostUs.count()
             << ", \"postTs\": "
             << dataTransfer.postTs.time_since_epoch().count()
             << ", \"completionTs\": "
             << dataTransfer.completionTs.time_since_epoch().count() << "},"
             << "\n";
    }
  }
  stream << "{}\n]" << std::endl;

  const auto statex = profiler_->getComm()->statex_.get();
  std::string fName = "nccl_qp_profiling.comm-" +
      std::to_string(profiler_->getComm()->statex_->commHash()) + ".rank-" +
      std::to_string(statex->rank()) + ".json";
  if (NCCL_CTRAN_QP_PROFILING_OUTPUT == "file") {
    std::ofstream f("\\tmp\\" + fName);
    f << stream.str();
    f.close();
  }
}
