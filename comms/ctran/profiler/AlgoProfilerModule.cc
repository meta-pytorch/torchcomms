// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <folly/String.h>
#include <chrono>
#include <fstream>
#include <sstream>

#include "comms/ctran/commstate/CommStateX.h"
#include "comms/ctran/profiler/AlgoProfilerModule.h"
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/ScubaLogger.h"

AlgoProfilerModule::AlgoProfilerModule(CtranProfiler* profiler)
    : profiler_(profiler) {
  // seed the random number generator for log
  if (profiler) {
    loggingConfig_.mt.seed(profiler_->getComm()->statex_->commHash());
    CLOGF(INFO, "AlgoProfilerModule initialized");
  }
}

AlgoProfilerModule::~AlgoProfilerModule() {
  CLOGF(INFO, "AlgoProfilerModule destroyed");

  if (NCCL_CTRAN_ALGO_PROFILING_OUTPUT == "file") {
    // when logging to file, we log everything at the end
    logAlgoToFile();
  } else if (NCCL_CTRAN_ALGO_PROFILING_OUTPUT == "scuba") {
    // when logging to scuba, we log every transfer as it completes
    // here we check if there are any remaining transfers that need to be
    // logged
    auto crtAlgo = getCrtAlgo();
    if (crtAlgo == nullptr) {
      return;
    }

    logCollectiveToScuba(crtAlgo);

    algos_.pop_back();
  } else {
    CLOGF(INFO, "AlgoProfilerModule wrong output mode");
  }
}

void AlgoProfilerModule::onAlgoStarted(const AlgoContext& context) {
  opCount_ = context.opCount;

  // When logging to Scuba, we log every collective as it completes, i.e., when
  // the next collective starts. When logging to file, we accumulate the
  // information about all collectives and log everything when the communicator
  // is destroyed.

  // if this is the first collective we track, we create a new Algo
  // object for it and return
  auto crtAlgo = getCrtAlgo();
  bool shouldCreateAlgo =
      profiler_->getProfileModuleLoggingConfig().shouldLogCollective_;

  if (crtAlgo == nullptr) {
    if (shouldCreateAlgo) {
      algos_.emplace_back();
      algos_.back().recvTransferStats.messageSizes =
          (context.recvMessageSizes.size() > 0)
          ? folly::join(",", context.recvMessageSizes)
          : std::to_string(context.recvMessageSize);
      algos_.back().sendTransferStats.messageSizes =
          (context.sendMessageSizes.size() > 0)
          ? folly::join(",", context.sendMessageSizes)
          : std::to_string(context.sendMessageSize);
    }
    return;
  }

  // otherwise, we log the previous collective (if output is scuba) and create a
  // new Algo object for the current collective
  if (NCCL_CTRAN_ALGO_PROFILING_OUTPUT == "scuba") {
    logCollectiveToScuba(crtAlgo);

    algos_.pop_back();
  }

  if (shouldCreateAlgo) {
    algos_.emplace_back();
    algos_.back().recvTransferStats.messageSizes =
        (context.recvMessageSizes.size() > 0)
        ? folly::join(",", context.recvMessageSizes)
        : std::to_string(context.recvMessageSize);
    algos_.back().sendTransferStats.messageSizes =
        (context.sendMessageSizes.size() > 0)
        ? folly::join(",", context.sendMessageSizes)
        : std::to_string(context.sendMessageSize);
  }
}

void AlgoProfilerModule::onAlgoCompleted() {
  if (!profiler_->getProfileModuleLoggingConfig().shouldLogCollective_) {
    return;
  }
  auto crtAlgo = getCrtAlgo();

  if (crtAlgo == nullptr) {
    CLOGF(
        WARN,
        "[Ctran algo profiler] onAlgoCompleted() called before onAlgoStarted()");
    return;
  }

  crtAlgo->endTs = std::chrono::high_resolution_clock::now();
}

void AlgoProfilerModule::onCtrlReceived(const CtranTransportEvent& event) {
  ctrlReceivedCount_++;
  auto crtAlgo = getCrtAlgo();
  if (crtAlgo == nullptr) {
    return;
  }

  auto& crtTransferStats = crtAlgo->sendTransferStats;

  if (crtTransferStats.maxCtrlReceivedTs ==
          std::chrono::time_point<std::chrono::high_resolution_clock>() ||
      crtTransferStats.maxCtrlReceivedTs < event.timestamp) {
    crtTransferStats.maxCtrlReceivedTs = event.timestamp;
  }
  if (crtTransferStats.minCtrlReceivedTs ==
          std::chrono::time_point<std::chrono::high_resolution_clock>() ||
      crtTransferStats.minCtrlReceivedTs > event.timestamp) {
    crtTransferStats.minCtrlReceivedTs = event.timestamp;
  }

  crtTransferStats.deviceName = event.deviceName;
  crtTransferStats.algorithmName = profiler_->getAlgorithmName();
}

void AlgoProfilerModule::onCtrlComplete(const CtranTransportEvent& event) {
  ctrlCompleteCount_++;
  auto crtAlgo = getCrtAlgo();
  if (crtAlgo == nullptr) {
    return;
  }

  auto& crtTransferStats = crtAlgo->recvTransferStats;

  if (crtTransferStats.minCtrlCompleteTs ==
          std::chrono::time_point<std::chrono::high_resolution_clock>() ||
      crtTransferStats.minCtrlCompleteTs > event.timestamp) {
    crtTransferStats.minCtrlCompleteTs = event.timestamp;
  }
  if (crtTransferStats.maxCtrlCompleteTs ==
          std::chrono::time_point<std::chrono::high_resolution_clock>() ||
      crtTransferStats.maxCtrlCompleteTs < event.timestamp) {
    crtTransferStats.maxCtrlCompleteTs = event.timestamp;
  }
  crtTransferStats.deviceName = event.deviceName;
  crtTransferStats.algorithmName = profiler_->getAlgorithmName();
}

void AlgoProfilerModule::onReadyToSend(const CtranTransportEvent& event) {
  readyToSendCount_++;
  auto crtAlgo = getCrtAlgo();
  if (crtAlgo == nullptr) {
    return;
  }

  crtAlgo->sendTransferStats.initTs = crtAlgo->startTs;
  if (crtAlgo->sendTransferStats.minReadyToSendTs ==
          std::chrono::time_point<std::chrono::high_resolution_clock>() ||
      crtAlgo->sendTransferStats.minReadyToSendTs > event.timestamp) {
    crtAlgo->sendTransferStats.minReadyToSendTs = event.timestamp;
  }
  if (crtAlgo->sendTransferStats.maxReadyToSendTs ==
          std::chrono::time_point<std::chrono::high_resolution_clock>() ||
      crtAlgo->sendTransferStats.maxReadyToSendTs < event.timestamp) {
    crtAlgo->sendTransferStats.maxReadyToSendTs = event.timestamp;
  }
}

void AlgoProfilerModule::onReadyToReceive(const CtranTransportEvent& event) {
  readyToReceiveCount_++;
  auto crtAlgo = getCrtAlgo();
  if (crtAlgo == nullptr) {
    return;
  }

  crtAlgo->recvTransferStats.initTs = crtAlgo->startTs;
  if (crtAlgo->recvTransferStats.minReadyToReceiveTs ==
          std::chrono::time_point<std::chrono::high_resolution_clock>() ||
      crtAlgo->recvTransferStats.minReadyToReceiveTs > event.timestamp) {
    crtAlgo->recvTransferStats.minReadyToReceiveTs = event.timestamp;
  }

  if (crtAlgo->recvTransferStats.maxReadyToReceiveTs ==
          std::chrono::time_point<std::chrono::high_resolution_clock>() ||
      crtAlgo->recvTransferStats.maxReadyToReceiveTs < event.timestamp) {
    crtAlgo->recvTransferStats.maxReadyToReceiveTs = event.timestamp;
  }
}

void AlgoProfilerModule::onPutIssued(const CtranTransportEvent& event) {
  putIssuedCount_++;
  auto crtAlgo = getCrtAlgo();
  if (crtAlgo == nullptr) {
    return;
  }

  auto& crtTransfer = crtAlgo->sendTransferStats;
  crtTransfer.totalBytes += event.totalBytes;
  crtTransfer.outstandingPutsCount++;
}

void AlgoProfilerModule::onPutComplete(const CtranTransportEvent& event) {
  putCompleteCount_++;
  auto crtAlgo = getCrtAlgo();
  if (crtAlgo == nullptr) {
    return;
  }

  auto& crtTransfer = crtAlgo->sendTransferStats;

  if (crtTransfer.endTs ==
          std::chrono::time_point<std::chrono::high_resolution_clock>() ||
      crtTransfer.endTs < event.timestamp) {
    crtTransfer.endTs = event.timestamp;
    crtTransfer.outstandingPutsCount--;
  }
}

void AlgoProfilerModule::onRecvComplete(const CtranTransportEvent& event) {
  recvCompleteCount_++;
  auto crtAlgo = getCrtAlgo();
  if (crtAlgo == nullptr) {
    return;
  }

  auto& crtTransfer = crtAlgo->recvTransferStats;

  crtTransfer.deviceName = event.deviceName;
  crtTransfer.totalBytes += event.totalBytes;
  crtTransfer.algorithmName = profiler_->getAlgorithmName();

  if (crtTransfer.endTs ==
          std::chrono::time_point<std::chrono::high_resolution_clock>() ||
      crtTransfer.endTs < event.timestamp) {
    crtTransfer.endTs = event.timestamp;
  }
}

void AlgoProfilerModule::onBufferRegistrationStart(
    const CtranRegistrationEvent& event) {
  auto crtAlgo = getCrtAlgo();
  if (crtAlgo == nullptr) {
    return;
  }

  auto& crtTransfer =
      (event.op == CtranRegistrationEvent::Operation::SEND
           ? crtAlgo->sendTransferStats
           : crtAlgo->recvTransferStats);

  crtTransfer.buffRegStartTs = event.timestamp;
}

void AlgoProfilerModule::onBufferRegistrationComplete(
    const CtranRegistrationEvent& event) {
  auto crtAlgo = getCrtAlgo();
  if (crtAlgo == nullptr) {
    return;
  }

  auto& crtTransfer =
      (event.op == CtranRegistrationEvent::Operation::SEND
           ? crtAlgo->sendTransferStats
           : crtAlgo->recvTransferStats);

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      event.timestamp - crtTransfer.buffRegStartTs)
                      .count();
  if (crtTransfer.bufferRegDurationUs < duration) {
    crtTransfer.bufferRegDurationUs = duration;
  }
}

void AlgoProfilerModule::writeDataTransferStatsToStream(
    std::stringstream& stream,
    const DataTransferStats& stats,
    int rank,
    const std::string& host,
    const std::string& direction) {
  if (direction == "send") {
    auto record = fmt::format(
        "rank: {}, host: {}, deviceName: {}, direction: {}, totalBytes: {}, messageSizes: {}, algorithmName: {}, initTs: {}, minReadyTs: {}, maxReadyTs: {}, minCtrlReceivedTs: {}, maxCtrlReceivedTs: {}, numberOfPuts: {}, endTs: {}, opCount: {}",
        rank,
        host,
        stats.deviceName,
        direction,
        stats.totalBytes,
        stats.messageSizes,
        stats.algorithmName,
        stats.initTs.time_since_epoch().count(),
        stats.minReadyToSendTs.time_since_epoch().count(),
        stats.maxReadyToSendTs.time_since_epoch().count(),
        stats.minCtrlReceivedTs.time_since_epoch().count(),
        stats.maxCtrlReceivedTs.time_since_epoch().count(),
        putCompleteCount_,
        stats.endTs.time_since_epoch().count(),
        opCount_);
    stream << record << std::endl;
  } else if (direction == "recv") {
    auto record = fmt::format(
        "rank: {}, host: {}, deviceName: {}, direction: {}, totalBytes: {}, messageSizes: {}, algorithmName: {}, initTs: {}, minReadyTs: {}, maxReadyTs: {}, minCtrlReceivedTs: {}, maxCtrlReceivedTs: {}, numberOfPuts: {}, endTs: {}, opCount: {}",
        rank,
        host,
        stats.deviceName,
        direction,
        stats.totalBytes,
        stats.messageSizes,
        stats.algorithmName,
        stats.initTs.time_since_epoch().count(),
        stats.minReadyToReceiveTs.time_since_epoch().count(),
        stats.maxReadyToReceiveTs.time_since_epoch().count(),
        stats.minCtrlCompleteTs.time_since_epoch().count(),
        stats.maxCtrlCompleteTs.time_since_epoch().count(),
        putCompleteCount_,
        stats.endTs.time_since_epoch().count(),
        opCount_);
    stream << record << std::endl;
  }
}

void AlgoProfilerModule::logAlgoToFile() {
  std::stringstream stream;
  const auto statex = profiler_->getComm()->statex_.get();

  auto rank = statex->rank();
  auto host = statex->host();

  stream << "[\n";
  for (auto& crtAlgo : algos_) {
    writeDataTransferStatsToStream(
        stream, crtAlgo.sendTransferStats, rank, host, "send");
    writeDataTransferStatsToStream(
        stream, crtAlgo.recvTransferStats, rank, host, "recv");
  }
  stream << "{}\n]" << std::endl;

  auto fName = fmt::format(
      "nccl_algo_profiling.rank-{}.comm-{}.json",
      statex->rank(),
      statex->commHash());
  if (NCCL_CTRAN_ALGO_PROFILING_OUTPUT == "file") {
    XLOGF(INFO, "Writing algo profiling to file: {}", fName);
    std::ofstream f("/tmp/" + fName);
    f << stream.str();
    f.close();
  }
}

bool AlgoProfilerModule::isTimestampValid(
    std::chrono::time_point<std::chrono::high_resolution_clock> ts) {
  return ts != std::chrono::time_point<std::chrono::high_resolution_clock>();
}

void AlgoProfilerModule::logCollStatsToScuba(
    AlgoProfilerModule::Algo* crtAlgo) {
  // table to log to is determined by the value of
  // NCCL_CTRAN_ALGO_PROFILING_OUTPUT
  auto timeWaitingForInitUs = fmax(
      crtAlgo->sendTransferStats.bufferRegDurationUs,
      crtAlgo->recvTransferStats.bufferRegDurationUs);

  long timeWaitingForControlUs = 0;
  if (isTimestampValid(crtAlgo->sendTransferStats.maxCtrlReceivedTs) &&
      isTimestampValid(crtAlgo->sendTransferStats.minReadyToSendTs) &&
      isTimestampValid(crtAlgo->recvTransferStats.maxCtrlCompleteTs) &&
      isTimestampValid(crtAlgo->recvTransferStats.minReadyToReceiveTs)) {
    timeWaitingForControlUs =
        std::chrono::duration_cast<std::chrono::microseconds>(
            max(crtAlgo->sendTransferStats.maxCtrlReceivedTs -
                    crtAlgo->sendTransferStats.minReadyToSendTs,
                crtAlgo->recvTransferStats.maxCtrlCompleteTs -
                    crtAlgo->recvTransferStats.minReadyToReceiveTs))
            .count();
  } else {
    CLOGF(
        WARN,
        "[Ctran algo profiler] One or more timestamps for timeWaitingForCtrl are invalid.");
  }

  long timeWaitingForDataUs = 0;
  if (isTimestampValid(crtAlgo->sendTransferStats.endTs) &&
      isTimestampValid(crtAlgo->sendTransferStats.minCtrlReceivedTs) &&
      isTimestampValid(crtAlgo->recvTransferStats.endTs) &&
      isTimestampValid(crtAlgo->recvTransferStats.minCtrlCompleteTs)) {
    timeWaitingForDataUs =
        std::chrono::duration_cast<std::chrono::microseconds>(
            max(crtAlgo->sendTransferStats.endTs -
                    crtAlgo->sendTransferStats.minCtrlReceivedTs,
                crtAlgo->recvTransferStats.endTs -
                    crtAlgo->recvTransferStats.minCtrlCompleteTs))
            .count();
  } else {
    CLOGF(
        WARN,
        "[Ctran algo profiler] One or more timestamps for calculate timeWaitingForData are invalid.");
  }

  long readyTs = 0;
  if (isTimestampValid(crtAlgo->sendTransferStats.maxReadyToSendTs) &&
      isTimestampValid(crtAlgo->sendTransferStats.maxReadyToSendTs)) {
    readyTs = max(crtAlgo->sendTransferStats.maxReadyToSendTs,
                  crtAlgo->recvTransferStats.maxReadyToReceiveTs)
                  .time_since_epoch()
                  .count();
  } else {
    CLOGF(
        WARN,
        "[Ctran algo profiler] One or more timestamps for calculate readyTs are invalid.");
  }

  long controlTs = 0;
  if (isTimestampValid(crtAlgo->sendTransferStats.maxCtrlReceivedTs) &&
      isTimestampValid(crtAlgo->recvTransferStats.maxCtrlCompleteTs)) {
    controlTs = max(crtAlgo->sendTransferStats.maxCtrlReceivedTs,
                    crtAlgo->recvTransferStats.maxCtrlCompleteTs)
                    .time_since_epoch()
                    .count();
  } else {
    CLOGF(
        WARN,
        "[Ctran algo profiler] One or more timestamps for calculate readyTs are invalid.");
  }

  long timeFromDataToCollEndUs = 0;
  if (isTimestampValid(crtAlgo->sendTransferStats.endTs) &&
      isTimestampValid(crtAlgo->recvTransferStats.endTs)) {
    timeFromDataToCollEndUs =
        std::chrono::duration_cast<std::chrono::microseconds>(
            crtAlgo->endTs -
            max(crtAlgo->sendTransferStats.endTs,
                crtAlgo->recvTransferStats.endTs))
            .count();
  } else {
    CLOGF(
        WARN,
        "[Ctran algo profiler] One or more timestamps for calculate timeFromDataToCollEndUs are invalid.");
  }

  long collDurationUs = 0;
  if (isTimestampValid(crtAlgo->endTs) && isTimestampValid(crtAlgo->startTs)) {
    collDurationUs = std::chrono::duration_cast<std::chrono::microseconds>(
                         crtAlgo->endTs - crtAlgo->startTs)
                         .count();
  } else {
    CLOGF(
        WARN,
        "[Ctran algo profiler] One or more timestamps for calculate collDurationUs are invalid.");
  }

  NcclScubaEvent scubaEvent(
      std::make_unique<CtranProfilerAlgoEvent>(
          &profiler_->getComm()->logMetaData_,
          "algoProfiling",
          "",
          0,
          0,
          crtAlgo->sendTransferStats.deviceName,
          "",
          profiler_->getAlgorithmName(),
          crtAlgo->sendTransferStats.messageSizes,
          crtAlgo->recvTransferStats.messageSizes,
          "",
          crtAlgo->sendTransferStats.totalBytes,
          crtAlgo->recvTransferStats.totalBytes,
          timeWaitingForInitUs,
          timeWaitingForControlUs,
          timeWaitingForDataUs,
          opCount_ -
              1, // data is collective in the previous onAlgoStarted() call
          readyTs,
          controlTs,
          timeFromDataToCollEndUs,
          collDurationUs));
  scubaEvent.record();
}

void AlgoProfilerModule::logCollectiveToScuba(
    AlgoProfilerModule::Algo* crtAlgo) {
  if (NCCL_CTRAN_ALGO_PROFILING_SAMPLING_MODE == "collective" or
      NCCL_CTRAN_ALGO_PROFILING_SAMPLING_MODE == "opcount") {
    // if the sampling mode is collective-based or opcount-based, we either
    // log/or ignore the entire collective.
    logCollStatsToScuba(crtAlgo);
  } else {
    CLOGF(
        WARN,
        "[Ctran algo profiler] Unknown sampling mode: {}",
        NCCL_CTRAN_ALGO_PROFILING_SAMPLING_MODE.c_str());
  }
}
