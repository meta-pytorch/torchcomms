// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/EventMgr.h"

#include <memory>
#include <mutex>

#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars
#include "comms/utils/logger/EventMgrHelperTypes.h"

NcclScubaSample CommEvent::toSample() {
  NcclScubaSample sample("CommEvent");
  sample.addInt("commId", commId);
  sample.addInt("commHash", commHash);
  sample.addNormal("commDesc", commDesc);
  sample.addInt("rank", rank);
  sample.addInt("nranks", nRanks);
  sample.addInt("localRank", localRank);
  sample.addInt("localRanks", localRanks);

  sample.addNormal("stage", stage);
  sample.addNormal("split", split);

  sample.addDouble("timerDeltaMs", timerDeltaMs);
  sample.addNormal("timestamp", timestamp);

  sample.addInt("iteration", iteration);
  return sample;
}

// Define as unique ptr to reset the flag for testing
static std::unique_ptr<std::once_flag> memoryEventFilterFlag =
    std::make_unique<std::once_flag>();
static EventGlobalRankFilter memoryEventFilter;

static std::unique_ptr<std::once_flag> memoryRegEventFilterFlag =
    std::make_unique<std::once_flag>();
static EventGlobalRankFilter memoryRegEventFilter;

void MemoryEvent::resetFilter() {
  memoryEventFilterFlag = std::make_unique<std::once_flag>();
  memoryRegEventFilterFlag = std::make_unique<std::once_flag>();
}

bool MemoryEvent::shouldLog() {
  std::call_once(*memoryEventFilterFlag, []() {
    memoryEventFilter.initialize(
        NCCL_FILTER_MEM_LOGGING_BY_RANKS, "NCCL_FILTER_MEM_LOGGING_BY_RANKS");
    memoryRegEventFilter.initialize(
        NCCL_FILTER_MEM_REG_LOGGING_BY_RANKS,
        "NCCL_FILTER_MEM_REG_LOGGING_BY_RANKS");
  });

  // Apply different filter for reg and non-reg events
  if (isRegMemEvent) {
    return memoryRegEventFilter.isAllowed();
  } else {
    return memoryEventFilter.isAllowed();
  }
}

static const char* scopeToString(
    meta::comms::memtrace::MemCallsite::Scope scope) {
  switch (scope) {
    case meta::comms::memtrace::MemCallsite::Scope::kNccl:
      return "nccl";
    case meta::comms::memtrace::MemCallsite::Scope::kCtran:
      return "ctran";
    case meta::comms::memtrace::MemCallsite::Scope::kMccl:
      return "mccl";
  }
  return "nccl";
}

NcclScubaSample MemoryEvent::toSample() {
  NcclScubaSample sample("MemoryEvent");
  sample.addInt("commHash", commHash);
  sample.addNormal("commDesc", commDesc);
  sample.addInt("rank", rank);
  sample.addInt("nranks", nRanks);
  sample.addInt("memoryAddr", memoryAddr);
  if (bytes.has_value()) {
    sample.addInt("bytes", bytes.value());
  }
  if (numSegments.has_value()) {
    sample.addInt("numSegments", numSegments.value());
  }
  if (durationUs.has_value()) {
    sample.addInt("durationUs", durationUs.value());
  }
  if (memType.has_value()) {
    sample.addNormal("memType", memType.value());
  }
  sample.addNormal("callsite_func", callsite);
  sample.addNormal("use", use);
  sample.addNormal("callsite_scope", scopeToString(scope));
  sample.addInt("iteration", iteration);
  return sample;
}

NcclScubaSample CtranProfilerEvent::toSample() {
  auto sample = CommEvent::toSample();
  sample.addNormal("type", "CtranProfilerEvent");
  sample.addInt("remoteRank", remoteRank);
  sample.addNormal("deviceName", deviceName);
  sample.addNormal("remoteHostName", remoteHostName);
  sample.addNormal("algorithmName", algorithmName);
  sample.addNormal("sendMessageSizes", sendMessageSizes);
  sample.addNormal("receiveMessageSizes", recvMessageSizes);
  return sample;
}

NcclScubaSample CtranProfilerSlowRankEvent::toSample() {
  auto sample = CtranProfilerEvent::toSample();
  sample.addNormal("type", "CtranProfilerSlowRankEvent");
  sample.addDouble("avgBw", avgBw);
  sample.addInt("wqeCount", wqeCount);
  sample.addDouble("rooflineBwGBps", rooflineBwGBps);
  sample.addDouble("rdmaPerfEfficiencyPerc", rdmaPerfEfficiencyPerc);
  return sample;
}

NcclScubaSample CtranProfilerAlgoEvent::toSample() {
  auto sample = CtranProfilerEvent::toSample();
  sample.addNormal("type", "CtranProfilerAlgoEvent");
  sample.addNormal("direction", direction);
  sample.addInt("opCount", opCount);
  sample.addInt("readyTs", readyTs);
  sample.addInt("controlTs", controlTs);
  sample.addInt("timeFromDataToCollEndUs", timeFromDataToCollEndUs);
  sample.addInt("collectiveDurationUs", collectiveDurationUs);
  sample.addInt("bufferRegistrationTimeUs", bufferRegistrationTimeUs);
  sample.addInt("controlSyncTimeUs", controlSyncTimeUs);
  sample.addInt("dataTransferTimeUs", dataTransferTimeUs);
  return sample;
}

NcclScubaSample CtranProfilerGpeEvent::toSample() {
  auto sample = CommEvent::toSample();
  sample.addNormal("type", "CtranProfilerGpeEvent");
  // NOTE: rank and commHash are added by CommEvent::toSample() above,
  // sourced from logMetaData. Do NOT re-add here.
  sample.addInt("opCount", opCount_);
  sample.addInt("opType", opType_);
  sample.addNormal("tracePoint", tracePoint_);
  sample.addInt("iterUs", iterUs_);
  sample.addInt("durationUs", durationUs_);
  sample.addInt("aborted", aborted_ ? 1 : 0);
  sample.addNormal("message", message_);
  return sample;
}

NcclScubaSample NetworkPerfMonitorEvent::toSample() {
  auto sample = CommEvent::toSample();
  sample.addNormal("type", "NetworkPerfMonitorEvent");
  sample.addInt("cudaDev", cudaDev_);
  sample.addInt("busId", busId_);
  sample.addDouble("avgBw", avgBw_);
  return sample;
}
