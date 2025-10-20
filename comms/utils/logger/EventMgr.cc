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
  sample.addNormal("callsite", callsite);
  sample.addNormal("use", use);
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
  sample.addInt("iteration", iteration);
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

NcclScubaSample NetworkPerfMonitorEvent::toSample() {
  auto sample = CommEvent::toSample();
  sample.addNormal("type", "NetworkPerfMonitorEvent");
  sample.addInt("cudaDev", cudaDev_);
  sample.addInt("busId", busId_);
  sample.addDouble("avgBw", avgBw_);
  return sample;
}
