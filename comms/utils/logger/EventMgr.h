// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <sys/types.h>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/NcclScubaSample.h"
#include "comms/utils/trainer/TrainerContext.h"

#define LOGGER_PG_ID_DEFAULT 0x80000000UL

enum class LoggerEventType {
  DebugEventType,
  CommEventType,
  MemoryEventType,
  CollEventType,
  CollSignatureEventType,
  CollTimestampEventType,
  ErrorEventType,
  CtranProfilerEventType,
  CtranProfilerSlowRankModuleEventType,
  CtranProfilerAlgoEventType,
  CommdumpEventType,
  TerminateEventType,
};

class LoggerEvent {
 public:
  virtual void setTimerDelta(double delta) = 0;
  virtual void setTimestamp() = 0;
  virtual bool shouldLog() = 0;
  virtual LoggerEventType getEventType() = 0;
  virtual ~LoggerEvent() = default;
  virtual NcclScubaSample toSample() = 0;
  virtual std::string getStage() = 0;
};

class CommEvent : public LoggerEvent {
 public:
  CommEvent() = default;
  CommEvent(
      const CommLogData* logMetaData,
      const std::string& stage,
      const std::string& split,
      double delta = 0.0)
      : commId(logMetaData ? logMetaData->commId : 0),
        commHash(
            logMetaData
                ? logMetaData->commHash
                : 0xfaceb00c12345678 /*Dummy placeholder value for commHash*/),
        commDesc(logMetaData ? std::string(logMetaData->commDesc) : ""),
        rank(logMetaData ? logMetaData->rank : 0),
        nRanks(logMetaData ? logMetaData->nRanks : 0),
        stage(stage),
        split(split),
        timerDeltaMs(delta) {}

  CommEvent(
      const CommLogData* logMetaData,
      int localRank,
      int localRanks,
      const std::string& stage)
      : commId(logMetaData ? logMetaData->commId : 0),
        commHash(logMetaData ? logMetaData->commHash : 0xfaceb00c12345678),
        commDesc(logMetaData ? std::string(logMetaData->commDesc) : ""),
        rank(logMetaData ? logMetaData->rank : 0),
        nRanks(logMetaData ? logMetaData->nRanks : 0),
        localRank(localRank),
        localRanks(localRanks),
        stage(stage) {}

  ~CommEvent() override = default;

  void setTimerDelta(double delta) override {
    timerDeltaMs = delta;
  }

  void setTimestamp() override {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    timestamp = std::to_string(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  }

  bool shouldLog() override {
    return true;
  }

  LoggerEventType getEventType() override {
    return LoggerEventType::CommEventType;
  }

  std::string getStage() override {
    return stage;
  }
  NcclScubaSample toSample() override;

 private:
  const unsigned long long commId = 0;
  const uint64_t commHash = 0xfaceb00c12345678;
  const std::string commDesc;
  const int rank = 0;
  const int nRanks = 0;
  int localRank = -1;
  int localRanks = -1;
  const std::string stage;
  const std::string split;
  double timerDeltaMs = 0.0;
  std::string timestamp;
};

class MemoryEvent : public LoggerEvent {
 public:
  MemoryEvent() = default;
  MemoryEvent(
      const CommLogData& logMetaData,
      const std::string& callsite,
      const std::string& use,
      uintptr_t memoryAddr,
      std::optional<int64_t> bytes = std::nullopt,
      std::optional<int> numSegments = std::nullopt,
      std::optional<int64_t> durationUs = std::nullopt,
      bool isRegMemEvent = false)
      : commHash(logMetaData.commHash),
        commDesc(logMetaData.commDesc),
        rank(logMetaData.rank),
        nRanks(logMetaData.nRanks),
        callsite(callsite),
        use(use),
        memoryAddr(memoryAddr),
        bytes(bytes),
        numSegments(numSegments),
        durationUs(durationUs),
        isRegMemEvent(isRegMemEvent) {
    iteration = ncclxGetIteration();
  }

  ~MemoryEvent() override = default;

  void setTimerDelta(double delta) override {}
  void setTimestamp() override {}
  bool shouldLog() override;
  std::string getStage() override {
    return std::string("");
  }

  LoggerEventType getEventType() override {
    return LoggerEventType::MemoryEventType;
  }

  // For testing only
  static void resetFilter();
  NcclScubaSample toSample() override;

 private:
  const struct CommLogData* logMetaData{};
  const uint64_t commHash = 0xfaceb00c12345678;
  const std::string commDesc = "undefined";
  const int rank = -1;
  const int nRanks = -1;
  const std::string callsite;
  const std::string use;
  // training step. If not set by trainer, it is always -1. Also see
  // ncclxGetIteration().
  int64_t iteration = -1;
  uintptr_t memoryAddr{};
  std::optional<int64_t> bytes;
  std::optional<int> numSegments;
  std::optional<int64_t> durationUs;
  bool isRegMemEvent = false;
};

class CtranProfilerEvent : public CommEvent {
 public:
  CtranProfilerEvent();
  CtranProfilerEvent(
      const struct CommLogData* logMetaData,
      const std::string& stage,
      const std::string& split,
      double delta,
      const int remoteRank,
      const std::string& deviceName,
      const std::string& remoteHostName,
      const std::string& algorithmName,
      const std::string& sendMessageSizes,
      const std::string& recvMessageSizes)
      : CommEvent(logMetaData, stage, split, delta),
        remoteRank(remoteRank),
        deviceName(deviceName),
        remoteHostName(remoteHostName),
        algorithmName(algorithmName),
        sendMessageSizes(sendMessageSizes),
        recvMessageSizes(recvMessageSizes) {}

  ~CtranProfilerEvent() override = default;

  LoggerEventType getEventType() override {
    return LoggerEventType::CtranProfilerEventType;
  }
  void setTimerDelta(double delta) override {}
  void setTimestamp() override {}
  bool shouldLog() override {
    return true;
  }
  NcclScubaSample toSample() override;

 private:
  const int remoteRank;
  const std::string deviceName, remoteHostName, algorithmName;
  const std::string sendMessageSizes;
  const std::string recvMessageSizes;
};

class CtranProfilerSlowRankEvent : public CtranProfilerEvent {
 public:
  CtranProfilerSlowRankEvent();
  CtranProfilerSlowRankEvent(
      const struct CommLogData* logMetaData,
      const std::string& stage,
      const std::string& split,
      double delta,
      const int remoteRank,
      const std::string& deviceName,
      const std::string& remoteHostName,
      const std::string& algorithmName,
      const std::string& sendMessageSizes,
      const std::string& recvMessageSizes,
      double avgBw,
      int wqeCount,
      double rooflineBwGBps,
      double rdmaPerfEfficiencyPerc)
      : CtranProfilerEvent(
            logMetaData,
            stage,
            split,
            delta,
            remoteRank,
            deviceName,
            remoteHostName,
            algorithmName,
            sendMessageSizes,
            recvMessageSizes),
        avgBw(avgBw),
        wqeCount(wqeCount),
        rooflineBwGBps(rooflineBwGBps),
        rdmaPerfEfficiencyPerc(rdmaPerfEfficiencyPerc) {}
  ~CtranProfilerSlowRankEvent() override = default;

  LoggerEventType getEventType() override {
    return LoggerEventType::CtranProfilerSlowRankModuleEventType;
  }
  void setTimerDelta(double delta) override {}
  void setTimestamp() override {}

  bool shouldLog() override {
    return true;
  }
  NcclScubaSample toSample() override;

 private:
  double avgBw;
  int wqeCount;
  double rooflineBwGBps;
  double rdmaPerfEfficiencyPerc;
};

class CtranProfilerAlgoEvent : public CtranProfilerEvent {
 public:
  CtranProfilerAlgoEvent();
  CtranProfilerAlgoEvent(
      const struct CommLogData* logMetaData,
      const std::string& stage,
      const std::string& split,
      double delta,
      const int remoteRank,
      const std::string& deviceName,
      const std::string& remoteHostName,
      const std::string& algorithmName,
      const std::string& sendMessageSizes,
      const std::string& recvMessageSizes,
      const std::string& direction,
      uint64_t sendTotalBytes,
      uint64_t recvTotalBytes,
      uint64_t bufferRegistrationTimeUs,
      uint64_t controlSyncTimeUs,
      uint64_t dataTransferTimeUs,
      uint64_t opCount,
      uint64_t readyTs,
      uint64_t controlTs,
      uint64_t timeFromDataToCollEndUs,
      uint64_t collectiveDurationUs)
      : CtranProfilerEvent(
            logMetaData,
            stage,
            split,
            delta,
            remoteRank,
            deviceName,
            remoteHostName,
            algorithmName,
            sendMessageSizes,
            recvMessageSizes),
        direction(direction),
        sendTotalBytes(sendTotalBytes),
        recvTotalBytes(recvTotalBytes),
        bufferRegistrationTimeUs(bufferRegistrationTimeUs),
        controlSyncTimeUs(controlSyncTimeUs),
        dataTransferTimeUs(dataTransferTimeUs),
        opCount(opCount),
        readyTs(readyTs),
        controlTs(controlTs),
        timeFromDataToCollEndUs(timeFromDataToCollEndUs),
        collectiveDurationUs(collectiveDurationUs) {
    iteration = ncclxGetIteration();
  }

  ~CtranProfilerAlgoEvent() override = default;

  LoggerEventType getEventType() override {
    return LoggerEventType::CtranProfilerAlgoEventType;
  }
  void setTimerDelta(double delta) override {}
  NcclScubaSample toSample() override;

 private:
  const std::string direction;
  uint64_t sendTotalBytes;
  uint64_t recvTotalBytes;
  uint64_t bufferRegistrationTimeUs;
  uint64_t controlSyncTimeUs;
  uint64_t dataTransferTimeUs;
  // training step. If not set by trainer, it is always -1. Also see
  // ncclxGetIteration().
  int64_t iteration = -1;
  uint64_t opCount;
  uint64_t readyTs;
  uint64_t controlTs;
  uint64_t timeFromDataToCollEndUs;
  uint64_t collectiveDurationUs;
};

class NetworkPerfMonitorEvent : public CommEvent {
 public:
  NetworkPerfMonitorEvent() = delete;

  NetworkPerfMonitorEvent(
      const CommLogData& logMetaData,
      int cudaDev,
      int64_t busId,
      double bandwidth)
      : CommEvent(&logMetaData, "", ""),
        cudaDev_(cudaDev),
        busId_(busId),
        avgBw_(bandwidth) {}

  ~NetworkPerfMonitorEvent() override = default;

  LoggerEventType getEventType() override {
    return LoggerEventType::CtranProfilerSlowRankModuleEventType;
  }

  NcclScubaSample toSample() override;

 private:
  const int cudaDev_{-1};
  const int64_t busId_{0};
  const double avgBw_{0};
};
