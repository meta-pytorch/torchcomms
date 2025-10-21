// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <random>

#include "comms/ctran/profiler/CtranProfiler.h"

/*
 * The algo profiling module is responsible for profiling the performance of
 * algorithms during training jobs.
 */
class AlgoProfilerModule : public CtranProfilerModule {
 public:
  struct Put {
    Put(std::chrono::time_point<std::chrono::high_resolution_clock> timestamp,
        uint64_t bytes)
        : issuedTs(timestamp), bytes(bytes) {}
    std::chrono::time_point<std::chrono::high_resolution_clock> issuedTs;
    std::chrono::time_point<std::chrono::high_resolution_clock> completeTs;
    uint64_t bytes;
  };

  struct DataTransferStats {
    enum Direction { SEND, RECV };
    int rank;
    int remoteRank;
    Direction direction;
    std::string deviceName;
    uint64_t totalBytes;
    std::string algorithmName;
    std::string messageSizes;
    int outstandingPutsCount;
    std::chrono::time_point<std::chrono::high_resolution_clock> initTs;
    std::chrono::time_point<std::chrono::high_resolution_clock>
        minReadyToSendTs;
    std::chrono::time_point<std::chrono::high_resolution_clock>
        maxReadyToSendTs;
    std::chrono::time_point<std::chrono::high_resolution_clock>
        minReadyToReceiveTs;
    std::chrono::time_point<std::chrono::high_resolution_clock>
        maxReadyToReceiveTs;
    std::chrono::time_point<std::chrono::high_resolution_clock>
        minCtrlReceivedTs;
    std::chrono::time_point<std::chrono::high_resolution_clock>
        maxCtrlReceivedTs;
    std::chrono::time_point<std::chrono::high_resolution_clock>
        minCtrlCompleteTs;
    std::chrono::time_point<std::chrono::high_resolution_clock>
        maxCtrlCompleteTs;
    std::chrono::time_point<std::chrono::high_resolution_clock> endTs;
    long bufferRegDurationUs{0};
    std::chrono::time_point<std::chrono::high_resolution_clock> buffRegStartTs;
  };

  /*
   * An Algo object captures the information about each collective run.
   */
  struct Algo {
    // when the collective algorithm starts execution
    std::chrono::time_point<std::chrono::high_resolution_clock> startTs{
        std::chrono::high_resolution_clock::now()};
    std::chrono::time_point<std::chrono::high_resolution_clock> endTs;
    // information about outgoing and incoming transfers during the collective
    // run
    DataTransferStats sendTransferStats;
    DataTransferStats recvTransferStats;
  };

  struct LoggingConfig {
    std::mt19937 mt;

    double getRandomNumber() {
      std::uniform_int_distribution<int> dist(0.0, 1.0);
      return dist(mt);
    }
  };

  AlgoProfilerModule(CtranProfiler* profiler);

  ~AlgoProfilerModule();

  void onAlgoStarted(const AlgoContext& context) override;
  void onAlgoCompleted() override;
  void onCtrlReceived(const CtranTransportEvent& event) override;
  void onCtrlComplete(const CtranTransportEvent& event) override;
  void onReadyToSend(const CtranTransportEvent& event) override;
  void onReadyToReceive(const CtranTransportEvent& event);
  void onPutIssued(const CtranTransportEvent& event) override;
  void onPutComplete(const CtranTransportEvent& event) override;
  void onRecvComplete(const CtranTransportEvent& event) override;
  void onBufferRegistrationComplete(
      const CtranRegistrationEvent& event) override;
  void onBufferRegistrationStart(const CtranRegistrationEvent& event) override;

  const int getCtrlReceivedCount() const {
    return ctrlReceivedCount_;
  }
  const int getReadyToSendCount() const {
    return readyToSendCount_;
  }
  const int getReadyToReceiveCount() const {
    return readyToReceiveCount_;
  }
  const int getPutIssuedCount() const {
    return putIssuedCount_;
  }
  const int getPutCompleteCount() const {
    return putCompleteCount_;
  }
  const int getRecvCompleteCount() const {
    return recvCompleteCount_;
  }
  Algo* getCrtAlgo() {
    if (!algos_.empty()) {
      return &algos_.back();
    }

    return nullptr;
  }

  void logAlgoToFile();
  void writeDataTransferStatsToStream(
      std::stringstream& stream,
      const DataTransferStats& stats,
      int rank,
      const std::string& host,
      const std::string& direction);
  void logCollectiveToScuba(AlgoProfilerModule::Algo* crtAlgo);
  void logCollStatsToScuba(AlgoProfilerModule::Algo* crtAlgo);
  bool isTimestampValid(
      std::chrono::time_point<std::chrono::high_resolution_clock> ts);

  CtranProfiler* profiler_;
  std::deque<Algo> algos_;
  LoggingConfig loggingConfig_;
  int ctrlReceivedCount_{0};
  int ctrlCompleteCount_{0};
  int readyToSendCount_{0};
  int readyToReceiveCount_{0};
  int putIssuedCount_{0};
  int putCompleteCount_{0};
  int recvCompleteCount_{0};
  uint64_t opCount_{0};
};
