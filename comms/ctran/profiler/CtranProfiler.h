// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <cstdint>

#include <deque>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/utils/logger/LogUtils.h"

#define MAX_PENDING_WQES 256

// execute the profiling command only if the condition is true
#define CTRAN_PROFILING_IF(cond_, cmd_) \
  if (cond_) {                          \
    cmd_;                               \
  }

/*
 * Ctran transport level events are triggered at the start or completion of
 * Ctran requests, such as isendCtrl, irecvCtrl, or iput.
 */
struct CtranTransportEvent {
  enum Type {
    IRECVCTRL_ISSUED /* the sender is ready to send, i.e., called `iRecvCtrl` */
    ,
    ISENDCTRL_ISSUED /* the receiver is ready to receive, i.e., called
                        `iSendCtrl` */
    ,
    IRECVCTRL_COMPLETE /* the sender received the control message from the
                          receiver */
    ,
    ISENDCTRL_COMPLETE /* the receiver control message is complete */,
    PUT_ISSUED /* the sender issued a put request */,
    PUT_COMPLETE /* the sender completed a put request */,
    GET_ISSUED /* the receiver issued a get request */,
    GET_COMPLETE /* the receiver completed a get request */,
    RECV_STARTED /* the receiver started to receive data from the sender */,
    RECV_COMPLETE /* the receiver completed receiving data from the sender */,
  };

  // when was the event recorded
  std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;

  // event type
  Type type;

  // event context: device name, remote rank and bytes transferred
  std::string deviceName;
  int remoteRank;
  uint64_t totalBytes;
  std::string algorithmName;
};

struct CtranRegistrationEvent {
  enum Type {
    BUFFER_REGISTRATION_START,
    BUFFER_REGISTRATION_COMPLETE,
  };
  enum Operation { SEND, RECV };

  // when was the event recorded
  std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;

  // event type
  Type type;

  // send buffer or recv buffer
  Operation op;
};

/*
 * Ctran RDMA level events are triggered when work requests are posted and
 * completed.
 */
struct CtranRdmaEvent {
  enum Type { WR_POSTED, WQE_COMPLETE };
  enum Operation { SEND, RECV };

  // when was the event recorded
  std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;

  // event type, direction, and work request id
  Type type;
  Operation op;
  uint64_t id;

  // event context: device name, remote rank, queue pair, and bytes transferred
  std::string deviceName;
  int remoteRank;
  std::string remoteHostName;
  uint32_t queuePair;
  uint64_t totalBytes;
  uint64_t deviceByteOffset;
  std::string algorithmName;
  size_t messageSize;
  size_t putSize;
  size_t getSize;
  size_t opCode;
};

/*
 * struct Wqe holds information about an RDMA work request.
 */
struct Wqe {
  enum Scope { NODE, RACK, ZONE, XZONE, UNKNOWN };
  uint64_t id;
  std::chrono::time_point<std::chrono::high_resolution_clock> postTs;
  std::chrono::time_point<std::chrono::high_resolution_clock> completionTs;
  std::chrono::microseconds timeFromPreviousWRPostUs;
  std::chrono::microseconds timeFromPreviousWRCompletionUs;
  int localRank;
  int globalRank;
  int remoteRank;
  uint32_t queuePair;
  std::string deviceName;
  std::string hostName;
  std::string remoteHostName;
  std::string rtsw;
  std::string remoteRtsw;
  Scope scope;
  uint64_t totalBytes;
  uint64_t bytesInFlightOnPost;
  uint64_t bytesInFlightOnComplete;
  uint64_t deviceByteOffsetAfterPost;
  std::string algorithmName;
  size_t messageSize;
  size_t putSize;
  size_t opCode;
};

/*
 * struct QueuePair holds information about the send and receive pending work
 * requests.
 */
struct QueuePair {
  // We consider only write events for now
  // TODO(lume): add recv requests
  std::deque<Wqe> sendQueue;
  uint64_t bytesInFlight;
  std::chrono::time_point<std::chrono::high_resolution_clock> lastPostTs;
  std::chrono::time_point<std::chrono::high_resolution_clock> lastCompletionTs;

  QueuePair();
};

struct LocalRankInfo {
  int localRank{-1};
  int globalRank{-1};
  std::string hostName;
};

struct AlgoContext {
  std::string algorithmName{"unknown"};
  std::vector<size_t> sendMessageSizes;
  std::vector<size_t> recvMessageSizes;
  size_t sendMessageSize{0};
  size_t recvMessageSize{0};
  uint64_t opCount{0};
};

struct ProfileModuleLoggingConfig {
  std::mt19937 mt;

  double getRandomNumber() {
    std::uniform_int_distribution<int> dist(0.0, 1.0);
    return dist(mt);
  }

  bool shouldLogCollective_{false};
  bool shouldLogDeviceTraffic_{false};
};
/*
 * Base class for Ctran profiler modules. Modules are separate objects that
 * perform specialized work on events collected and distributed by the profiler.
 * A module implements the `CtranProfilerModule` class and overrides the
 * functions corresponding to events it is interested in.
 */
class CtranProfilerModule {
 public:
  virtual ~CtranProfilerModule() {}
  virtual void onAlgoStarted(const AlgoContext& context) {}
  virtual void onAlgoCompleted() {}
  virtual void onCtrlReceived(const CtranTransportEvent& event) {}
  virtual void onCtrlComplete(const CtranTransportEvent& event) {}
  virtual void onReadyToSend(const CtranTransportEvent& event) {}
  virtual void onReadyToReceive(const CtranTransportEvent& event) {}
  virtual void onPutIssued(const CtranTransportEvent& event) {}
  virtual void onPutComplete(const CtranTransportEvent& event) {}
  virtual void onRecvStarted(const CtranTransportEvent& event) {}
  virtual void onRecvComplete(const CtranTransportEvent& event) {}
  virtual void onWqeComplete(const Wqe& wqe) {}
  virtual void onBufferRegistrationComplete(
      const CtranRegistrationEvent& event) {}
  virtual void onBufferRegistrationStart(const CtranRegistrationEvent& event) {}
  virtual bool shouldLogForCollective() {
    return true;
  }
};

/*
 * The Ctran profiler collects transport events, builds an internal state based
 * on information from these events, and distributes these events (or new events
 * generated by changes in the internal state) to installed modules.
 */
class CtranProfiler {
 public:
  CtranProfiler(CtranComm* comm);
  CtranProfiler(int rank, const std::string& hostname);
  ~CtranProfiler();
  void handleTransportEvent(const CtranTransportEvent& event);
  void handleRdmaEvent(const CtranRdmaEvent& event);
  void handleRegistrationEvent(const CtranRegistrationEvent& event);
  void handleAlgoStarted(AlgoContext algoContext);
  void handleAlgoCompleted();
  void CtranProfilerInit();
  bool shouldHandleRdmaEvent();
  bool shouldHandleTransportEvent();
  void setLoggingConfig(int opCount);
  void genBufferRegEvent(
      CtranRegistrationEvent::Operation op,
      CtranRegistrationEvent::Type type);

  /**
   * Installs a module of type T.
   */
  template <typename T>
  T* installModule() {
    static_assert(
        std::is_base_of<CtranProfilerModule, T>::value,
        "T must derive from CtranProfilerModule");
    CLOGF_SUBSYS(INFO, INIT, "Adding profiler module {}", typeid(T).name());
    modules_.push_back(std::make_unique<T>(this));
    return dynamic_cast<T*>(modules_.back().get());
  }

  /**
   * Returns a vector of pointers to modules of type T.
   */
  template <typename T>
  std::vector<T*> getModules() {
    CLOGF_SUBSYS(INFO, INIT, "Getting profiler module {}", typeid(T).name());
    static_assert(
        std::is_base_of<CtranProfilerModule, T>::value,
        "T must derive from CtranProfilerModule");

    std::vector<T*> modulePtrList;
    for (const auto& module : modules_) {
      auto modulePtr = dynamic_cast<T*>(module.get());
      if (modulePtr) {
        modulePtrList.emplace_back(modulePtr);
      }
    }
    return modulePtrList;
  }

  void removeAllModules() {
    modules_.clear();
  }

  CtranComm* getComm() const {
    return comm_;
  }

  std::string getAlgorithmName() const {
    return algoContext_.algorithmName;
  }

  size_t getSendMessageSize(int rank) const {
    if (algoContext_.sendMessageSizes.size() == 0) {
      return algoContext_.sendMessageSize;
    }
    if (algoContext_.sendMessageSizes.size() > rank) {
      return algoContext_.sendMessageSizes[rank];
    }
    return 0;
  }

  size_t getRecvMessageSize(int rank) const {
    if (algoContext_.recvMessageSizes.size() == 0) {
      return algoContext_.recvMessageSize;
    }
    if (algoContext_.recvMessageSizes.size() > rank) {
      return algoContext_.recvMessageSizes[rank];
    }
    return 0;
  }

  ProfileModuleLoggingConfig getProfileModuleLoggingConfig() const {
    return profileModuleLoggingConfig_;
  }

 private:
  // profiler modules
  std::vector<std::unique_ptr<CtranProfilerModule>> modules_;
  // pointer to the CTRAN communicator; used to get the rank and stream
  // information
  CtranComm* comm_{nullptr};
  // queues of pending send and receive work requests for each queue pair
  std::unordered_map<uint32_t, QueuePair> pendingWqes_;

  LocalRankInfo localRankInfo_;
  AlgoContext algoContext_;
  ProfileModuleLoggingConfig profileModuleLoggingConfig_;
};
