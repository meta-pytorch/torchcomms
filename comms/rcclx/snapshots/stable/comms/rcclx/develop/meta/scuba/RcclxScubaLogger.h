#pragma once
#include <cuda_runtime.h>
#include <folly/Singleton.h>
#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <folly/logging/xlog.h>
#include <folly/synchronization/HazptrThreadPoolExecutor.h>
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/EventsScubaUtil.h"
#include "comms/utils/logger/Logger.h"
#include "comms/utils/logger/LoggingFormat.h"
class RcclxScubaLogger {
 public:
  // Singleton access
  static RcclxScubaLogger& getInstance() {
    static RcclxScubaLogger instance;
    return instance;
  }

  RcclxScubaLogger(const RcclxScubaLogger&) = delete;
  RcclxScubaLogger& operator=(const RcclxScubaLogger&) = delete;

  NcclScubaEvent& GetCommInitFuncEvent() {
    if (!_initialized) {
      throw std::runtime_error(
          "InitializeInitScubaEvents needs to be called first");
    }
    return *_commInitFuncEvent;
  }

  NcclScubaEvent& GetInitBootstrapEvent() {
    if (!_initialized) {
      throw std::runtime_error(
          "InitializeInitScubaEvents needs to be called first");
    }
    return *_initBootstrapEvent;
  }

  NcclScubaEvent& GetInitTransportsRankEvent() {
    if (!_initialized) {
      throw std::runtime_error(
          "InitializeInitScubaEvents needs to be called first");
    }
    return *_initTransportsRankEvent;
  }

  NcclScubaEvent& GetCommInitRankConfigEvent() {
    if (!_initialized) {
      throw std::runtime_error(
          "InitializeInitScubaEvents needs to be called first");
    }
    return *_commInitRankConfigEvent;
  }

  NcclScubaEvent& GetCommFinalizeEvent() {
    if (!_initialized) {
      throw std::runtime_error(
          "InitializeInitScubaEvents needs to be called first");
    }
    return *_commFinalizeEvent;
  }

  NcclScubaEvent& GetCommDestroyEvent() {
    if (!_initialized) {
      throw std::runtime_error(
          "InitializeInitScubaEvents needs to be called first");
    }
    return *_commDestroyEvent;
  }

  NcclScubaEvent& GetCommAbortEvent() {
    if (!_initialized) {
      throw std::runtime_error(
          "InitializeInitScubaEvents needs to be called first");
    }
    return *_commAbortEvent;
  }

  NcclScubaEvent& GetCommSplitEvent() {
    if (!_initialized) {
      throw std::runtime_error(
          "InitializeInitScubaEvents needs to be called first");
    }
    return *_commSplitEvent;
  }

  NcclScubaEvent& GetP2pPreconnectEvent() {
    if (!_initialized) {
      throw std::runtime_error(
          "InitializeInitScubaEvents needs to be called first");
    }
    return *_p2pPreconnectEvent;
  }

  NcclScubaEvent& GetCollPreconnectEvent() {
    if (!_initialized) {
      throw std::runtime_error(
          "InitializeInitScubaEvents needs to be called first");
    }
    return *_collPreconnectEvent;
  }

  void InitializeInitScubaEvents(
      uint64_t commIdHash,
      uint64_t commHash,
      std::string commDesc,
      int rank,
      int nRanks);

  bool isInitialized() const {
    return _initialized;
  }

 private:
  RcclxScubaLogger();
  ~RcclxScubaLogger() {}

  bool _initialized = false;
  std::unique_ptr<NcclScubaEvent> _commInitFuncEvent;
  std::unique_ptr<NcclScubaEvent> _initBootstrapEvent;
  std::unique_ptr<NcclScubaEvent> _initTransportsRankEvent;
  std::unique_ptr<NcclScubaEvent> _commInitRankConfigEvent;
  std::unique_ptr<NcclScubaEvent> _commFinalizeEvent;
  std::unique_ptr<NcclScubaEvent> _commDestroyEvent;
  std::unique_ptr<NcclScubaEvent> _commAbortEvent;
  std::unique_ptr<NcclScubaEvent> _commSplitEvent;
  std::unique_ptr<NcclScubaEvent> _p2pPreconnectEvent;
  std::unique_ptr<NcclScubaEvent> _collPreconnectEvent;
};

static void initFolly() {
  // Adapted from folly/init/Init.cpp
  // We can't use folly::init directly because:
  // - we don't have gflags
  // - xlformers already initialized the signal handler

  // Move from the registration phase to the "you can actually instantiate
  // things now" phase.
  folly::SingletonVault::singleton()->registrationComplete();

  auto const follyLoggingEnv = std::getenv(folly::kLoggingEnvVarName);
  auto const follyLoggingEnvOr = follyLoggingEnv ? follyLoggingEnv : "";
  folly::initLoggingOrDie(follyLoggingEnvOr);

  // Set the default hazard pointer domain to use a thread pool executor
  // for asynchronous reclamation
  folly::enable_hazptr_thread_pool_executor();
}

static void initNcclLogger() {
  NcclLogger::init(
      NcclLoggerInitConfig{
          .contextName = "comms.rcclx",
          .logPrefix = "RCCL",
          .logFilePath =
              meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
          .logLevel = meta::comms::logger::loggerLevelToFollyLogLevel(
              meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)),
          .threadContextFn = []() {
            int cudaDev = -1;
            cudaGetDevice(&cudaDev);
            return cudaDev;
          }});
  // Init logging for NCCL header inside meta directory.
  // This is due to the buck2 behavior of copying the header files to the
  // buck-out directory.
  // For logging in src/include headers, they are using NCCL logging
  // (INFO/WARN/ERROR) which will inherit the loggging category from debug.cc
  NcclLogger::init(
      NcclLoggerInitConfig{
          .contextName = "meta",
          .logPrefix = "RCCL",
          .logFilePath =
              meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
          .logLevel = meta::comms::logger::loggerLevelToFollyLogLevel(
              meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)),
          .threadContextFn = []() {
            int cudaDev = -1;
            cudaGetDevice(&cudaDev);
            return cudaDev;
          }});
}
