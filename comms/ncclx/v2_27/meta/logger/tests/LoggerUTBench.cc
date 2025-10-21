// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/String.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <atomic>
#include <string>

#include "LoggerUtil.h"
#include "ScubaLoggerTestMixin.h"
#include "debug.h" // @manual

#include "comms/testinfra/TestUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/EventMgr.h"
#include "comms/utils/logger/Logger.h"

static constexpr int kGlobalRank = 5;
static constexpr int kNranks = 16;
static const struct CommLogData kcommLogMetadata = CommLogData{
    12881726743803,
    17952292033,
    "logger_bench_ut_pg",
    kGlobalRank,
    kNranks};

const static std::vector<std::string> keventStage = {
    "Init START",
    "Bootstrap start",
    "Bootstrap Complete"};

static std::string scubaLogFile{};

class NcclLoggerBenchEnv : public testing::Environment,
                           public ScubaLoggerTestMixin {
 public:
  void SetUp() override {
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "INIT,COLL", 1);
    setenv("RANK", std::to_string(kGlobalRank).c_str(), 1);
    setenv("WORLD_SIZE", std::to_string(kNranks).c_str(), 1);
    setenv("HPC_JOB_NAME", "jobUnitTest", 1);
    setenv("HPC_JOB_VERSION", "12", 1);
    setenv("HPC_JOB_ATTEMPT_INDEX", "33", 1);
    setenv("TW_RESTART_ID", "10", 1);
    setenv(
        "NCCL_HPC_JOB_IDS",
        "HPC_JOB_NAME,HPC_JOB_VERSION,HPC_JOB_ATTEMPT_INDEX",
        1);
    setenv("NCCL_LOGGER_MODE", "async", 1);
    setenv("NCCL_COMM_EVENT_LOGGING", "pipe:nccl_structured_logging", 1);

    // Set up dummy values for environment variables for Scuba test and also
    // call initEnv.
    ScubaLoggerTestMixin::SetUp();
    // close logger to force unregistration of folly logger factory
    NcclLogger::close();
  }

  void TearDown() override {}
};

class NcclLoggerBenchTest : public ::testing::Test {
 public:
  enum class LogType { SCUBA, FILE };

  NcclLoggerBenchTest() = default;
  void SetUp() override {
    totalRecords = 0;
    logTmpFile = std::make_unique<folly::test::TemporaryFile>();
  }

  void TearDown() override {
    auto logBytesEnd = getLogFileSize();
    auto loggedBytes = getLogFileSize() - logBytesStart;
    LOG(INFO) << "====== Total records ("
              << (logType == LogType::SCUBA ? "scuba" : "file")
              << "): " << totalRecords << " ======";
    LOG(INFO) << "====== Bytes logged: "
              << folly::prettyPrint(
                     loggedBytes, folly::PrettyType::PRETTY_BYTES)
              << " ======";
    if (logType == LogType::SCUBA) {
      // this file is not closed, so we need to keep trace of size manually.
      logBytesStart = logBytesEnd;
    }
  }

  std::string getTmpLogFile() {
    return logTmpFile->path().string();
  }

  void finishLogging() {
    NcclLogger::close();
  }

  void setLogType(LogType type) {
    this->logType = type;
  }

  size_t getLogFileSize() {
    if (logType == LogType::SCUBA) {
      return std::filesystem::file_size(getCommEventScubaFile());
    } else {
      return std::filesystem::file_size(getTmpLogFile());
    }
  }

  void ncclLoggerBenchTest(bool useScubaEvent = false) {
    int numThreads = 100;
    std::atomic<bool> run_threads(true);

    std::vector<std::thread> threads;
    std::cout << "Starting concurrent threads to log\n";

    for (int i = 0; i < numThreads; ++i) {
      threads.emplace_back([this, &run_threads, i, useScubaEvent]() {
        int stageID = 0;
        while (run_threads) {
          if (logType == LogType::SCUBA) {
            auto stage = "Bootstrap_" + std::to_string(i) + "_" +
                std::to_string(stageID);

            if (useScubaEvent) {
              NcclScubaEvent event1(std::make_unique<CommEvent>(
                  &kcommLogMetadata, keventStage[0], keventStage[0]));
              event1.record();

              NcclScubaEvent event2(std::make_unique<CommEvent>(
                  &kcommLogMetadata, keventStage[1], keventStage[1]));
              event2.startAndRecord();
              event2.stopAndRecord();

            } else {
              NcclScubaEvent event(std::make_unique<CommEvent>(
                  &kcommLogMetadata, keventStage[0], keventStage[0]));
              event.record();

              NcclScubaEvent event1(std::make_unique<CommEvent>(
                  &kcommLogMetadata, keventStage[1], keventStage[1]));
              event1.startAndRecord();
              event1.stopAndRecord();
            }
          } else {
            std::string dummyLog = "Dummy log " + std::to_string(i) + "_" +
                std::to_string(stageID);
            INFO(NCCL_INIT, "%s", dummyLog.c_str());
          }
          ++stageID;
          ++totalRecords;
        }
      });
    }

    std::this_thread::sleep_for(std::chrono::seconds(5));
    run_threads = false;
    for (auto& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

 private:
  size_t logBytesStart{0};
  std::atomic<uint64_t> totalRecords{0};
  std::unique_ptr<folly::test::TemporaryFile> logTmpFile;
  LogType logType{LogType::SCUBA};
};

TEST_F(NcclLoggerBenchTest, CommBenchScubaLog) {
  setLogType(LogType::SCUBA);
  initNcclLogger();
  ncclLoggerBenchTest();
  finishLogging();
}

TEST_F(NcclLoggerBenchTest, CommBenchScubaLogWithEventApi) {
  setLogType(LogType::SCUBA);
  initNcclLogger();
  ncclLoggerBenchTest(true);
  finishLogging();
}

TEST_F(NcclLoggerBenchTest, CommBenchDebugLog) {
  folly::test::TemporaryFile tmpFile;
  setLogType(LogType::FILE);
  EnvRAII env(NCCL_DEBUG_FILE, getTmpLogFile());
  // Reset ncclDebugLevel to force debug sub-system to be re-initialized
  ncclDebugLevel = -1;
  initNcclLogger();
  ncclLoggerBenchTest();
  finishLogging();
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(new NcclLoggerBenchEnv);
  return RUN_ALL_TESTS();
}
