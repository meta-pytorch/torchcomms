// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <unistd.h>
#include <atomic>
#include <string>
#include <string_view>

#include <fmt/format.h>
#include <folly/Singleton.h>
#include <folly/init/Init.h>
#include <folly/json/dynamic.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "LoggerUtil.h"
#include "ScubaLoggerTestMixin.h"
#include "debug.h" // @manual

#include "comms/testinfra/TestUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/EventMgr.h"
#include "comms/utils/logger/Logger.h"
#include "comms/utils/logger/tests/MockScubaTable.h"

static constexpr std::string_view kHpcJobName = "jobUnitTest";
static constexpr std::string_view kHpcJobVersion = "13";
static constexpr std::string_view kHpcJobAttempt = "55";
static constexpr std::string_view kHpcjobQuorumRestartId = "72";
static const std::string kTestStr = "test1\n";
static constexpr unsigned long long kcommId = 12881726743803;
static constexpr uint64_t kcommHash = 17952292033;
static constexpr int kRank = 2;
static constexpr int kNranks = 8;
static constexpr std::string_view kGlobalRank = "5";
static constexpr std::string_view kWorldSize = "16";

static const char* kcommDesc = "test_pg:20";
static const struct CommLogData kcommLogMetadata =
    CommLogData{kcommId, kcommHash, kcommDesc, kRank, kNranks};
static constexpr std::string_view kScubaTableComm = "nccl_structured_logging";

const static std::vector<std::string> keventStage = {
    "Init START",
    "BootStrap START",
    "BootStrap COMPLETE",
};

class NcclLoggerTestEnv : public ::testing::Environment {
 public:
  void SetUp() override {
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "INIT,COLL", 1);

    // Set up dummy values for environment variables for Scuba test
    setenv("RANK", kGlobalRank.data(), 1);
    setenv("WORLD_SIZE", kWorldSize.data(), 1);
    setenv("HPC_JOB_NAME", kHpcJobName.data(), 1);
    setenv("HPC_JOB_VERSION", kHpcJobVersion.data(), 1);
    setenv("HPC_JOB_ATTEMPT_INDEX", kHpcJobAttempt.data(), 1);
    setenv("TW_RESTART_ID", kHpcjobQuorumRestartId.data(), 1);
    setenv(
        "NCCL_HPC_JOB_IDS",
        "HPC_JOB_NAME,HPC_JOB_VERSION,HPC_JOB_ATTEMPT_INDEX",
        1);
    setenv("NCCL_COMM_EVENT_LOGGING", "pipe:nccl_structured_logging", 1);
    setenv("NCCL_DEBUG_LOGGING_ASYNC", "0", 1);

    initEnv();
    // close logger to force unregistration of folly logger factory
    NcclLogger::close();
  }

  void TearDown() override {}
};

class NcclLoggerTest : public ::testing::Test, public ScubaLoggerTestMixin {
 public:
  NcclLoggerTest() = default;
  void SetUp() override {
    mockPassthru = false;
    this->scubaTableComm = kScubaTableComm;
  }

  void TearDown() override {}

  void finishLogging() {
    NcclLogger::close();
  }

  void initLogging() {
    folly::Singleton<const DataTableAllTables, DataTableAllTablesTag>::
        make_mock([this]() {
          return new DataTableAllTables(createAllMockTables(mockPassthru));
        });
    ncclDebugLevel = -1;
    initNcclLogger();
  }

  void CommEventWriteLog() {
    NcclScubaEvent event(
        std::make_unique<CommEvent>(
            &kcommLogMetadata, keventStage[0], keventStage[0]));
    event.record();

    NcclScubaEvent event1(
        std::make_unique<CommEvent>(
            &kcommLogMetadata, "BootStrap", "BootStrap"));
    event1.startAndRecord();
    event1.stopAndRecord();

    // test logging localRank, localRanks
    NcclScubaEvent event2(
        std::make_unique<CommEvent>(
            &kcommLogMetadata, kRank / 2, kNranks / 2, keventStage[0]));
    event2.record();
  }

  void CommEventReadJson(const std::string& output) {
    std::istringstream iss(output);
    folly::dynamic jsonLog;
    int count = 0;
    std::string line;
    while (std::getline(iss, line)) {
      if (count < 3) {
        jsonLog = folly::parseJson(line);

        EXPECT_EQ(jsonLog["int"]["commId"].asInt(), kcommId);
        EXPECT_EQ(jsonLog["int"]["commHash"].asInt(), kcommHash);
        EXPECT_EQ(jsonLog["int"]["rank"].asInt(), kRank);
        EXPECT_EQ(jsonLog["int"]["nranks"].asInt(), kNranks);
        EXPECT_EQ(
            jsonLog["int"]["globalRank"].asInt(),
            std::stoi(std::string(kGlobalRank)));
        EXPECT_EQ(
            jsonLog["int"]["worldSize"].asInt(),
            std::stoi(std::string(kWorldSize)));
        EXPECT_EQ(
            jsonLog["int"]["jobAttempt"].asInt(),
            std::stoi(std::string(kHpcJobAttempt)));
        EXPECT_EQ(
            jsonLog["int"]["jobVersion"].asInt(),
            std::stoi(std::string(kHpcJobVersion)));
        EXPECT_EQ(
            jsonLog["int"]["jobQuorumRestartId"].asInt(),
            std::stoi(std::string(kHpcjobQuorumRestartId)));

        EXPECT_EQ(jsonLog["normal"]["commDesc"].asString(), kcommDesc);
        EXPECT_EQ(jsonLog["normal"]["stage"].asString(), keventStage[count]);
        EXPECT_EQ(jsonLog["normal"]["jobName"].asString(), kHpcJobName);

        EXPECT_GE(jsonLog["double"]["timerDeltaMs"].asDouble(), 0);
        EXPECT_TRUE(jsonLog["int"].count("time"));

      } else {
        jsonLog = folly::parseJson(line);

        EXPECT_EQ(jsonLog["int"]["commId"].asInt(), kcommId);
        EXPECT_EQ(jsonLog["int"]["commHash"].asInt(), kcommHash);
        EXPECT_EQ(jsonLog["int"]["rank"].asInt(), kRank);
        EXPECT_EQ(jsonLog["int"]["nranks"].asInt(), kNranks);
        EXPECT_EQ(jsonLog["int"]["localRank"].asInt(), kRank / 2);
        EXPECT_EQ(jsonLog["int"]["localRanks"].asInt(), kNranks / 2);
        EXPECT_EQ(
            jsonLog["int"]["globalRank"].asInt(),
            std::stoi(std::string(kGlobalRank)));
        EXPECT_EQ(
            jsonLog["int"]["worldSize"].asInt(),
            std::stoi(std::string(kWorldSize)));
        EXPECT_EQ(
            jsonLog["int"]["jobAttempt"].asInt(),
            std::stoi(std::string(kHpcJobAttempt)));
        EXPECT_EQ(
            jsonLog["int"]["jobVersion"].asInt(),
            std::stoi(std::string(kHpcJobVersion)));
        EXPECT_EQ(
            jsonLog["int"]["jobQuorumRestartId"].asInt(),
            std::stoi(std::string(kHpcjobQuorumRestartId)));

        EXPECT_EQ(jsonLog["normal"]["commDesc"].asString(), kcommDesc);
        EXPECT_EQ(jsonLog["normal"]["stage"].asString(), keventStage[0]);
        EXPECT_EQ(jsonLog["normal"]["jobName"].asString(), kHpcJobName);
      }

      ++count;
    }

    EXPECT_EQ(count, 4);
  }

 protected:
  std::string scubaTableComm;
  bool mockPassthru{false};
};

//
//  Event logging general tests
//

TEST_F(NcclLoggerTest, EventStdout) {
  // no stdout event logging
  initLogging();
  testing::internal::CaptureStdout();
  NcclScubaEvent event(
      std::make_unique<CommEvent>(
          &kcommLogMetadata, "CommInit START", "CommInitRank"));
  event.record();
  finishLogging();

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::Not(testing::HasSubstr("CommInitRank")));
}

//
//  Communicator Event logging tests
//

TEST_F(NcclLoggerTest, CommEventPipe) {
  folly::test::TemporaryDirectory tmpDir;
  auto scubaLogDirGuard =
      EnvRAII(NCCL_SCUBA_LOG_FILE_PREFIX, tmpDir.path().string());
  auto eventGuard = EnvRAII(
      NCCL_COMM_EVENT_LOGGING, std::string("pipe:nccl_structured_logging"));
  initLogging();
  CommEventWriteLog();
  finishLogging();

  auto output = getMessages(LoggerEventType::CommEventType);
  CommEventReadJson(output);
}

TEST_F(NcclLoggerTest, CommEventScuba) {
  auto eventGuard = EnvRAII(
      NCCL_COMM_EVENT_LOGGING, std::string("scuba:nccl_structured_logging"));
  initLogging();
  CommEventWriteLog();
  finishLogging();

  EXPECT_TRUE(getWriteCount(LoggerEventType::CommEventType) == 4);
  CommEventReadJson(getMessages(LoggerEventType::CommEventType));
}

TEST_F(NcclLoggerTest, WarnLogTest) {
  testing::internal::CaptureStdout();
  initLogging();
  std::string TestStr = "TESTING";
  WARN("%s", TestStr.c_str());
  finishLogging();

  std::string output = testing::internal::GetCapturedStdout();
  std::cout << "output: " << output << std::endl;
  EXPECT_THAT(output, testing::HasSubstr(TestStr));
  EXPECT_THAT(output, testing::HasSubstr("NCCL WARN"));
}

TEST_F(NcclLoggerTest, InfoLogTest) {
  testing::internal::CaptureStdout();
  initLogging();
  std::string TestStr = "TESTING";
  INFO(NCCL_ALL, "%s", TestStr.c_str());
  finishLogging();

  std::string output = testing::internal::GetCapturedStdout();
  std::cout << "output: " << output << std::endl;
  EXPECT_THAT(output, testing::HasSubstr(TestStr));
  EXPECT_THAT(output, testing::HasSubstr("NCCL INFO"));
}

TEST_F(NcclLoggerTest, IsScubaTest) {
  folly::test::TemporaryDirectory tmpDir;
  auto scubaLogDirGuard =
      EnvRAII(NCCL_SCUBA_LOG_FILE_PREFIX, tmpDir.path().string());
  std::string table_env = "scuba:" + this->scubaTableComm;
  auto eventGuard = EnvRAII(NCCL_COMM_EVENT_LOGGING, table_env);
  initLogging();
  CommEventWriteLog();

  std::string logFile{};
  try {
    logFile = getCommEventScubaFile();
  } catch (std::exception&) {
  }
  EXPECT_TRUE(logFile.empty());
  finishLogging();
}

TEST_F(NcclLoggerTest, IsPipeTest) {
  folly::test::TemporaryDirectory tmpDir;
  auto scubaLogDirGuard =
      EnvRAII(NCCL_SCUBA_LOG_FILE_PREFIX, tmpDir.path().string());
  auto eventGuard =
      EnvRAII(NCCL_COMM_EVENT_LOGGING, std::string("pipe:comm_table_test"));
  initLogging();
  CommEventWriteLog();
  std::string logFile{};
  try {
    logFile = getCommEventScubaFile("comm_table_test");
  } catch (std::exception&) {
  }
  EXPECT_TRUE(!logFile.empty());
  finishLogging();
}

TEST_F(NcclLoggerTest, CommConcurrentLog) {
  folly::test::TemporaryDirectory tmpDir;
  auto scubaLogDirGuard =
      EnvRAII(NCCL_SCUBA_LOG_FILE_PREFIX, tmpDir.path().string());
  auto eventGuard = EnvRAII(
      NCCL_COMM_EVENT_LOGGING, std::string("pipe:nccl_structured_logging"));
  mockPassthru = true;
  initLogging();

  int numThreads = 10;
  std::atomic<bool> run_threads(true);
  std::vector<std::thread> threads(numThreads);

  std::cout << "Starting concurrent threads to log\n";

  for (int i = 0; i < numThreads; ++i) {
    threads.emplace_back([&run_threads, i]() {
      int stageID = 0;
      while (run_threads) {
        auto stage =
            "Bootstrap_" + std::to_string(i) + "_" + std::to_string(stageID);

        NcclScubaEvent event(
            std::make_unique<CommEvent>(
                &kcommLogMetadata, keventStage[0], keventStage[0]));
        event.record();

        NcclScubaEvent event1(
            std::make_unique<CommEvent>(&kcommLogMetadata, "Init", "Init"));
        event1.startAndRecord();
        event1.stopAndRecord();
        ++stageID;
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
  LOG(INFO) << "Finished concurrent threads to log\n";
  finishLogging();
}

// Equivalent to recordStart/End with a logger event.
// NcclLogger::recordStart(
//     std::make_unique<CommEvent>(
//         &comm->logMetaData,
//         std::string(kncclCommFinalize) + " START",
//         ""),
//     getThreadUniqueId(kncclCommFinalize.data()));
//
// code to measusre
//
// NcclLogger::recordEnd(
// std::make_unique<CommEvent>(
//     &comm->logMetaData,
//     std::string(kncclCommFinalize) + " COMPLETE",
//     ""),
// getThreadUniqueId(kncclCommFinalize.data()));
TEST_F(NcclLoggerTest, NcclScubaEventRecordStartEnd) {
  folly::test::TemporaryDirectory tmpDir;
  auto scubaLogDirGuard =
      EnvRAII(NCCL_SCUBA_LOG_FILE_PREFIX, tmpDir.path().string());
  auto eventGuard = EnvRAII(
      NCCL_COMM_EVENT_LOGGING, std::string("pipe:nccl_structured_logging"));
  initLogging();

  // recordStart/recordEnd
  NcclScubaEvent event1(
      std::make_unique<CommEvent>(
          &kcommLogMetadata, keventStage[0], keventStage[0]));
  event1.startAndRecord();
  // code to measure
  sleep(1);
  event1.stopAndRecord();
  finishLogging();

  auto output = getMessages(LoggerEventType::CommEventType);
  std::istringstream iss(output);
  std::string line;
  folly::dynamic jsonLogStart, jsonLogEnd;
  std::getline(iss, line);
  jsonLogStart = folly::parseJson(line);
  EXPECT_EQ(
      jsonLogStart["normal"]["stage"].asString(), keventStage[0] + " START");
  std::getline(iss, line);
  jsonLogEnd = folly::parseJson(line);
  EXPECT_EQ(
      jsonLogEnd["normal"]["stage"].asString(), keventStage[0] + " COMPLETE");
  EXPECT_GE(jsonLogEnd["double"]["timerDeltaMs"].asDouble(), 1000);
}

// Equivalent to simple record with logger event.
TEST_F(NcclLoggerTest, NcclScubaEventRecord) {
  folly::test::TemporaryDirectory tmpDir;
  auto scubaLogDirGuard =
      EnvRAII(NCCL_SCUBA_LOG_FILE_PREFIX, tmpDir.path().string());
  auto eventGuard = EnvRAII(
      NCCL_COMM_EVENT_LOGGING, std::string("pipe:nccl_structured_logging"));
  initLogging();
  // simple record
  auto commEvent2 = std::make_unique<CommEvent>(
      &kcommLogMetadata, keventStage[0], keventStage[0]);
  NcclScubaEvent event2(std::move(commEvent2));
  event2.record();
  finishLogging();

  auto output = getMessages(LoggerEventType::CommEventType);
  std::istringstream iss(output);
  std::string line;
  folly::dynamic jsonLog;
  std::getline(iss, line);
  jsonLog = folly::parseJson(line);
  EXPECT_EQ(jsonLog["normal"]["stage"].asString(), keventStage[0]);
}

// Equivalent to record with multiple stages.
// timer = timer.start();
// record("stage0", timer.lap().count());
// record("stage1", timer.lap().count());
TEST_F(NcclLoggerTest, NcclScubaEventRecordAndLap) {
  folly::test::TemporaryDirectory tmpDir;
  auto scubaLogDirGuard =
      EnvRAII(NCCL_SCUBA_LOG_FILE_PREFIX, tmpDir.path().string());
  auto eventGuard = EnvRAII(
      NCCL_COMM_EVENT_LOGGING, std::string("pipe:nccl_structured_logging"));
  initLogging();

  NcclScubaEvent event3(&kcommLogMetadata);
  // code1 to measure
  sleep(1);
  event3.lapAndRecord("stage1");
  // code2 to measure
  sleep(1);
  event3.lapAndRecord("stage2");

  finishLogging();

  auto output = getMessages(LoggerEventType::CommEventType);
  std::istringstream iss(output);
  std::string line;
  folly::dynamic jsonLog;
  std::getline(iss, line);
  jsonLog = folly::parseJson(line);
  EXPECT_EQ(jsonLog["normal"]["stage"].asString(), "stage1");
  EXPECT_GE(jsonLog["double"]["timerDeltaMs"].asDouble(), 1000);
  std::getline(iss, line);
  jsonLog = folly::parseJson(line);
  EXPECT_EQ(jsonLog["normal"]["stage"].asString(), "stage2");
  EXPECT_GE(jsonLog["double"]["timerDeltaMs"].asDouble(), 1000);
}

TEST_F(NcclLoggerTest, NcclScubaEventRecordStage) {
  folly::test::TemporaryDirectory tmpDir;
  auto scubaLogDirGuard =
      EnvRAII(NCCL_SCUBA_LOG_FILE_PREFIX, tmpDir.path().string());
  auto eventGuard = EnvRAII(
      NCCL_COMM_EVENT_LOGGING, std::string("pipe:nccl_structured_logging"));
  initLogging();
  // Record something without any communicator context
  NcclScubaEvent event("dummy stage");
  event.record();
  finishLogging();

  auto output = getMessages(LoggerEventType::CommEventType);
  std::istringstream iss(output);
  std::string line;
  folly::dynamic jsonLog;
  std::getline(iss, line);
  jsonLog = folly::parseJson(line);
  EXPECT_EQ(jsonLog["normal"]["stage"].asString(), "dummy stage");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(new NcclLoggerTestEnv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
