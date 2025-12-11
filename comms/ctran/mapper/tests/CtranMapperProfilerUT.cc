// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <filesystem>

#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperRegMem.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#include "comms/ctran/utils/Utils.h"

class CtranMapperProfilerTest : public ::testing::Test {
 public:
  std::unique_ptr<TestCtranCommRAII> dummyCommRAII;
  double expectedDurMS;
  CtranMapperProfilerTest() = default;

 protected:
  void SetUp() override {
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "INIT", 1);
    setenv("NCCL_IGNORE_TOPO_LOAD_FAILURE", "true", 1);
    ncclCvarInit();

    // A random duration between 0-5ms to test the timer
    srand(time(NULL));
    expectedDurMS = rand() % 5 + 1;
  }
  void TearDown() override {}
};

TEST_F(CtranMapperProfilerTest, Timer) {
  auto timer = std::unique_ptr<CtranMapperTimer>(new CtranMapperTimer());
  EXPECT_THAT(timer, testing::NotNull());

  usleep(expectedDurMS * 1000);

  double durMs = timer->durationMs();
  EXPECT_GE(durMs, expectedDurMS);

  double durUs = timer->durationUs();
  EXPECT_GE(durUs, expectedDurMS * 1000);
}

TEST_F(CtranMapperProfilerTest, TimestampPoint) {
  int peer = 0;
  auto tp = std::unique_ptr<CtranMapperTimestampPoint>(
      new CtranMapperTimestampPoint(peer));
  EXPECT_THAT(tp, testing::NotNull());
}

TEST_F(CtranMapperProfilerTest, Timestamp) {
  auto dummyAlgo = "Ring";
  auto ts = std::unique_ptr<CtranMapperTimestamp>(
      new CtranMapperTimestamp(dummyAlgo));
  EXPECT_THAT(ts, testing::NotNull());
}

TEST_F(CtranMapperProfilerTest, TimestampInsert) {
  auto dummyAlgo = "Ring";
  auto ts = std::unique_ptr<CtranMapperTimestamp>(
      new CtranMapperTimestamp(dummyAlgo));
  EXPECT_THAT(ts, testing::NotNull());

  auto begin = std::chrono::high_resolution_clock::now();

  usleep(expectedDurMS * 1000);
  ts->recvCtrl.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putIssued.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putComplete.push_back(CtranMapperTimestampPoint(0));

  double durMs = 0.0;
  durMs = std::chrono::duration_cast<std::chrono::milliseconds>(
              ts->recvCtrl[0].now.time_since_epoch() - begin.time_since_epoch())
              .count();
  // recvCtrl should take >= `expectedDurMS` ms
  EXPECT_GE(durMs, expectedDurMS);

  durMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          ts->putIssued[0].now.time_since_epoch() - begin.time_since_epoch())
          .count();
  // putIssued should take >= 2 * `expectedDurMS` ms
  EXPECT_GE(durMs, expectedDurMS * 2);

  durMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          ts->putComplete[0].now.time_since_epoch() - begin.time_since_epoch())
          .count();
  // putComplete should take >= 3 * `expectedDurMS` ms
  EXPECT_GE(durMs, expectedDurMS * 3);
}

TEST_F(CtranMapperProfilerTest, MapperFlushTimerStdout) {
  setenv("NCCL_CTRAN_PROFILING", "stdout", 1);
  ncclCvarInit();
  dummyCommRAII = createDummyCtranComm();

  auto dummyAlgo = "Ring";
  auto mapper = std::make_unique<CtranMapper>(dummyCommRAII->ctranComm.get());
  EXPECT_THAT(mapper, testing::NotNull());

  auto ts = std::unique_ptr<CtranMapperTimestamp>(
      new CtranMapperTimestamp(dummyAlgo));
  EXPECT_THAT(ts, testing::NotNull());

  usleep(expectedDurMS * 1000);
  ts->recvCtrl.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putIssued.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putComplete.push_back(CtranMapperTimestampPoint(0));

  mapper->timestamps.push_back(std::move(ts));

  auto kExpectedOutput1 = "Communication Profiling";
  auto kExpectedOutput2 = dummyAlgo;
  testing::internal::CaptureStdout();

  mapper->reportProfiling(true);

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput1));
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput2));
  mapper.reset();
}

// NOTE: this test is disabled by default since NCCL INFO log does not seem
// printing within the captured window reliably. Need to investigate further.
TEST_F(CtranMapperProfilerTest, DISABLED_MapperFlushTimerInfo) {
  setenv("NCCL_CTRAN_PROFILING", "info", 1);
  ncclCvarInit();
  dummyCommRAII = createDummyCtranComm();

  auto dummyAlgo = "Ring";
  auto mapper = std::make_unique<CtranMapper>(dummyCommRAII->ctranComm.get());
  EXPECT_THAT(mapper, testing::NotNull());

  auto ts = std::unique_ptr<CtranMapperTimestamp>(
      new CtranMapperTimestamp(dummyAlgo));
  EXPECT_THAT(ts, testing::NotNull());

  usleep(expectedDurMS * 1000);
  ts->recvCtrl.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putIssued.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putComplete.push_back(CtranMapperTimestampPoint(0));

  mapper->timestamps.push_back(std::move(ts));

  auto kExpectedOutput1 = "NCCL INFO";
  auto kExpectedOutput2 = "Communication Profiling";
  auto kExpectedOutput3 = dummyAlgo;
  testing::internal::CaptureStdout();

  mapper->reportProfiling(true);
  // Ensure the output is flushed before check
  fflush(stdout);

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput1));
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput2));
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput3));
  mapper.reset();
}

TEST_F(CtranMapperProfilerTest, MapperFlushTimerKineto) {
  auto outputDir = "/tmp";
  auto pid = getpid();
  // NOTE: this is default prefix of output name, need to be consistent with the
  // code in CtranMapper.cc
  auto prefix = "nccl_ctran_log." + std::to_string(pid);
  setenv("NCCL_CTRAN_PROFILING", "kineto", 1);
  setenv("NCCL_CTRAN_KINETO_PROFILE_DIR", outputDir, 1);
  ncclCvarInit();
  dummyCommRAII = createDummyCtranComm();

  auto dummyAlgo = "Ring";
  auto mapper = std::make_unique<CtranMapper>(dummyCommRAII->ctranComm.get());
  EXPECT_THAT(mapper, testing::NotNull());

  auto ts = std::unique_ptr<CtranMapperTimestamp>(
      new CtranMapperTimestamp(dummyAlgo));
  EXPECT_THAT(ts, testing::NotNull());

  usleep(expectedDurMS * 1000);
  ts->recvCtrl.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putIssued.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putComplete.push_back(CtranMapperTimestampPoint(0));

  mapper->timestamps.push_back(std::move(ts));

  mapper->reportProfiling(true);

  bool foundFile = false;
  for (auto& entry : std::filesystem::directory_iterator(outputDir)) {
    if (entry.path().has_filename() &&
        entry.path().filename().string().rfind(prefix, 0) == 0) {
      foundFile = true;
      // NOTE: uncomment below to check the content of the file
      /*
      std::ifstream file(entry.path());
      std::string contents;
      if (file.is_open()) {
        file >> contents;
        std::cout << "Contents of " << entry.path() << ": " << contents
                  << std::endl;
        file.close();
      } else {
        std::cout << "Failed to open " << entry.path() << std::endl;
      }
      */
    }
  }
  EXPECT_TRUE(foundFile);
  mapper.reset();
}

TEST_F(CtranMapperProfilerTest, regSnapshot) {
  constexpr int numComms = 10;
  std::vector<std::unique_ptr<TestCtranCommRAII>> comms(numComms);
  std::vector<std::unique_ptr<CtranMapper>> mappers(numComms);

  for (int i = 0; i < numComms; ++i) {
    comms[i] = createDummyCtranComm();
    mappers[i] = std::make_unique<CtranMapper>(comms[i]->ctranComm.get());
    EXPECT_THAT(mappers[i], testing::NotNull());
  }

  // Record registration after communicator initialization to skip any internal
  // buffer registration
  EnvRAII env(NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT, 0);

  const size_t bufSize = 8192;
  void* repeatedBuf = nullptr;
  void* repeatHdl = nullptr;
  CUDACHECK_TEST(cudaMalloc(&repeatedBuf, bufSize));

  // Register the repeatedBuf
  EXPECT_EQ(mappers[0]->regMem(repeatedBuf, bufSize, &repeatHdl), commSuccess);

  void* bufs[numComms];
  for (int i = 0; i < numComms; ++i) {
    CUDACHECK_TEST(cudaMalloc(&bufs[i], bufSize));
  }

  // Stress regMem by multiple threads, each registeres via a different mapper
  // and communicator
  std::vector<std::thread> threads;
  for (int i = 0; i < numComms; i++) {
    std::thread t(
        [&](int tid) {
          // Help label in NCCL logging
          std::string threadName = "TestThread" + std::to_string(tid);
          ctran::commSetMyThreadLoggingName(threadName.c_str());

          auto& mapper = mappers[tid];
          void *hdl1 = nullptr, *hdl2 = nullptr;
          // Register a local buffer per thread
          EXPECT_EQ(
              mapper->regMem(bufs[tid], bufSize, &hdl1, true), commSuccess);
          EXPECT_THAT(hdl1, testing::NotNull());

          // Register the repeatedBuf, expect to return the same handle
          EXPECT_EQ(
              mapper->regMem(repeatedBuf, bufSize, &hdl2, true), commSuccess);
          EXPECT_EQ(hdl2, repeatHdl);

          // Release local registration handles
          EXPECT_EQ(mapper->deregMem(hdl1), commSuccess);
          EXPECT_EQ(mapper->deregMem(hdl2), commSuccess);
        },
        i);
    threads.push_back(std::move(t));
  }

  for (auto& t : threads) {
    t.join();
  }

  // Last deregMem should release the registration of repeatedBuf
  EXPECT_EQ(mappers[0]->deregMem(repeatHdl), commSuccess);

  for (int i = 0; i < numComms; ++i) {
    CUDACHECK_TEST(cudaFree(bufs[i]));
  }
  CUDACHECK_TEST(cudaFree(repeatedBuf));

  auto regCache = CtranMapperRegCache::getInstance();
  CHECK_VALID_REGCACHE(regCache);

  regCache->profiler.rlock()->reportSnapshot();

  auto snapshot = regCache->profiler.rlock()->getSnapshot();
  EXPECT_EQ(snapshot.currentNumCache, 0);
  EXPECT_EQ(snapshot.currentNumReg, 0);

  EXPECT_EQ(snapshot.totalNumCache, numComms + 1);
  EXPECT_EQ(snapshot.totalNumReg, numComms + 1);
  EXPECT_EQ(snapshot.totalNumDereg, numComms + 1);
  EXPECT_EQ(snapshot.totalNumDynamicReg, 0);
}
