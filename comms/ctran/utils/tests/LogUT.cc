// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <sys/syscall.h>
#include <unistd.h>

#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <fmt/core.h>
#include <fmt/std.h>
#include <gtest/gtest.h>

#include <gmock/gmock.h>
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/utils/CtranLogUtils.h"
#include "comms/ctran/utils/LogInit.h"
#include "comms/ctran/utils/Utils.h"
#include "comms/testinfra/TestXPlatUtils.h"

class CtranUtilsLogTest : public ::testing::Test {
 public:
  CtranUtilsLogTest() = default;

  void SetUp() override {
    ctran::logging::initCtranLogging(true /*alwaysInit*/);
    auto& logger =
        meta::comms::logger::getSpdlogLogger(ctran::logging::kCtranLoggerName);
    logger.configure(
        "CTRAN",
        []() {
          int cudaDev = -1;
          (void)cudaGetDevice(&cudaDev);
          return cudaDev;
        },
        [](std::string_view message) {
          meta::comms::logger::setLastError(std::string{message}, {});
        },
        false);
    logger.configureOutput("");
    logger.set_level(spdlog::level::info);
  }

  void TearDown() override {
    meta::comms::logger::getSpdlogLogger(ctran::logging::kCtranLoggerName)
        .flush();
  }

  int getCurrentGpuIndex() {
    int gpuIndex = -1;
    CUDACHECK_TEST(cudaGetDevice(&gpuIndex));
    return gpuIndex;
  }
};

TEST_F(CtranUtilsLogTest, TestCtranLogFormat) {
  testing::internal::CaptureStdout();
  CTRAN_LOG(INFO, "Test message with value: {}", 42);
  meta::comms::logger::getSpdlogLogger(ctran::logging::kCtranLoggerName)
      .flush();
  const auto output = testing::internal::GetCapturedStdout();

  EXPECT_THAT(
      output, ::testing::HasSubstr(fmt::format("[{}]", getCurrentGpuIndex())));
  EXPECT_THAT(output, testing::HasSubstr("CTRAN INFO"));
  EXPECT_THAT(output, testing::HasSubstr("Test message with value: 42"));
}

TEST_F(CtranUtilsLogTest, TestCtranLoggerPreservesLastError) {
  auto& logger =
      meta::comms::logger::getSpdlogLogger(ctran::logging::kCtranLoggerName);
  EXPECT_EQ(logger.name(), ctran::logging::kCtranLoggerName);
  logger.set_level(spdlog::level::info);

  CTRAN_LOG(ERR, "Spdlog CTRAN error {}", 42);

  EXPECT_THAT(
      meta::comms::logger::getLastCommsError(),
      testing::HasSubstr("Spdlog CTRAN error 42"));
}

TEST_F(CtranUtilsLogTest, TestCtranErrEvaluatesArgumentsOnce) {
  auto& logger =
      meta::comms::logger::getSpdlogLogger(ctran::logging::kCtranLoggerName);
  logger.set_level(spdlog::level::err);

  int evaluated = 0;
  CTRAN_ERR(commInternalError, "CTRAN error {}", ++evaluated);

  EXPECT_EQ(evaluated, 1);
  EXPECT_THAT(
      meta::comms::logger::getLastCommsError(),
      testing::HasSubstr("CTRAN error 1"));
}

TEST_F(CtranUtilsLogTest, TestCtranDbg5UsesTraceLevel) {
  auto& logger =
      meta::comms::logger::getSpdlogLogger(ctran::logging::kCtranLoggerName);
  logger.configure("CTRAN", []() { return 0; }, {}, false);
  logger.configureOutput({});

  int argumentEvaluations = 0;
  testing::internal::CaptureStdout();
  logger.set_level(spdlog::level::debug);
  CTRAN_LOG(DBG5, "suppressed trace {}", ++argumentEvaluations);
  CTRAN_LOG_STREAM(DBG5) << "suppressed trace " << ++argumentEvaluations;

  logger.set_level(spdlog::level::trace);
  CTRAN_LOG(DBG5, "enabled trace {}", ++argumentEvaluations);
  CTRAN_LOG_STREAM(DBG5) << "enabled trace " << ++argumentEvaluations;
  logger.flush();
  const auto output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(argumentEvaluations, 2);
  EXPECT_THAT(output, testing::Not(testing::HasSubstr("suppressed trace")));
  EXPECT_THAT(output, testing::HasSubstr("enabled trace 1"));
  EXPECT_THAT(output, testing::HasSubstr("enabled trace 2"));
}

TEST_F(CtranUtilsLogTest, StandaloneLoggingPreservesLegacyDelivery) {
  ctran::logging::configureStandaloneCtranLogging(spdlog::level::warn);

  auto& logger =
      meta::comms::logger::getSpdlogLogger(ctran::logging::kCtranLoggerName);
  EXPECT_FALSE(logger.usesAsyncLogging());
  EXPECT_FALSE(logger.should_log(spdlog::level::info));
  EXPECT_TRUE(logger.should_log(spdlog::level::warn));

  ctran::logging::configureStandaloneCtranLogging(spdlog::level::info);
  EXPECT_FALSE(logger.usesAsyncLogging());
  EXPECT_TRUE(logger.should_log(spdlog::level::info));
}

TEST_F(CtranUtilsLogTest, TestCtranLogFirstNPreservesEnabledBudget) {
  auto& logger =
      meta::comms::logger::getSpdlogLogger(ctran::logging::kCtranLoggerName);
  std::vector<std::string> errors;
  logger.configure(
      "CTRAN",
      []() { return 0; },
      [&](std::string_view error) { errors.emplace_back(error); },
      false);

  int evaluated = 0;
  auto log = [&] {
    CTRAN_LOG_FIRST_N(ERR, 2, "rate-limited error {}", ++evaluated);
  };

  logger.set_level(spdlog::level::off);
  log();
  log();
  EXPECT_TRUE(errors.empty());
  EXPECT_EQ(evaluated, 0);

  logger.set_level(spdlog::level::err);
  log();
  log();
  log();
  EXPECT_THAT(
      errors,
      testing::ElementsAre("rate-limited error 1", "rate-limited error 2"));
  EXPECT_EQ(evaluated, 2);
}

TEST_F(CtranUtilsLogTest, TestCtranLogIfPreservesLevelAndSubsystemGates) {
  auto& logger =
      meta::comms::logger::getSpdlogLogger(ctran::logging::kCtranLoggerName);
  logger.configure("CTRAN", []() { return 0; }, {}, false);

  int filteredConditionEvaluations = 0;
  int fatalConditionEvaluations = 0;
  int conditionEvaluations = 0;
  int argumentEvaluations = 0;
  auto logIf = [&] {
    CTRAN_LOG_IF(
        INFO,
        ++conditionEvaluations == 1,
        "CTRAN IF {}",
        ++argumentEvaluations);
  };

  testing::internal::CaptureStdout();
  logger.set_level(spdlog::level::off);
  CTRAN_LOG_IF(DBG, ++filteredConditionEvaluations, "FILTERED DBG");
  CTRAN_LOG_IF(WARN, ++filteredConditionEvaluations, "FILTERED WARN");
  CTRAN_LOG_IF(ERR, ++filteredConditionEvaluations, "FILTERED ERR");
  CTRAN_LOG_IF(CRITICAL, ++filteredConditionEvaluations, "FILTERED CRITICAL");
  CTRAN_LOG_IF(FATAL, ++fatalConditionEvaluations == 0, "DISABLED FATAL");
  logger.set_level(spdlog::level::warn);
  logIf();
  logger.set_level(spdlog::level::info);
  logIf();
  logIf();

  meta::comms::logger::setSubSystemMask(meta::comms::logger::SubSystem::ENV);
  CTRAN_LOG_SUBSYS(INFO, ENV, "CTRAN ENABLED SUBSYSTEM");
  CTRAN_LOG_SUBSYS(INFO, COLL, "CTRAN DISABLED SUBSYSTEM");
  logger.flush();
  const auto output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(filteredConditionEvaluations, 0);
  EXPECT_EQ(fatalConditionEvaluations, 1);
  EXPECT_EQ(conditionEvaluations, 2);
  EXPECT_EQ(argumentEvaluations, 1);
  EXPECT_THAT(output, testing::HasSubstr("CTRAN IF 1"));
  EXPECT_THAT(output, testing::HasSubstr("CTRAN ENABLED SUBSYSTEM"));
  EXPECT_THAT(
      output, testing::Not(testing::HasSubstr("CTRAN DISABLED SUBSYSTEM")));
}

TEST_F(CtranUtilsLogTest, TestCtranLogEveryMsPreservesEnabledBudget) {
  auto& logger =
      meta::comms::logger::getSpdlogLogger(ctran::logging::kCtranLoggerName);
  std::vector<std::string> errors;
  logger.configure(
      "CTRAN",
      []() { return 0; },
      [&](std::string_view error) { errors.emplace_back(error); },
      false);

  int evaluated = 0;
  auto log = [&] {
    CTRAN_LOG_EVERY_MS(ERR, 60'000, "rate-limited error {}", ++evaluated);
  };

  logger.set_level(spdlog::level::off);
  log();
  logger.set_level(spdlog::level::err);
  log();
  log();

  EXPECT_THAT(errors, testing::ElementsAre("rate-limited error 1"));
  EXPECT_EQ(evaluated, 1);
}

TEST_F(CtranUtilsLogTest, TestCtranTraceFormat) {
  auto traceGuard = EnvRAII(NCCL_CTRAN_ENABLE_TRACE_LOG, true);
  auto& logger =
      meta::comms::logger::getSpdlogLogger(ctran::logging::kCtranLoggerName);
  logger.configure("CTRAN", []() { return 0; }, {}, false);
  logger.set_level(spdlog::level::info);
  meta::comms::logger::setSubSystemMask(meta::comms::logger::SubSystem::COLL);

  testing::internal::CaptureStdout();
  CTRAN_LOG_TRACE(COLL, "trace value {}", 42);
  logger.flush();
  const auto output = testing::internal::GetCapturedStdout();

  EXPECT_THAT(output, testing::HasSubstr("[TRACE] TestBody: trace value 42"));
}

TEST_F(CtranUtilsLogTest, TestCtranLogPreservesProducerThreadIds) {
  int tid0 = 0, tid1 = 0;
  testing::internal::CaptureStdout();
  std::thread t1([&tid0]() {
    tid0 = syscall(SYS_gettid);
    CTRAN_LOG(INFO, "Test message with value: {}", 42);
  });
  t1.join();
  std::thread t2([&tid1]() {
    tid1 = syscall(SYS_gettid);
    CTRAN_LOG(INFO, "Test message with value: {}", 43);
  });
  t2.join();
  meta::comms::logger::getSpdlogLogger(ctran::logging::kCtranLoggerName)
      .flush();
  const auto output = testing::internal::GetCapturedStdout();
  const auto hostname = ctran::utils::getHostname();
  const auto pid = getpid();

  EXPECT_THAT(
      output, testing::HasSubstr(fmt::format("{}:{}:{}", hostname, pid, tid0)));
  EXPECT_THAT(
      output, testing::HasSubstr(fmt::format("{}:{}:{}", hostname, pid, tid1)));
}

TEST_F(CtranUtilsLogTest, TestCtranSubsystemMaskSupportsCombinations) {
  meta::comms::logger::setSubSystemMask(
      ::meta::comms::logger::SubSystem::ALLOC |
      ::meta::comms::logger::SubSystem::NET);

  testing::internal::CaptureStdout();
  CTRAN_LOG_SUBSYS(
      INFO, ALLOC | COLL, "This should be logged (ALLOC enabled) {}", 44);
  CTRAN_LOG_SUBSYS(
      INFO, COLL | P2P, "This should NOT be logged (both disabled) {}", 45);
  meta::comms::logger::setSubSystemMask(::meta::comms::logger::SubSystem::ALL);
  CTRAN_LOG_SUBSYS(INFO, COLL | P2P, "This should be logged {}", 103);
  meta::comms::logger::getSpdlogLogger(ctran::logging::kCtranLoggerName)
      .flush();
  const auto output = testing::internal::GetCapturedStdout();

  EXPECT_THAT(
      output, testing::HasSubstr("This should be logged (ALLOC enabled) 44"));
  EXPECT_THAT(
      output,
      testing::Not(
          testing::HasSubstr("This should NOT be logged (both disabled) 45")));
  EXPECT_THAT(output, testing::HasSubstr("This should be logged 103"));
}
