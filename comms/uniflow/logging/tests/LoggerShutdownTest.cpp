// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/logging/Logger.h"

#include <mutex>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace uniflow::logging::testing {
// Defined in Logger.cpp so they run against that translation unit's copy of
// spdlog's statically-linked registry. Declared here rather than in Logger.h to
// keep test-only entry points out of the public OSS header.
void shutdownRegistryForTesting();
bool globalThreadPoolAliveForTesting();
} // namespace uniflow::logging::testing

namespace {

// The error handler can fire from the async worker thread while it drains.
class ErrorLog {
 public:
  void record(const std::string& message) {
    const std::lock_guard lock{mutex_};
    messages_.push_back(message);
  }

  std::vector<std::string> messages() const {
    const std::lock_guard lock{mutex_};
    return messages_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<std::string> messages_;
};

/*
 * Shutting the registry down is process-global and irreversible, so this test
 * gets its own binary: any later test that logged would hit the drop path below
 * instead of the behavior it meant to check.
 *
 * Scope: this does NOT cover the deliberate leak in getLogger(). A leaked
 * pointer and a plain function-local static both keep the logger alive for the
 * whole process, and differ only in whether __cxa_atexit registers a
 * destructor -- observable at exit, after this binary has already reported its
 * results. What is covered is the contract that makes the leak worth having:
 * reaching the logger once the pool is gone drops the message and reports it,
 * instead of writing through a destroyed sink.
 */
TEST(LoggerShutdownTest, DropsAndReportsOnceThreadPoolIsGone) {
  auto* const logger = uniflow::logging::getLogger();
  ASSERT_NE(logger, nullptr);
  ASSERT_TRUE(uniflow::logging::testing::globalThreadPoolAliveForTesting());

  ErrorLog errors;
  logger->set_error_handler(
      [&errors](const std::string& message) { errors.record(message); });

  UNIFLOW_LOG_INFO("delivered while the pool is alive");
  EXPECT_TRUE(errors.messages().empty());

  uniflow::logging::testing::shutdownRegistryForTesting();
  ASSERT_FALSE(uniflow::logging::testing::globalThreadPoolAliveForTesting());

  // Registry shutdown drops its own reference to the logger; the logger itself
  // outlives it, so callers still racing at exit get a live object.
  EXPECT_EQ(uniflow::logging::getLogger(), logger);

  UNIFLOW_LOG_INFO("dropped once the pool is gone");
  auto observed = errors.messages();
  ASSERT_EQ(observed.size(), 1U);
  EXPECT_NE(
      observed[0].find("thread pool doesn't exist anymore"), std::string::npos);

  // flush_() takes the same expired-weak_ptr branch as sink_it_().
  logger->flush();
  observed = errors.messages();
  ASSERT_EQ(observed.size(), 2U);
  EXPECT_NE(
      observed[1].find("thread pool doesn't exist anymore"), std::string::npos);
}

} // namespace
