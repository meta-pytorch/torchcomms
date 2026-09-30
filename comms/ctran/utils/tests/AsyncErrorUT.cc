// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>

#include "comms/common/AsyncErrorState.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/utils/AsyncError.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/utils/cvars/nccl_cvars.h"

using ctran::utils::Exception;

namespace {

class ScopedBoolOverride {
 public:
  ScopedBoolOverride(bool& value, bool replacement)
      : value_(value), original_(value) {
    value_ = replacement;
  }

  ~ScopedBoolOverride() {
    value_ = original_;
  }

 private:
  bool& value_;
  bool original_;
};

} // namespace

void expectFaultToleranceAbortReason(
    commResult_t result,
    comms::fault_tolerance::AbortReason expectedReason) {
  auto comm = std::make_unique<CtranComm>(
      comms::fault_tolerance::createAbort(/*enabled=*/true));
  CTRAN_ASYNC_ERR_GUARD_FAULT_TOLERANCE(
      comm, { throw ::ctran::utils::Exception("UT error", result); }, -1, 0);

  const auto abortInfo = comm->getAbortInfo();
  ASSERT_TRUE(abortInfo.has_value());
  EXPECT_EQ(abortInfo->reason, expectedReason);
  EXPECT_THAT(
      abortInfo->context,
      ::testing::AllOf(
          ::testing::HasSubstr("op_type=-1"),
          ::testing::HasSubstr("op_count=0"),
          ::testing::HasSubstr("UT error")));
}

TEST(AsyncErrorTest, AbortOnError) {
  // Do not check exit code since CTRAN_LOG(FATAL) may trigger core dump, and
  // changed SIGABRT to core dump
  EXPECT_DEATH(
      {
        NCCL_CTRAN_ABORT_ON_ERROR = true;
        auto comm = std::make_unique<CtranComm>(
            comms::fault_tolerance::createAbort(/*enabled=*/false));
        NCCL_CTRAN_ABORT_ON_ERROR = false;
        CTRAN_ASYNC_ERR_GUARD(comm, {
          throw Exception("test error on thread 0", commInternalError);
        });
      },
      "test error on thread 0")
      << "Expected to abort on error";
}

TEST(AsyncErrorTest, FaultToleranceDisabledRecordsBeforeRethrow) {
  ScopedBoolOverride abortOnError(NCCL_CTRAN_ABORT_ON_ERROR, false);
  auto comm = std::make_unique<CtranComm>(
      comms::fault_tolerance::createAbort(/*enabled=*/false));
  EXPECT_THROW(
      {
        CTRAN_ASYNC_ERR_GUARD_FAULT_TOLERANCE(
            comm,
            { throw ::ctran::utils::Exception("UT error", commRemoteError); },
            -1,
            0);
      },
      ::ctran::utils::Exception)
      << "Expected to throw exception on error";
  EXPECT_EQ(comm->getAsyncResult(), commRemoteError);
  EXPECT_THAT(
      comm->getAbort()->getAsyncError().message,
      ::testing::HasSubstr("UT error"));
}

TEST(AsyncErrorTest, FaultToleranceEnabled) {
  ScopedBoolOverride abortOnError(NCCL_CTRAN_ABORT_ON_ERROR, false);
  expectFaultToleranceAbortReason(
      commRemoteError, comms::fault_tolerance::AbortReason::NETWORK_ERROR);
  expectFaultToleranceAbortReason(
      commTimeout, comms::fault_tolerance::AbortReason::TIMED_OUT);
  expectFaultToleranceAbortReason(
      commUserAbort, comms::fault_tolerance::AbortReason::ABORTED);
  expectFaultToleranceAbortReason(
      commInternalError, comms::fault_tolerance::AbortReason::INTERNAL_ERROR);
}

TEST(AsyncErrorTest, FaultToleranceEnabledRuntimeErrorRecordsInternalAbort) {
  ScopedBoolOverride abortOnError(NCCL_CTRAN_ABORT_ON_ERROR, false);
  auto comm = std::make_unique<CtranComm>(
      comms::fault_tolerance::createAbort(/*enabled=*/true));

  CTRAN_ASYNC_ERR_GUARD_FAULT_TOLERANCE(
      comm, { throw std::runtime_error("UT runtime error"); }, -1, 0);

  const auto abortInfo = comm->getAbortInfo();
  ASSERT_TRUE(abortInfo.has_value());
  EXPECT_EQ(
      abortInfo->reason, comms::fault_tolerance::AbortReason::INTERNAL_ERROR);
  EXPECT_THAT(
      abortInfo->context,
      ::testing::AllOf(
          ::testing::HasSubstr("op_type=-1"),
          ::testing::HasSubstr("op_count=0"),
          ::testing::HasSubstr("UT runtime error"),
          ::testing::HasSubstr("commInternalError")));
}

TEST(AsyncErrorTest, ErrorUsesLastWriterWhileAbortKeepsFirstReason) {
  ScopedBoolOverride abortOnError(NCCL_CTRAN_ABORT_ON_ERROR, false);
  auto comm = std::make_unique<CtranComm>(
      comms::fault_tolerance::createAbort(/*enabled=*/true));

  CTRAN_ASYNC_ERR_GUARD_FAULT_TOLERANCE(
      comm, { throw Exception("first error", commRemoteError); }, -1, 0);
  CTRAN_ASYNC_ERR_GUARD_FAULT_TOLERANCE(
      comm, { throw Exception("second error", commTimeout); }, -1, 1);

  EXPECT_EQ(comm->getAsyncResult(), commTimeout);
  const auto abortInfo = comm->getAbortInfo();
  ASSERT_TRUE(abortInfo.has_value());
  EXPECT_EQ(
      abortInfo->reason, comms::fault_tolerance::AbortReason::NETWORK_ERROR);
  EXPECT_THAT(abortInfo->context, ::testing::HasSubstr("first error"));
}
