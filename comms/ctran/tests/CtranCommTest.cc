// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"

namespace ctran::testing {

TEST(CtranCommTest, AbortUnavailable) {
  EXPECT_THROW(CtranComm comm(/*abort=*/nullptr), ctran::utils::Exception);
}

TEST(CtranCommTest, AbortAvailableAndEnabled) {
  auto abort = comms::fault_tolerance::createAbort(/*enabled=*/true);
  CtranComm comm(abort);
  ASSERT_NE(comm.getAbort(), nullptr);

  EXPECT_TRUE(comm.abortEnabled());

  comm.setAbort();

  EXPECT_TRUE(comm.testAbort());
}

TEST(CtranCommTest, AbortReasonAndContextAreForwarded) {
  auto abort = comms::fault_tolerance::createAbort(/*enabled=*/true);
  CtranComm comm(abort);
  const std::string context{"collective\0failed", 17};

  comm.setAbort(
      comms::fault_tolerance::AbortInfo{
          .reason = comms::fault_tolerance::AbortReason::INTERNAL_ERROR,
          .context = context,
      });

  const auto abortInfo = comm.getAbortInfo();
  ASSERT_TRUE(abortInfo.has_value());
  EXPECT_EQ(
      *abortInfo,
      (comms::fault_tolerance::AbortInfo{
          .reason = comms::fault_tolerance::AbortReason::INTERNAL_ERROR,
          .context = context,
      }));
  EXPECT_EQ(
      comm.abortMessage(),
      "comm aborted reason=internal_error context=\"collective\\000failed\"");
}

TEST(CtranCommTest, AbortMessageEscapesContext) {
  auto abort = comms::fault_tolerance::createAbort(/*enabled=*/true);
  CtranComm comm(abort);

  comm.setAbort(
      comms::fault_tolerance::AbortInfo{
          .reason = comms::fault_tolerance::AbortReason::NETWORK_ERROR,
          .context = "peer=\"rank 1\"\\socket",
      });

  EXPECT_EQ(
      comm.abortMessage(),
      "comm aborted reason=network_error context=\"peer=\\\"rank 1\\\"\\\\socket\"");
}

TEST(CtranCommTest, AbortAvailableAndEnabledDoubleAbort) {
  auto abort = comms::fault_tolerance::createAbort(/*enabled=*/true);
  CtranComm comm(abort);
  ASSERT_NE(comm.getAbort(), nullptr);

  EXPECT_TRUE(comm.abortEnabled());

  comm.setAbort();
  comm.setAbort();

  EXPECT_TRUE(comm.testAbort());
}

TEST(CtranCommTest, AbortAvailableAndDisabled) {
  auto abort = ::comms::fault_tolerance::createAbort(/*enabled=*/false);
  CtranComm comm(abort);
  ASSERT_NE(comm.getAbort(), nullptr);

  EXPECT_FALSE(comm.abortEnabled());

  comm.setAbort();

  // disabled abort should not be set
  EXPECT_FALSE(comm.testAbort());
}

TEST(CtranCommTest, ctranCommConfigTest) {
  auto abort = comms::fault_tolerance::createAbort(/*enabled=*/true);
  ctranConfig config = {
      .backends = {CommBackend::IB, CommBackend::NVL, CommBackend::SOCKET}};

  CtranComm comm(abort, config);
  EXPECT_EQ(comm.config_.backends.size(), 3);

  /// Explicitly create comm with false abort as first argument is unomittable
  CtranComm comm2(comms::fault_tolerance::createAbort(false));
  EXPECT_EQ(comm2.config_.backends.size(), 0);
}

TEST(CtranCommTest, DeviceAllToAllvIsRetired) {
  EXPECT_FALSE(ctranDeviceAllToAllvSupport(nullptr));
  EXPECT_EQ(
      ctranDeviceAllToAllv(
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          commInt32,
          nullptr,
          nullptr,
          1,
          1,
          {}),
      commInvalidUsage);
}

} // namespace ctran::testing
