// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/common/fault_tolerance/Abort.h"
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

} // namespace ctran::testing
