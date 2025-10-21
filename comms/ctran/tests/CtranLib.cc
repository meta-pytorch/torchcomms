// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/utils/Exception.h"

using ctran::utils::Exception;

TEST(CtranLib, CreateCommunicator) {
  auto comm = std::make_unique<CtranComm>();

  EXPECT_EQ(comm->ctran_.get(), nullptr);
  EXPECT_EQ(comm->bootstrap_.get(), nullptr);
  EXPECT_EQ(comm->collTrace_.get(), nullptr);
  EXPECT_EQ(comm->colltraceNew_.get(), nullptr);
  EXPECT_EQ(comm->memCache_.get(), nullptr);
  EXPECT_EQ(comm->statex_.get(), nullptr);
}
