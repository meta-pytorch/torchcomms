// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gtest/gtest.h>
#include "comms/ctran/CtranComm.h"

namespace ctran {

class CtranEnvironmentBase : public ::testing::Environment {
 public:
  void SetUp() override;
  void TearDown() override;
};

// CtranDistTestFixture is a fixture for testing Ctran with multiple
// processes/ranks
// It currently uses MpiBootstrap to create a CtranComm, can extend to use
// other bootstrap methods like TcpStore for torchrun
class CtranDistTestFixture : public ::testing::Test {
 public:
 protected:
  void SetUp() override;
  void TearDown() override;

  // helper function to create a CtranComm
  // add more configs to return CtranComm with different flavors
  std::unique_ptr<CtranComm> makeCtranComm();

  int globalRank{-1};
  int numRanks{-1};
  int localRank{-1};
  int numLocalRanks_{-1};
};

} // namespace ctran
