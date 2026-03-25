// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/DistTestBase.h"

namespace ctran {

// Detect which initialization environment to use
InitEnvType getInitEnvType();

// Ctran-specific environment that inherits DistEnvironmentBase and adds
// ctran-specific env vars (NCCL_CTRAN_ENABLE, profiling, etc.)
class CtranDistEnvironment : public meta::comms::DistEnvironmentBase {
 public:
  void SetUp() override;
};

// Backwards compatibility alias for existing tests
using CtranEnvironmentBase = CtranDistEnvironment;

// CtranDistTestFixture is a fixture for testing Ctran with multiple
// processes/ranks that supports both MPI and TCPStore bootstrap methods.
// Rank info, bootstrap, and per-test PrefixStore come from DistBaseTest.
class CtranDistTestFixture : public CtranTestFixtureBase,
                             public meta::comms::DistBaseTest {
 public:
 protected:
  void SetUp() override;
  void TearDown() override;

  std::unique_ptr<CtranComm> makeCtranComm();

  bool enableNolocal{false};

 private:
  std::vector<std::string>
  exchangeInitUrls(const std::string& selfUrl, int numRanks, int selfRank);
};

} // namespace ctran
