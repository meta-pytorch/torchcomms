// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <mpi.h>
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "caffe2/torch/csrc/distributed/c10d/TCPStore.hpp"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/utils/CudaRAII.h"

namespace ctran {

std::unique_ptr<c10d::TCPStore> createTcpStore(bool isServer);

// Detect which initialization environment to use
InitEnvType getInitEnvType();

// Base environment for distributed tests (handles MPI_Init/Finalize)
class CtranDistEnvironment : public ::testing::Environment {
 public:
  void SetUp() override;
  void TearDown() override;
};

// Backwards compatibility alias for existing tests
using CtranEnvironmentBase = CtranDistEnvironment;

// CtranDistTestFixture is a fixture for testing Ctran with multiple
// processes/ranks that supports both MPI and TCPStore bootstrap methods.
class CtranDistTestFixture : public CtranTestFixtureBase {
 public:
 protected:
  void SetUp() override;
  void TearDown() override;

  std::unique_ptr<CtranComm> makeCtranComm();

  // Rank information
  int globalRank{-1};
  int numRanks{-1};
  int localRank{-1};
  int numLocalRanks_{-1};
  bool enableNolocal{false};

 private:
  void setUpMpi();
  void setUpTcpStore();

  // TCP Store support
  std::unique_ptr<c10d::TCPStore> tcpStore_{nullptr};
  bool isTcpStoreServer() const;
  std::vector<std::string>
  exchangeInitUrls(const std::string& selfUrl, int numRanks, int selfRank);

  // Test counter for TCP Store key generation
  static std::atomic<int> testCount_;
};

} // namespace ctran
