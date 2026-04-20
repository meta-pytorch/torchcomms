// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/DistTestBase.h"

namespace ctran {

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

  // Intra-node (NVL domain) barrier using CtranComm's bootstrap
  void barrierNvlDomain(CtranComm* comm);

  // Type-safe wrapper for intra-node all-gather
  template <typename T>
  void allGatherNvlDomain(CtranComm* comm, std::vector<T>& data) {
    auto resFuture = comm->bootstrap_->allGatherNvlDomain(
        data.data(),
        sizeof(T),
        comm->statex_->localRank(),
        comm->statex_->nLocalRanks(),
        comm->statex_->localRankToRanks());
    COMMCHECK_TEST(static_cast<commResult_t>(std::move(resFuture).get()));
  }

  bool enableNolocal{false};
};

// Dump colltrace records from a standalone CtranComm's colltraceNew_.
// Returns a map with keys like "CT_pastColls", "CT_pendingColls",
// "CT_currentColls". Returns empty map if colltrace is not initialized.
std::unordered_map<std::string, std::string> dumpCollTrace(CtranComm* comm);

// Poll until CT_currentColls drains to "[]", then return the final dump.
// On single-node configs, CudaWaitEvent may delay colltrace transitions,
// so a fixed sleep is insufficient. Polls every 50ms up to timeoutMs.
std::unordered_map<std::string, std::string> waitForCollTraceDrain(
    CtranComm* comm,
    int timeoutMs = 5000);

} // namespace ctran
