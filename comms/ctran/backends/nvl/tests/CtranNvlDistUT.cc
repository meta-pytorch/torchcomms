// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <pthread.h>
#include <stdlib.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/backends/nvl/CtranNvl.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"

#include "comms/testinfra/TestsCuUtils.h"
#if !defined(USE_ROCM)
// needed because we use ncclMemAlloc to test kMemNcclMemAlloc mem type.
// cuMem API is not supported on AMD so we don't test it on AMD.
#include "comms/testinfra/TestUtils.h"
#endif

class CtranNvlTest : public ctran::CtranDistTestFixture {
 public:
  CtranNvlTest() = default;
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    CtranDistTestFixture::SetUp();
    comm_ = makeCtranComm();
    comm = comm_.get();

    // Check epoch lock for the entire test
    NCCL_CTRAN_IB_EPOCH_LOCK_ENFORCE_CHECK = true;

    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    CtranDistTestFixture::TearDown();
  }

 protected:
  std::unique_ptr<CtranComm> comm_{nullptr};
  CtranComm* comm{nullptr};
};

class CtranNvlTestSuite : public CtranNvlTest,
                          public ::testing::WithParamInterface<MemAllocType> {};

TEST_P(CtranNvlTestSuite, NormalInitialize) {
  // Expect CtranNvl to be initialized without internal error
  try {
    auto ctranNvl = std::make_unique<CtranNvl>(this->comm);
  } catch (const std::bad_alloc&) {
    GTEST_SKIP() << "NVL backend failed to allocate. Skip test";
  }
}

TEST_F(CtranNvlTest, PrecomputedPhysicalDomainIsTrusted) {
  auto rankTopologies = comm->statex_->rankTopologiesRef();
  std::vector<std::vector<int>> domains;
  for (int rank = 0; rank < comm->statex_->nRanks(); ++rank) {
    const std::string host = rankTopologies[rank].host;
    auto domain = domains.begin();
    for (; domain != domains.end(); ++domain) {
      if (std::string{rankTopologies[domain->front()].host} == host) {
        break;
      }
    }
    if (domain == domains.end()) {
      domains.push_back({rank});
    } else {
      domain->push_back(rank);
    }
  }
  comm->statex_->setPrecomputedTopology(
      std::move(rankTopologies),
      std::move(domains),
      /*fabricActive=*/false);

  auto bootstrap = std::move(comm->bootstrap_);
  auto restoreBootstrap =
      folly::makeGuard([&] { comm->bootstrap_ = std::move(bootstrap); });
  CtranNvl ctranNvl{comm};
  for (int rank = 0; rank < comm->statex_->nRanks(); ++rank) {
    const bool sameHost = comm->statex_->host() == comm->statex_->host(rank);
    EXPECT_EQ(ctranNvl.isSupported(rank), sameHost);
    EXPECT_FALSE(ctranNvl.isNvlFabric(rank));
  }
}

TEST_F(CtranNvlTest, PrecomputedFabricDomainIsTrusted) {
  std::vector<int> ranks;
  auto rankTopologies = comm->statex_->rankTopologiesRef();
  for (int rank = 0; rank < comm->statex_->nRanks(); ++rank) {
    ranks.push_back(rank);
    std::snprintf(
        rankTopologies[rank].host,
        sizeof(rankTopologies[rank].host),
        "precomputed-host-%d",
        rank);
  }
  comm->statex_->setPrecomputedTopology(
      std::move(rankTopologies),
      {std::move(ranks)},
      /*fabricActive=*/true);

  auto bootstrap = std::move(comm->bootstrap_);
  auto restoreBootstrap =
      folly::makeGuard([&] { comm->bootstrap_ = std::move(bootstrap); });
  CtranNvl ctranNvl{comm};
  for (int rank = 0; rank < comm->statex_->nRanks(); ++rank) {
    EXPECT_TRUE(ctranNvl.isSupported(rank));
    EXPECT_EQ(ctranNvl.isNvlFabric(rank), rank != comm->statex_->rank());
  }
}

INSTANTIATE_TEST_SUITE_P(
    CtranNvlTestInstance,
    CtranNvlTestSuite,
#if !defined(USE_ROCM)
    ::testing::Values(kMemNcclMemAlloc, kMemCudaMalloc));
#else
    ::testing::Values(kMemCudaMalloc));
#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
