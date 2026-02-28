// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>
#include <optional>

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comm.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/hints/CommHintConfig.h" // @manual
#include "meta/hints/GlobalHints.h" // @manual
#include "nccl.h"
#include "transport.h"

#include "comms/ctran/Ctran.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include "VerifyAlgoStatsUtil.h"

// Hint lifecycle tests

class CommWithNoLocalTest : public ::testing::Test {
 public:
  CommWithNoLocalTest() = default;

  void SetUp() override {
    initEnv();
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
  }

  void TearDown() override {
    ncclx::resetGlobalHint(std::string(ncclx::HintKeys::kCommNoLocal));
  }

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
};

TEST_F(CommWithNoLocalTest, NoLocalDisabledByDefault) {
  NcclCommRAII comm{globalRank, numRanks, localRank};
  ASSERT_NE(comm.get(), nullptr);
  EXPECT_FALSE(comm->noLocal_);
}

namespace {
enum class TestCommCreateMode { kDefault, kSplit };
enum class CollectiveOp { kAllGather, kReduceScatter };
} // namespace

class CommWithNoLocalTestParam : public CommWithNoLocalTest,
                                 public ::testing::WithParamInterface<
                                     std::tuple<TestCommCreateMode, bool>> {};

TEST_P(CommWithNoLocalTestParam, NoLocalEnableByHint) {
  const auto& [createMode, blockingInit] = GetParam();

  // Default disabled
  NcclCommRAII comm1{globalRank, numRanks, localRank};
  ASSERT_NE(comm1.get(), nullptr);
  EXPECT_FALSE(comm1->noLocal_);

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = blockingInit ? 1 : 0;
  const auto commDescStr = fmt::format("{}-{}", kNcclUtCommDesc, "noLocal");
  config.commDesc = commDescStr.c_str();

  // Enable by hint
  ASSERT_EQ(
      ncclx::setGlobalHint(std::string(ncclx::HintKeys::kCommNoLocal), "1"),
      ncclSuccess);

  // Use appropriate RAII wrapper based on creation mode
  std::optional<NcclCommRAII> comm2Default;
  std::optional<NcclCommSplitRAII> comm2Split;
  ncclComm_t comm2;
  if (createMode == TestCommCreateMode::kDefault) {
    comm2Default.emplace(globalRank, numRanks, localRank, false, &config);
    comm2 = comm2Default->get();
  } else {
    comm2Split.emplace(comm1.get(), 1, this->globalRank, &config);
    comm2 = comm2Split->get();
  }
  ASSERT_NE(comm2, nullptr);

  // If nonblocking init, wait till async init is done
  if (!blockingInit) {
    auto commStatus = ncclInProgress;
    do {
      ASSERT_EQ(ncclCommGetAsyncError(comm2, &commStatus), ncclSuccess);
      if (commStatus == ncclInProgress) {
        sched_yield();
      }
    } while (commStatus == ncclInProgress);
  }

  EXPECT_TRUE(comm2->noLocal_);

  ASSERT_TRUE(
      ncclx::resetGlobalHint(std::string(ncclx::HintKeys::kCommNoLocal)));

  // Now disabled again
  {
    NcclCommRAII comm3{globalRank, numRanks, localRank};
    ASSERT_NE(comm3.get(), nullptr);
    EXPECT_FALSE(comm3->noLocal_);
  }
}

INSTANTIATE_TEST_SUITE_P(
    CommWithNoLocalTestInstance,
    CommWithNoLocalTestParam,
    ::testing::Combine(
        ::testing::Values(
            TestCommCreateMode::kDefault,
            TestCommCreateMode::kSplit),
        ::testing::Values(true, false)),
    [&](const testing::TestParamInfo<CommWithNoLocalTestParam::ParamType>&
            info) {
      return fmt::format(
          "{}_{}",
          std::get<0>(info.param) == TestCommCreateMode::kDefault ? "default"
                                                                  : "split",
          std::get<1>(info.param) ? "blockingInit" : "nonblockingInit");
    });

// E2E transport verification tests

class CommWithNoLocalCollTest
    : public NcclxBaseTest,
      public ::testing::WithParamInterface<
          std::tuple<bool /*noLocal*/, CollectiveOp>> {
 public:
  void SetUp() override {
    NcclxBaseTest::SetUp();
    algoStats_.enable();
    ASSERT_EQ(
        ncclx::setGlobalHint(std::string(ncclx::HintKeys::kCommUseCtran), "1"),
        ncclSuccess);
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    ncclx::resetGlobalHint(std::string(ncclx::HintKeys::kCommNoLocal));
    ncclx::resetGlobalHint(std::string(ncclx::HintKeys::kCommUseCtran));
    NcclxBaseTest::TearDown();
  }

  void verifyTransport(ncclComm_t comm, bool expectNoLocal) {
    INFO(
        NCCL_INIT,
        "noLocal comm nChannels=%d nRanks=%d",
        comm->nChannels,
        comm->nRanks);
    ASSERT_GT(comm->nChannels, 0);

    bool foundLocal = false;
    for (int c = 0; c < comm->nChannels; c++) {
      for (int peer = 0; peer < comm->nRanks; peer++) {
        if (peer == comm->rank) {
          continue;
        }
        for (int connIdx = 0; connIdx < NCCL_MAX_CONNS; connIdx++) {
          auto* sendConn = &comm->channels[c].peers[peer]->send[connIdx];
          if (sendConn->connected) {
            const bool isLocalSend =
                (sendConn->transportComm ==
                     &ncclTransports[TRANSPORT_P2P]->send ||
                 sendConn->transportComm ==
                     &ncclTransports[TRANSPORT_SHM]->send);
            if (expectNoLocal) {
              EXPECT_FALSE(isLocalSend)
                  << "channel=" << c << " peer=" << peer
                  << " connIdx=" << connIdx << " uses local send transport";
            } else {
              foundLocal |= isLocalSend;
            }
          }
          auto* recvConn = &comm->channels[c].peers[peer]->recv[connIdx];
          if (recvConn->connected) {
            const bool isLocalRecv =
                (recvConn->transportComm ==
                     &ncclTransports[TRANSPORT_P2P]->recv ||
                 recvConn->transportComm ==
                     &ncclTransports[TRANSPORT_SHM]->recv);
            if (expectNoLocal) {
              EXPECT_FALSE(isLocalRecv)
                  << "channel=" << c << " peer=" << peer
                  << " connIdx=" << connIdx << " uses local recv transport";
            } else {
              foundLocal |= isLocalRecv;
            }
          }
        }
      }
    }
    if (!expectNoLocal) {
      EXPECT_TRUE(foundLocal) << "expected at least one local transport";
    }
  }

  void verifyCtranBackend(
      ncclComm_t comm,
      bool expectNoLocal,
      CollectiveOp collectiveOp) {
    if (!ctranInitialized(comm->ctranComm_.get())) {
      return;
    }

    bool supported = false;
    switch (collectiveOp) {
      case CollectiveOp::kAllGather:
        supported = ctranAllGatherSupport(
            comm->ctranComm_.get(), NCCL_ALLGATHER_ALGO::ctran);
        break;
      case CollectiveOp::kReduceScatter:
        supported = ctranReduceScatterSupport(
            comm->ctranComm_.get(), NCCL_REDUCESCATTER_ALGO::ctran);
        break;
    }
    if (!supported) {
      return;
    }

    auto* mapper = comm->ctranComm_->ctran_->mapper.get();
    const int totalPuts = mapper->iPutCount[CtranMapperBackend::IB] +
        mapper->iPutCount[CtranMapperBackend::NVL];
    if (totalPuts == 0) {
      return;
    }

    if (expectNoLocal) {
      EXPECT_EQ(mapper->iPutCount[CtranMapperBackend::NVL], 0)
          << "noLocal: expected no NVL puts";
      EXPECT_GT(mapper->iPutCount[CtranMapperBackend::IB], 0)
          << "noLocal: expected IB puts";
    } else {
      if (comm->ctranComm_->statex_->nLocalRanks() > 1) {
        EXPECT_GT(mapper->iPutCount[CtranMapperBackend::NVL], 0)
            << "default: expected NVL puts for local peers";
      }
    }
  }

  // Runs the collective and verifies data correctness.
  // Caller is responsible for CVAR overrides and post-run verification.
  void run(ncclComm_t comm, CollectiveOp collectiveOp) {
    const size_t count = 50 * 1024 * 1024;
    const size_t totalCount = count * numRanks;
    const bool isAllGather = (collectiveOp == CollectiveOp::kAllGather);
    const size_t sendSize = (isAllGather ? count : totalCount) * sizeof(int);
    const size_t recvSize = (isAllGather ? totalCount : count) * sizeof(int);

    std::vector<TestMemSegment> sendSegs, recvSegs;
    int* sendBuf = reinterpret_cast<int*>(
        testAllocBuf(sendSize, kMemNcclMemAlloc, sendSegs));
    int* recvBuf = reinterpret_cast<int*>(
        testAllocBuf(recvSize, kMemNcclMemAlloc, recvSegs));

    void* sendHandle = nullptr;
    void* recvHandle = nullptr;
    NCCLCHECK_TEST(ncclCommRegister(comm, sendBuf, sendSize, &sendHandle));
    NCCLCHECK_TEST(ncclCommRegister(comm, recvBuf, recvSize, &recvHandle));

    std::vector<int> expectedVals;

    switch (collectiveOp) {
      case CollectiveOp::kAllGather: {
        assignChunkValue(sendBuf, count, globalRank + 1);
        assignChunkValue(recvBuf, totalCount, -1);
        ASSERT_EQ(
            ncclAllGather(sendBuf, recvBuf, count, ncclInt, comm, stream),
            ncclSuccess);
        expectedVals.reserve(numRanks);
        for (int r = 0; r < numRanks; r++) {
          expectedVals.push_back(r + 1);
        }
        break;
      }
      case CollectiveOp::kReduceScatter: {
        for (int r = 0; r < numRanks; r++) {
          assignChunkValue(
              sendBuf + r * count, count, globalRank * numRanks + r);
        }
        assignChunkValue(recvBuf, count, -1);
        ASSERT_EQ(
            ncclReduceScatter(
                sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream),
            ncclSuccess);
        int val = 0;
        for (int r = 0; r < numRanks; r++) {
          val += r * numRanks + globalRank;
        }
        expectedVals.push_back(val);
        break;
      }
    }
    CUDACHECK_TEST(cudaDeviceSynchronize());

    for (size_t r = 0; r < expectedVals.size(); r++) {
      const size_t errs = checkChunkValue(
          recvBuf + r * count,
          count,
          expectedVals[r],
          0,
          globalRank,
          stream,
          0);
      EXPECT_EQ(errs, 0) << "Rank " << globalRank << " chunk " << r << " has "
                         << errs << " errors";
    }

    NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
    NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));

    testFreeBuf(sendBuf, sendSize, kMemNcclMemAlloc);
    testFreeBuf(recvBuf, recvSize, kMemNcclMemAlloc);
  }

  ncclx::test::VerifyAlgoStatsHelper algoStats_;
  cudaStream_t stream{nullptr};
};

TEST_P(CommWithNoLocalCollTest, BaselineRun) {
  const auto [noLocal, collectiveOp] = GetParam();

  if (noLocal) {
    ASSERT_EQ(
        ncclx::setGlobalHint(std::string(ncclx::HintKeys::kCommNoLocal), "1"),
        ncclSuccess);
  }

  NcclCommRAII comm{
      globalRank, numRanks, localRank, false, nullptr, server.get()};
  ASSERT_NE(comm.get(), nullptr);
  EXPECT_EQ(comm->noLocal_, noLocal);

  run(comm.get(), collectiveOp);
  const char* collectiveName = (collectiveOp == CollectiveOp::kAllGather)
      ? "AllGather"
      : "ReduceScatter";
  algoStats_.dump(comm.get(), collectiveName);
  verifyTransport(comm.get(), noLocal);
}

TEST_P(CommWithNoLocalCollTest, CtranRun) {
  const auto [noLocal, collectiveOp] = GetParam();

  if (!noLocal) {
    GTEST_SKIP() << "CtranRun only tests noLocal mode";
  }

  ASSERT_EQ(
      ncclx::setGlobalHint(std::string(ncclx::HintKeys::kCommNoLocal), "1"),
      ncclSuccess);

  NcclCommRAII comm{
      globalRank, numRanks, localRank, false, nullptr, server.get()};
  ASSERT_NE(comm.get(), nullptr);
  EXPECT_TRUE(comm->noLocal_);

  if (collectiveOp == CollectiveOp::kAllGather) {
    auto algoGuard = EnvRAII(NCCL_ALLGATHER_ALGO, NCCL_ALLGATHER_ALGO::ctran);
    run(comm.get(), collectiveOp);
  } else {
    auto algoGuard =
        EnvRAII(NCCL_REDUCESCATTER_ALGO, NCCL_REDUCESCATTER_ALGO::ctran);
    run(comm.get(), collectiveOp);
  }
  verifyCtranBackend(comm.get(), true, collectiveOp);
}

INSTANTIATE_TEST_SUITE_P(
    CommWithNoLocalCollTestInstance,
    CommWithNoLocalCollTest,
    ::testing::Combine(
        ::testing::Values(true, false),
        ::testing::Values(
            CollectiveOp::kAllGather,
            CollectiveOp::kReduceScatter)),
    [](const testing::TestParamInfo<CommWithNoLocalCollTest::ParamType>& info) {
      return fmt::format(
          "{}_{}",
          std::get<0>(info.param) ? "noLocal" : "default",
          std::get<1>(info.param) == CollectiveOp::kAllGather
              ? "AllGather"
              : "ReduceScatter");
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
