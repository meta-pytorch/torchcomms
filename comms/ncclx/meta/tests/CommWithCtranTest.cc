// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <unistd.h>
#include <fstream>
#include <memory>
#include <set>

#include <fmt/format.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/backends/ib/CtranIbSingleton.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/testinfra/TestUtils.h"
#include "meta/hints/GlobalHints.h" // @manual
#include "nccl.h"

#define dceil(x, y) ((x / y) + !!(x % y))

namespace {

constexpr int kMaxHostLen = 256;

struct RankIdentity {
  char hostname[kMaxHostLen]{};
  int pid{-1};
};

void validateHostAndGPid(
    const ncclx::CommStateX* statex,
    const std::vector<RankIdentity>& allRankIds,
    int numRanks) {
  std::set<std::string> gPids;
  for (int r = 0; r < numRanks; r++) {
    EXPECT_EQ(statex->host(r), std::string(allRankIds[r].hostname))
        << "rank " << r << " host mismatch";

    const std::string expectedGPid =
        fmt::format("{}:{}", allRankIds[r].hostname, allRankIds[r].pid);
    EXPECT_EQ(statex->gPid(r), expectedGPid)
        << "rank " << r << " gPid mismatch";

    auto [it, inserted] = gPids.insert(statex->gPid(r));
    EXPECT_TRUE(inserted) << "gPid collision at rank " << r << ": "
                          << statex->gPid(r);
  }
}

} // namespace

class CommWithCtranTest : public NcclxBaseTestFixture {
 public:
  void SetUp() override {
    // Init NCCL env so that creating communicator in each test case will not
    // initialize CVAR again, and we can override.
    NcclxBaseTestFixture::SetUp();
  }

  void TearDown() override {
    NcclxBaseTestFixture::TearDown();
  }

  void verifyPostCommResourceLeak() {
    auto s = CtranIbSingleton::getInstance();
    if (s) {
      EXPECT_EQ(s->getActiveRegCount(), 0);
    }
    EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), 0);
    EXPECT_EQ(ctran::utils::getActiveIpcRemMemCount(), 0);
  }
};

TEST_F(CommWithCtranTest, CtranEnable) {
  EnvRAII env(NCCL_CTRAN_ENABLE, true);
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};
  ASSERT_NE(comm.get(), nullptr);
  ASSERT_NE(comm->ctranComm_.get(), nullptr);
  ASSERT_TRUE(ctranInitialized(comm->ctranComm_.get()));
}

// Verify that CommStateX preserves real hostname and has unique gPid for all
// ranks after ctran initialization via createCommStateXFromNcclComm.
TEST_F(CommWithCtranTest, StatexHostAndGpidPreserved) {
  EnvRAII env(NCCL_CTRAN_ENABLE, true);
  ncclComm_t comm = ncclx::test::createNcclComm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(comm, nullptr);
  ASSERT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  auto* statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);

  std::vector<RankIdentity> allRankIds(numRanks);
  gethostname(allRankIds[globalRank].hostname, kMaxHostLen);
  allRankIds[globalRank].pid = getpid();
  oobAllGather(allRankIds, sizeof(RankIdentity));
  validateHostAndGPid(statex, allRankIds, numRanks);

  ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

TEST_F(CommWithCtranTest, CtranDisable) {
  EnvRAII env(NCCL_CTRAN_ENABLE, false);
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  ASSERT_NE(nullptr, comm.get());
  EXPECT_EQ(nullptr, comm->ctranComm_);
  EXPECT_FALSE(ctranInitialized(nullptr));

  // All CTran collective support functions should return false with nullptr
  EXPECT_FALSE(ctranAllGatherSupport(nullptr, NCCL_ALLGATHER_ALGO));
  EXPECT_FALSE(ctranAllReduceSupport(nullptr, NCCL_ALLREDUCE_ALGO::ctran));
  EXPECT_FALSE(ctranBroadcastSupport(nullptr, NCCL_BROADCAST_ALGO));
  EXPECT_FALSE(ctranReduceScatterSupport(nullptr, NCCL_REDUCESCATTER_ALGO));
  EXPECT_FALSE(ctranSendRecvSupport(0, nullptr));
  EXPECT_FALSE(ctranAllToAllSupport(
      1048576, commInt, nullptr, NCCL_ALLTOALL_ALGO::ctran));
  EXPECT_FALSE(ctranAllToAllvSupport(nullptr));
}

TEST_F(CommWithCtranTest, CtranCommInitialized) {
  EnvRAII env(NCCL_CTRAN_ENABLE, true);
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  const auto ncclComm = static_cast<ncclComm_t>(comm);
  ASSERT_NE(nullptr, ncclComm);
  ASSERT_NE(nullptr, ncclComm->ctranComm_);

  EXPECT_EQ(comm->ctranComm_->opCount_, &ncclComm->opCount);
  EXPECT_EQ(comm->ctranComm_->config_, makeCtranConfigFrom(ncclComm));
  EXPECT_EQ(comm->ctranComm_->logMetaData_, ncclComm->logMetaData);
  EXPECT_EQ(comm->ctranComm_->runtimeConn_, ncclComm->runtimeConn);
}

TEST_F(CommWithCtranTest, CTranAllGatherOverrideConfig) {
  EnvRAII ctranEnv(NCCL_CTRAN_ENABLE, true);
  EnvRAII env(NCCL_ALLGATHER_ALGO, NCCL_ALLGATHER_ALGO::orig);
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.ncclAllGatherAlgo = "ctring";
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config};

  ASSERT_NE(nullptr, comm.get());
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);
  EXPECT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  auto algo = comm->ctranComm_->ctran_->algo->getAllGatherAlgo();
  EXPECT_TRUE(algo == NCCL_ALLGATHER_ALGO::ctring);
}

TEST_F(CommWithCtranTest, CTranAllGatherOverrideConfigSplitComm) {
  EnvRAII ctranEnv(NCCL_CTRAN_ENABLE, true);
  EnvRAII env(NCCL_ALLGATHER_ALGO, NCCL_ALLGATHER_ALGO::orig);
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  std::string algoStr = "ctring";
  config.ncclAllGatherAlgo = algoStr.c_str();
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config};

  // change pointer to be an invalid value, should still work since
  // we copy the string to the config
  algoStr = "badconfig";
  ncclx::test::NcclCommSplitRAII childComm{
      comm.get(), globalRank % 2, globalRank / 2};
  ASSERT_NE(nullptr, childComm.get());

  ASSERT_NE(nullptr, comm.get());
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);
  EXPECT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  auto algo = comm->ctranComm_->ctran_->algo->getAllGatherAlgo();
  EXPECT_TRUE(algo == NCCL_ALLGATHER_ALGO::ctring);

  auto childAlgo = childComm->ctranComm_->ctran_->algo->getAllGatherAlgo();
  EXPECT_TRUE(childAlgo == NCCL_ALLGATHER_ALGO::ctring);
}

TEST_F(CommWithCtranTest, PostCommDestroy) {
  EnvRAII env(NCCL_CTRAN_ENABLE, true);
  ncclComm_t comm = ncclx::test::createNcclComm(
      globalRank, numRanks, localRank, bootstrap_.get());

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  NCCLCHECK_TEST(ncclCommDestroy(comm));

  ASSERT_FALSE(ctranInitialized(comm->ctranComm_.get()));
  // Do not check ctran->isInitialized() as it is already destroyed
}

TEST_F(CommWithCtranTest, RegMemReuseInMultiComms) {
  EnvRAII env(NCCL_CTRAN_ENABLE, true);
  constexpr int numComms = 5;

  // Create first communicator with RAII
  ncclx::test::NcclCommRAII firstComm{
      globalRank, numRanks, localRank, bootstrap_.get()};
  ASSERT_NE(nullptr, firstComm.get());
  ASSERT_NE(nullptr, firstComm->ctranComm_->ctran_);

  if (!ctranAllGatherSupport(
          firstComm->ctranComm_.get(), NCCL_ALLGATHER_ALGO)) {
    GTEST_SKIP() << "ctranAllGather is not supported. Skip test";
  }

  // Create split communicators with RAII
  std::vector<std::unique_ptr<ncclx::test::NcclCommSplitRAII>> splitComms;
  splitComms.reserve(numComms - 1);
  for (int c = 1; c < numComms; c++) {
    splitComms.push_back(
        std::make_unique<ncclx::test::NcclCommSplitRAII>(
            firstComm.get(), 0, globalRank));
    ASSERT_NE(nullptr, splitComms.back()->get());
  }

  // Raw pointer array for uniform indexed access
  std::vector<ncclComm_t> comms;
  comms.reserve(numComms);
  comms.push_back(firstComm.get());
  for (const auto& sc : splitComms) {
    comms.push_back(sc->get());
  }

  std::vector<cudaStream_t> streams(numComms, nullptr);
  for (int c = 0; c < numComms; c++) {
    CUDACHECK_TEST(cudaStreamCreate(&streams[c]));
  }

  auto regCache = ctran::RegCache::getInstance();
  ASSERT_NE(regCache, nullptr);

  auto allHandles = regCache->getSegments();
  size_t commNumHdls = allHandles.size();

  void* buf = nullptr;
  constexpr size_t count = 4096;
  size_t bufSize = count * sizeof(int) * numRanks * numComms;
  NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
  ASSERT_NE(buf, nullptr);

  // Assign value and register buffer with all comms
  std::vector<void*> hdls(numComms, nullptr);
  std::vector<void*> recvBufs(numComms, nullptr);
  std::vector<void*> sendBufs(numComms, nullptr);

  CUDACHECK_TEST(cudaMemset(buf, 0, bufSize));
  for (int c = 0; c < numComms; c++) {
    recvBufs[c] = (char*)buf + c * count * sizeof(int) * numRanks;
    sendBufs[c] = (char*)recvBufs[c] + count * sizeof(int) * globalRank;

    // Assign different value for each comm
    std::vector<int> vals(count, 0);
    for (int i = 0; i < count; i++) {
      vals[i] = globalRank + c;
    }

    CUDACHECK_TEST(cudaMemcpy(
        sendBufs[c], vals.data(), count * sizeof(int), cudaMemcpyHostToDevice));

    // Register entire buffer range with all comms
    NCCLCHECK_TEST(ncclCommRegister(comms[c], buf, bufSize, &hdls[c]));
    ASSERT_NE(hdls[c], nullptr);
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Run collective on each communicator concurrently, each will be handled by a
  // separate GPE thread
  for (int c = 0; c < numComms; c++) {
    COMMCHECK_TEST(ctranAllGather(
        sendBufs[c],
        recvBufs[c],
        count,
        commInt,
        comms[c]->ctranComm_.get(),
        streams[c],
        NCCL_ALLGATHER_ALGO));
  }

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Expect all communicators shared the same buffer registration
  // Thus, we'd see only 1 additional handle in the cache
  allHandles = regCache->getSegments();
  EXPECT_EQ(allHandles.size(), 1 + commNumHdls);

  // Check that all values are correct
  for (int c = 0; c < numComms; c++) {
    std::vector<int> expVals(count * numRanks, 0);
    std::vector<int> vals(count * numRanks, -1);
    CUDACHECK_TEST(cudaMemcpy(
        vals.data(),
        recvBufs[c],
        count * sizeof(int) * numRanks,
        cudaMemcpyDeviceToHost));

    for (int i = 0; i < numRanks; i++) {
      for (int j = 0; j < count; j++) {
        expVals[i * count + j] = i + c;
      }
    }

    EXPECT_THAT(vals, ::testing::ElementsAreArray(expVals))
        << " at comm " << c << " buf " << recvBufs[c];
  }

  // Deregister buffers and destroy streams (comms destroyed by RAII)
  for (int c = 0; c < numComms; c++) {
    NCCLCHECK_TEST(ncclCommDeregister(comms[c], hdls[c]));
    CUDACHECK_TEST(cudaStreamDestroy(streams[c]));
  }

  NCCLCHECK_TEST(ncclMemFree(buf));
}

TEST_F(CommWithCtranTest, CommAbort) {
  EnvRAII env(NCCL_CTRAN_ENABLE, true);
  ncclComm_t comm = ncclx::test::createNcclComm(
      globalRank, numRanks, localRank, bootstrap_.get());

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  // Expect shared resource has been released properly
  ncclResult_t res = ncclCommAbort(comm);
  ASSERT_EQ(res, ncclSuccess);

  verifyPostCommResourceLeak();
}

TEST_F(CommWithCtranTest, CommAbortWithRegMem) {
  EnvRAII env(NCCL_CTRAN_ENABLE, true);
  ncclComm_t comm = ncclx::test::createNcclComm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  ctran::RegCache* regCache = ctran::RegCache::getInstance().get();
  ASSERT_NE(regCache, nullptr);

  auto allHandles = regCache->getSegments();
  size_t commNumHdls = allHandles.size();

  void* buf = nullptr;
  constexpr size_t count = 8192;
  size_t bufSize = count * sizeof(int) * numRanks;
  NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
  ASSERT_NE(buf, nullptr);
  void* hdl = nullptr;

  ncclCommRegister(comm, buf, bufSize, &hdl);
  ASSERT_NE(hdl, nullptr);

  if (!ctranAllGatherSupport(comm->ctranComm_.get(), NCCL_ALLGATHER_ALGO)) {
    ncclCommAbort(comm);
    GTEST_SKIP() << "ctranAllGather is not supported. Skip test";
  }
  // Run a collective to ensure buffer has been remote registered
  COMMCHECK_TEST(ctranAllGather(
      buf,
      buf,
      count,
      commInt,
      comm->ctranComm_.get(),
      0,
      NCCL_ALLGATHER_ALGO));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  allHandles = regCache->getSegments();
  EXPECT_EQ(allHandles.size(), 1 + commNumHdls);

  // Do not deregister buffer; expect commAbort to finish properly.
  // Resource leak is checked below
  ncclResult_t res = ncclCommAbort(comm);
  ASSERT_EQ(res, ncclSuccess);

  // Explicit trigger regCache destruction before check
  EXPECT_EQ(regCache->destroy(), commSuccess);

  NCCLCHECK_TEST(ncclMemFree(buf));

  verifyPostCommResourceLeak();
}

TEST_F(CommWithCtranTest, CommFailureWithInvalidTopology) {
  EnvRAII env(NCCL_CTRAN_ENABLE, true);
  const std::string invalidTopoFilepath = "/tmp/invalid_comm_topology.txt";
  std::ofstream invalidFile(invalidTopoFilepath);
  invalidFile << "DEVICE_NAME=testhost.facebook.com" << std::endl;
  // Invalid topology format - should have exactly 4 parts separated by '/'
  invalidFile << "DEVICE_BACKEND_NETWORK_TOPOLOGY=part1/part2/part3"
              << std::endl;
  invalidFile.close();

  EnvRAII topoFileEnv(NCCL_TOPO_FILE_PATH, invalidTopoFilepath);
  // Disable NCCL_IGNORE_TOPO_LOAD_FAILURE to ensure topology loading failure
  // causes comm creation to fail
  EnvRAII ignoreTopoFailureEnv(NCCL_IGNORE_TOPO_LOAD_FAILURE, false);

  EXPECT_THROW(
      {
        try {
          ncclx::test::createNcclComm(
              globalRank, numRanks, localRank, bootstrap_.get(), true);
        } catch (const std::exception& e) {
          EXPECT_THAT(e.what(), testing::HasSubstr("NCCL error"));
          throw;
        }
      },
      std::exception);

  // Clean up temporary files
  unlink(invalidTopoFilepath.c_str());
}

TEST_F(CommWithCtranTest, IgnoreCommFailureWithInvalidTopology) {
  EnvRAII env(NCCL_CTRAN_ENABLE, true);
  EnvRAII ignoreTopoFailureEnv(NCCL_IGNORE_TOPO_LOAD_FAILURE, true);
  const std::string dummyTopoFilepath = "/tmp/invalid_comm_topology.txt";
  std::ofstream dummyFile(dummyTopoFilepath);
  dummyFile << "DEVICE_NAME=testhost.facebook.com" << std::endl;
  dummyFile << "DEVICE_BACKEND_NETWORK_TOPOLOGY=part1/part2/part3" << std::endl;
  dummyFile.close();

  EnvRAII topoFileEnv(NCCL_TOPO_FILE_PATH, dummyTopoFilepath);

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get(), true};

  EXPECT_NE(nullptr, comm.get())
      << "Communicator creation should not fail when NCCL_IGNORE_TOPO_LOAD_FAILURE=1";

  // Clean up temporary files
  unlink(dummyTopoFilepath.c_str());
}

TEST_F(CommWithCtranTest, CtranOpCount) {
  EnvRAII env(NCCL_CTRAN_ENABLE, true);
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get(), true};

  // Update opCount mimic Ctran collectives
  constexpr int kNumOps = 10;
  auto ctranComm = comm->ctranComm_.get();
  for (int i = 0; i < kNumOps; ++i) {
    ctranComm->ctran_->updateOpCount();
  }

  // Expect that both external NCCL opCount and ctranOpCount have been updated
  EXPECT_EQ(ctranComm->ctran_->getOpCount(), kNumOps);
  EXPECT_EQ(ctranComm->ctran_->getCtranOpCount(), kNumOps);

  // Update opCount mimic baseline collectives
  for (int i = 0; i < kNumOps; ++i) {
    comm->opCount++;
  }

  // Expect only external NCCL opCount has been updated
  EXPECT_EQ(ctranComm->ctran_->getOpCount(), kNumOps * 2);
  EXPECT_EQ(ctranComm->ctran_->getCtranOpCount(), kNumOps);
}

namespace {
enum class TestCommCreateMode { kDefault, kSplit };
}
class CommWithCtranTestParam : public CommWithCtranTest,
                               public ::testing::WithParamInterface<
                                   std::tuple<TestCommCreateMode, bool>> {};

TEST_P(CommWithCtranTestParam, CtranEnableByHint) {
  const auto& [createMode, blockingInit] = GetParam();

  EnvRAII env(NCCL_CTRAN_ENABLE, false);
  // Default disabled
  ncclComm_t comm1 = ncclx::test::createNcclComm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(comm1, nullptr);
  ASSERT_FALSE(ctranInitialized(comm1->ctranComm_.get()));

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = blockingInit ? 1 : 0;
  ncclx::Hints ctranHints;
  config.hints = &ctranHints;

  // Enable by hint
  ASSERT_EQ(
      ncclx::setGlobalHint(std::string(ncclx::HintKeys::kCommUseCtran), "1"),
      ncclSuccess);
  ncclComm_t comm2;
  if (createMode == TestCommCreateMode::kDefault) {
    comm2 = ncclx::test::createNcclComm(
        globalRank, numRanks, localRank, bootstrap_.get(), false, &config);
  } else {
    ASSERT_EQ(
        ncclCommSplit(comm1, 1, globalRank, &comm2, &config), ncclSuccess);
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

  ASSERT_TRUE(ctranInitialized(comm2->ctranComm_.get()));
  ASSERT_TRUE(
      ncclx::resetGlobalHint(std::string(ncclx::HintKeys::kCommUseCtran)));

  // Now it should be disabled again after hint reset
  ncclComm_t comm3 = ncclx::test::createNcclComm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(comm3, nullptr);
  ASSERT_FALSE(ctranInitialized(comm3->ctranComm_.get()));

  ASSERT_EQ(ncclCommDestroy(comm3), ncclSuccess);
  ASSERT_EQ(ncclCommDestroy(comm2), ncclSuccess);
  ASSERT_EQ(ncclCommDestroy(comm1), ncclSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    CommWithCtranTestInstance,
    CommWithCtranTestParam,
    ::testing::Combine(
        ::testing::Values(
            TestCommCreateMode::kDefault,
            TestCommCreateMode::kSplit),
        ::testing::Values(true, false)),
    [&](const testing::TestParamInfo<CommWithCtranTestParam::ParamType>& info) {
      return fmt::format(
          "{}_{}",
          std::get<0>(info.param) == TestCommCreateMode::kDefault ? "default"
                                                                  : "split",
          std::get<1>(info.param) ? "blockingInit" : "nonblockingInit");
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
