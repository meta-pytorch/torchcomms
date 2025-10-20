// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

#include "comm.h"
#include "nccl.h"

#include "comms/ctran/memory/Utils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/transport/transportExt.h"

class NcclxMemDistTest : public NcclxBaseTest {
 protected:
  void SetUp() override {
    setenv("NCCL_USE_MEM_CACHE", "1", 1);
    setenv("NCCL_COLLTRACE", "trace", 1);
    setenv("NCCL_LAZY_SETUP_CHANNELS", "1", 1);
    setenv("NCCL_DEBUG_SUBSYS", "ALLOC", 0);
    setenv("NCCL_DEBUG", "INFO", 0);
    NcclxBaseTest::SetUp();
    config = NCCL_CONFIG_INITIALIZER;
    config.commDesc = "NcclxMemDistTest";
    config.splitShare = 0;
  }

 public:
  ncclConfig_t config{};
};

TEST_F(NcclxMemDistTest, InitOnly) {
  auto comm = createNcclComm(
      this->globalRank,
      this->numRanks,
      this->localRank,
      false,
      &config,
      server.get());
  ASSERT_NE(nullptr, comm);

  EXPECT_NE(comm->memCache, nullptr);
  EXPECT_EQ(comm->memCache->getNumUsedReg(), 0);

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}
TEST_F(NcclxMemDistTest, InitAbort) {
  EnvRAII commAbortScop(NCCL_COMM_ABORT_SCOPE, NCCL_COMM_ABORT_SCOPE::none);
  auto comm = createNcclComm(
      this->globalRank,
      this->numRanks,
      this->localRank,
      false,
      &config,
      server.get());
  ASSERT_NE(nullptr, comm);

  EXPECT_NE(comm->memCache, nullptr);
  EXPECT_EQ(comm->memCache->getNumUsedReg(), 0);

  NCCLCHECK_TEST(ncclCommAbort(comm));
}

TEST_F(NcclxMemDistTest, InitWithCtran) {
  EnvRAII<bool> ctranGaurd(NCCL_CTRAN_ENABLE, true);
  auto comm = createNcclComm(
      this->globalRank,
      this->numRanks,
      this->localRank,
      false,
      &config,
      server.get());
  ASSERT_NE(nullptr, comm);

  EXPECT_NE(comm->memCache, nullptr);
  EXPECT_EQ(comm->memCache->getNumUsedReg(), 0);

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(NcclxMemDistTest, allocateShareableBuffer) {
  EnvRAII<size_t> poolSizeGuard(NCCL_MEM_POOL_SIZE, 0);
  void* ptr = nullptr;
  size_t size = 1 << 21; // 2MB
  auto comm = createNcclComm(
      this->globalRank,
      this->numRanks,
      this->localRank,
      false,
      &config,
      server.get());
  size_t before, after, total;
  CUDACHECK_TEST(cudaMemGetInfo(&before, &total));
  ncclx::memory::allocatorIpcDesc ipcDesc;
  std::string use = "ncclx.ut";
  std::string key = folly::sformat("{}:{:#x}", use, comm->commHash);
  EXPECT_EQ(
      ncclx::memory::allocateShareableBuffer(
          size,
          /*refcount=*/0,
          &ipcDesc,
          &ptr,
          comm->memCache,
          &comm->logMetaData,
          use.c_str()),
      ncclSuccess);
  EXPECT_NE(ptr, nullptr);
  CUDACHECK_TEST(cudaMemGetInfo(&after, &total));
  EXPECT_EQ(before - after, size);
  if (comm->memCache) {
    EXPECT_EQ(comm->memCache->release({key}), ncclSuccess);
  } else {
    // can be freed by ncclCudaFree
    EXPECT_EQ(ncclCudaFree(ptr), ncclSuccess);
  }
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(NcclxMemDistTest, getTransportBufferKeys) {
  // disable lazy feature to allocate all buffers at init time
  EnvRAII lazyChannel(NCCL_LAZY_SETUP_CHANNELS, false);
  EnvRAII runtimeConn(NCCL_RUNTIME_CONNECT, 0L);
  EnvRAII disableMemCache(NCCL_USE_MEM_CACHE, false);

  auto comm = createNcclComm(this->globalRank, this->numRanks, this->localRank);
  ASSERT_NE(nullptr, comm);

  std::vector<std::string> keys;
  ncclx::transport::getTransportBufKeys(
      comm, &comm->graphs[NCCL_ALGO_RING], /*connIndex=*/0, keys);

  // expect to get keys to be more than 0
  EXPECT_GT(keys.size(), 0);

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
