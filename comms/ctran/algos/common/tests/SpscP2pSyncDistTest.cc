// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/ctran/algos/common/SpscP2pSync.h"
#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

using ctran::algos::SpscP2pSync;
using ctran::utils::CtranIpcDesc;
using ctran::utils::CtranIpcMem;
using ctran::utils::CtranIpcRemMem;

__global__ void SpscP2pSyncTestKernel(
    int myLocalRank,
    int numIter,
    int count, // number of elements in data
    int* shmData, // count elements
    SpscP2pSync* postSync,
    // received data from producer by consumer, total count * numIter elements
    // returned to host for test correctness check
    int* outData);

class SpscP2pSyncDistTest : public NcclxBaseTest {
 public:
  SpscP2pSyncDistTest() = default;

 protected:
  void SetUp() override {
    CUDACHECK_ASSERT(cudaSetDevice(cudaDev_));
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    NcclxBaseTest::SetUp();

    // Ensure cuda driver functions have been loaded
    COMMCHECK_ASSERT(ctran::utils::commCudaLibraryInit());

    if (!ctran::utils::CtranIpcSupport()) {
      GTEST_SKIP() << "CTran IPC is not supported on this platform. Skip test.";
    }

    commDeprecated_ = createNcclComm(globalRank, numRanks, localRank);
    comm_ = commDeprecated_->ctranComm_.get();

    if (comm_->statex_->nLocalRanks() != 2) {
      GTEST_SKIP() << "Requires 2 local ranks, skip test" << std::endl;
    }
    if (comm_->statex_->nNodes() > 1) {
      GTEST_SKIP() << "Requires single nodes, skip test" << std::endl;
    }

    CUDACHECK_ASSERT(cudaEventCreate(&start_));
    CUDACHECK_ASSERT(cudaEventCreate(&stop_));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(commDeprecated_));
    CUDACHECK_ASSERT(cudaEventDestroy(start_));
    CUDACHECK_ASSERT(cudaEventDestroy(stop_));
    NcclxBaseTest::TearDown();
  }

  void allGather(void* buf, const int len);
  void barrier();
  void initIpcBufs(const size_t nBytes, std::vector<void*>& remBufs);
  void releaseIpcBufs();

 protected:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  CtranComm* comm_{nullptr};
  ncclComm_t commDeprecated_{nullptr};
  const struct CommLogData dummyLogMetaData_ = {
      0,
      0xfaceb00c12345678 /*Dummy placeholder value for commHash*/,
      "TestComm",
      0,
      0};
  int cudaDev_{0};
  std::unique_ptr<CtranIpcMem> ipcMem_;
  std::vector<std::unique_ptr<CtranIpcRemMem>> ipcRemMems_;
};

void SpscP2pSyncDistTest::initIpcBufs(
    const size_t nBytes,
    std::vector<void*>& remBufs) {
  // Allocate local memory
  ASSERT_NO_THROW(
      ipcMem_ = std::make_unique<CtranIpcMem>(
          nBytes, localRank, &dummyLogMetaData_, "Test"));

  // Exchange with the other ranks on the same node.
  // Setup already checked all ranks on the same node
  const auto nRanks = comm_->statex_->nRanks();
  const auto rank = comm_->statex_->rank();

  std::vector<CtranIpcDesc> ipcDescs(nRanks);
  COMMCHECK_ASSERT(ipcMem_->ipcExport(ipcDescs[rank]));
  allGather(ipcDescs.data(), sizeof(CtranIpcDesc));

  // Import remote recvBuf from all other peers
  ipcRemMems_.resize(nRanks);
  remBufs.resize(nRanks, nullptr);
  try {
    for (int peer = 0; peer < nRanks; peer++) {
      if (peer == rank) {
        remBufs[peer] = ipcMem_->getBase();
        continue;
      }
      ipcRemMems_[peer] = std::make_unique<CtranIpcRemMem>(
          ipcDescs[peer], localRank, &dummyLogMetaData_, "Test");
      remBufs[peer] = ipcRemMems_[peer]->getBase();
    }
  } catch (std::exception& e) {
    GTEST_FAIL() << "Failed to import remote memory: " << e.what();
  }
}

void SpscP2pSyncDistTest::releaseIpcBufs() {
  COMMCHECK_ASSERT(ipcMem_->free());

  const auto nRanks = comm_->statex_->nRanks();
  const auto rank = comm_->statex_->rank();
  for (int peer = 0; peer < nRanks; peer++) {
    if (peer == rank) {
      continue;
    }
    auto& remMem = ipcRemMems_[peer];
    COMMCHECK_ASSERT(remMem->release());
  }
}

void SpscP2pSyncDistTest::barrier() {
  int nRanks = comm_->statex_->nRanks();
  int rank = comm_->statex_->rank();
  auto resFuture = comm_->bootstrap_->barrier(rank, nRanks);
  COMMCHECK_ASSERT(static_cast<commResult_t>(std::move(resFuture).get()));
}

void SpscP2pSyncDistTest::allGather(void* buf, const int len) {
  int nRanks = comm_->statex_->nRanks();
  int rank = comm_->statex_->rank();
  auto resFuture = comm_->bootstrap_->allGather(buf, len, rank, nRanks);
  COMMCHECK_ASSERT(static_cast<commResult_t>(std::move(resFuture).get()));
}

class SpscP2pSyncDistTestParamFixture
    : public SpscP2pSyncDistTest,
      public ::testing::WithParamInterface<int> {};

TEST_P(SpscP2pSyncDistTestParamFixture, Check) {
  auto count = GetParam();

  int numIter = 100;
  const auto myLocalRank = comm_->statex_->localRank();
  // rank 0 is producer, rank 1 is consumer
  const auto consumerRank = 1;
  const size_t dataSize = ctran::utils::align(count * sizeof(int), (size_t)16);

  // dataSize
  int* shmData = nullptr;
  // dataSize * numIter
  int* outputData = nullptr;

  std::vector<void*> remBufs;
  // shmData + sync on consumerRank; for simplicity, allocate on both ranks
  initIpcBufs(dataSize + sizeof(SpscP2pSync), remBufs);

  // local outputData on consumer rank
  if (consumerRank == myLocalRank) {
    CUDACHECK_ASSERT(cudaMalloc((void**)&outputData, dataSize * numIter));
  }

  SpscP2pSync* sync =
      reinterpret_cast<SpscP2pSync*>((char*)remBufs[consumerRank] + dataSize);
  shmData = reinterpret_cast<int*>(remBufs[consumerRank]);

  // consumer to initialize sync and data
  if (consumerRank == myLocalRank) {
    SpscP2pSync syncH = SpscP2pSync();
    CUDACHECK_ASSERT(
        cudaMemcpy(sync, &syncH, sizeof(SpscP2pSync), cudaMemcpyDefault));
    cudaMemset(shmData, 0, dataSize);
    cudaMemset(outputData, 0, dataSize * numIter);
  }
  // waits consumer finish initialization before both ranks start
  barrier();

  dim3 grid = {1, 1, 1};
  dim3 block = {256, 1, 1};
  void* execArgs[6] = {
      (void*)&myLocalRank,
      (void*)&numIter,
      (void*)&count,
      (void*)&shmData,
      (void*)&sync,
      (void*)&outputData};
  CUDACHECK_ASSERT(
      cudaLaunchKernel((void*)SpscP2pSyncTestKernel, grid, block, execArgs));
  CUDACHECK_ASSERT(cudaDeviceSynchronize());

  // consumer check output for each iteration
  if (consumerRank == myLocalRank) {
    std::vector<int> checkData(dataSize * numIter, -1);
    CUDACHECK_ASSERT(cudaMemcpy(
        checkData.data(), outputData, dataSize * numIter, cudaMemcpyDefault));
    for (int x = 0; x < numIter; x++) {
      const auto offset = x * count;
      auto it = checkData.begin() + offset;
      std::vector<int> dataIter = std::vector<int>(it, it + count);
      std::vector<int> expData(count);
      for (auto c = 0; c < count; c++) {
        expData[c] = x * count + c;
      }
      ASSERT_EQ(dataIter, expData)
          << "iter " << x << " at offset " << offset << std::endl;
#if 0
    std::cout << "iter " << x << " received data: "
              << folly::join(", ", dataIter) << std::endl;
#endif
    }
  }
  if (consumerRank == myLocalRank) {
    CUDACHECK_ASSERT(cudaFree(outputData));
  }
  releaseIpcBufs();
}

// TODO: convert to Google benchmark for dist environment
TEST_F(SpscP2pSyncDistTestParamFixture, Perf) {
  constexpr auto count = 0; // skip data copy

  int numWarm = 10;
  int numIter = 1000;
  int numRuns = 100;

  const auto myLocalRank = comm_->statex_->localRank();
  // rank 0 is producer, rank 1 is consumer
  const auto consumerRank = 1;

  // dataSize
  int *shmData = nullptr, *outputData = nullptr;

  std::vector<void*> remBufs;
  // sync on consumerRank; for simplicity, allocate on both ranks
  initIpcBufs(sizeof(SpscP2pSync), remBufs);

  SpscP2pSync* sync =
      reinterpret_cast<SpscP2pSync*>((char*)remBufs[consumerRank]);

  // consumer to initialize sync and data
  if (consumerRank == myLocalRank) {
    SpscP2pSync syncH = SpscP2pSync();
    CUDACHECK_ASSERT(
        cudaMemcpy(sync, &syncH, sizeof(SpscP2pSync), cudaMemcpyDefault));
  }
  // waits consumer finish initialization before both ranks start
  barrier();

  dim3 grid = {1, 1, 1};
  dim3 block = {256, 1, 1};
  void* execArgs[6] = {
      (void*)&myLocalRank,
      (void*)&numIter,
      (void*)&count,
      (void*)&shmData,
      (void*)&sync,
      (void*)&outputData};

  for (auto x = 0; x < numWarm; x++) {
    CUDACHECK_ASSERT(
        cudaLaunchKernel((void*)SpscP2pSyncTestKernel, grid, block, execArgs));
  }
  CUDACHECK_ASSERT(cudaDeviceSynchronize());

  CUDACHECK_ASSERT(cudaEventRecord(start_, 0));
  for (auto x = 0; x < numRuns; x++) {
    CUDACHECK_ASSERT(
        cudaLaunchKernel((void*)SpscP2pSyncTestKernel, grid, block, execArgs));
  }
  CUDACHECK_ASSERT(cudaEventRecord(stop_, 0));
  CUDACHECK_ASSERT(cudaEventSynchronize(stop_));
  CUDACHECK_ASSERT(cudaDeviceSynchronize());
  float timeMs;
  ASSERT_EQ(cudaEventElapsedTime(&timeMs, start_, stop_), cudaSuccess);

  std::cout
      << fmt::format(
             "Perf: {:.2f} us per iteration, averaged from {} runs, each with {} iterations",
             timeMs * 1e3 / numIter / numRuns,
             numRuns,
             numIter)
      << std::endl;
  releaseIpcBufs();
}

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    SpscP2pSyncDistTestParamFixture,
    ::testing::Values(1024, 15, 1048571),
    [&](const testing::TestParamInfo<
        SpscP2pSyncDistTestParamFixture::ParamType>& info) {
      return std::to_string(info.param) + "count";
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
