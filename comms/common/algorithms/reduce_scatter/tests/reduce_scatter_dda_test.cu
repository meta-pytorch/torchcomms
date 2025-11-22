#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>
#include "comms/common/IpcGpuBarrier.cuh"
#include "comms/common/algorithms/reduce_scatter/reduce_scatter_dda.cuh"
#include "comms/common/tests/TestBaselineBootstrap.h"
#include "comms/rcclx/develop/meta/lib/tests/RcclxTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::rcclx;
using namespace meta::comms;

namespace {
constexpr int NUMRANKS = 8;
const int cnt = 1024 * 1024;
const int nBlocks = 32;
const int nThreads = 128;
} // namespace

template <typename ElementType>
class ReduceScatterDdaTest : public RcclxBaseTestFixture {
 public:
  void SetUp() override {
    RcclxBaseTestFixture::SetUp();

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    NCCL_CHECK(
        ncclCommInitRankConfig(&comm, numRanks, commId, globalRank, &config));
    XLOGF(INFO, "rank {} init done; total ranks: {}", globalRank, numRanks);

    ASSERT_EQ(numRanks, NUMRANKS);
    CUDA_CHECK(cudaSetDevice(globalRank));

    auto bootstrap = std::make_shared<TestBaselineBootstrap>(comm);
    memHandler =
        std::make_unique<IpcMemHandler>(bootstrap, globalRank, NUMRANKS);
    ipcBuf =
        std::make_unique<DeviceBuffer>(sizeof(ElementType) * NUMRANKS * cnt);
    memHandler->addSelfDeviceMemPtr(ipcBuf->get());
    memHandler->exchangeMemPtrs();

    void* ipcBufs[NUMRANKS];
    for (int i = 0; i < NUMRANKS; ++i) {
      ipcBufs[i] = memHandler->getPeerDeviceMemPtr(i);
    }
    allRankIpcBufs =
        std::make_unique<DeviceBuffer>(sizeof(ElementType*) * NUMRANKS);
    CUDA_CHECK(cudaMemcpy(
        allRankIpcBufs->get(),
        ipcBufs,
        sizeof(ElementType*) * NUMRANKS,
        cudaMemcpyHostToDevice));

    auto barrierInit =
        IpcGpuBarrier::mallocAndInit(numRanks, nBlocks, globalRank, bootstrap);
    barrierResources = std::move(barrierInit.first);
    barrier = std::move(barrierInit.second);
  }

  void TearDown() override {
    ncclCommFinalize(comm);
    ncclCommDestroy(comm);
    RcclxBaseTestFixture::TearDown();
  }

 public:
  ncclComm_t comm{nullptr};
  std::unique_ptr<DeviceBuffer> ipcBuf;
  std::unique_ptr<DeviceBuffer> allRankIpcBufs;
  std::unique_ptr<IpcMemHandler> memHandler;
  std::unique_ptr<IpcGpuBarrierResources> barrierResources;
  IpcGpuBarrier barrier;
};

TYPED_TEST_SUITE_P(ReduceScatterDdaTest);

// The range of the elements in the data arrary
const int RAND_RANGE = 100;

__global__ void initRand(curandState_t* randStates, int size, int nRanks) {
  for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    for (int i = 0; i < nRanks; i++) {
      // The init param of curandState is same for a given randState idx on any
      // ranks, so it will generate the same sequence of random numbers. This
      // allows us to compute the ground truth of allReduce locally without peer
      // rank communication.
      curand_init(
          i + 1 /* seed */,
          idx /* sequence */,
          0 /* offset */,
          &randStates[idx * nRanks + i]);
    }
  }
}

template <typename T>
__global__ void genData(
    curandState_t* randStates,
    T* data,
    T* acc,
    T* groundTruth,
    int selfRank,
    int nRanks,
    int size) {
  for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    T sum = 0;
    for (int i = 0; i < nRanks; ++i) {
      // double val = curand_uniform_double(&randStates[idx * nRanks + i]) *
      // RAND_RANGE;
      double val = RAND_RANGE + i;

      // downcast to T
      T hval = val;
      sum += hval;

      if (i == selfRank) {
        data[idx] = hval;
      }
    }
    groundTruth[idx] = sum;
  }
}

TYPED_TEST_P(ReduceScatterDdaTest, ddaReduceScatterIpcTest) {
  using ElementType = TypeParam;

  // The IpcGpuBarrier requires numThreads >= numRanks
  ASSERT_GE(nThreads, NUMRANKS);

  // we do 128 bit load in the dda kernel, so the data memory must align with
  // 128 bits
  ASSERT_EQ(cnt * sizeof(ElementType) % sizeof(uint4), 0);

  // prepare the sendbuff on each rank
  DeviceBuffer randStateBuf(sizeof(curandState_t) * NUMRANKS * NUMRANKS * cnt);
  curandState_t* randStates_d =
      reinterpret_cast<curandState_t*>(randStateBuf.get());
  initRand<<<nBlocks, nThreads>>>(randStates_d, NUMRANKS * cnt, NUMRANKS);

  DeviceBuffer sendbuf(sizeof(ElementType) * NUMRANKS * cnt);
  ElementType* sendbuf_d = reinterpret_cast<ElementType*>(sendbuf.get());

  DeviceBuffer accbuf(sizeof(ElementType) * NUMRANKS * cnt);
  ElementType* accbuf_d = reinterpret_cast<ElementType*>(accbuf.get());

  DeviceBuffer groundTruth(sizeof(ElementType) * NUMRANKS * cnt);
  ElementType* groundTruth_d =
      reinterpret_cast<ElementType*>(groundTruth.get());

  genData<<<nBlocks, nThreads>>>(
      randStates_d,
      sendbuf_d,
      accbuf_d,
      groundTruth_d,
      this->globalRank,
      NUMRANKS,
      NUMRANKS * cnt);

  DeviceBuffer recvbuff(sizeof(ElementType) * cnt);
  ElementType* recvbuff_d = reinterpret_cast<ElementType*>(recvbuff.get());
  ddaReduceScatterIpc<ElementType, NUMRANKS, false /*hasAcc*/>
      <<<nBlocks, nThreads>>>(
          (ElementType**)this->allRankIpcBufs->get(),
          recvbuff_d,
          cnt,
          sendbuf_d,
          this->globalRank,
          this->barrier,
          accbuf_d);

  cudaDeviceSynchronize();

  // compare with ground truth
  ElementType myresults_h[cnt];
  ElementType groundTruth_h[cnt];
  CUDA_CHECK(cudaMemcpy(
      myresults_h, recvbuff_d, sizeof(ElementType) * cnt, cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(
      groundTruth_h,
      groundTruth_d,
      sizeof(ElementType) * cnt,
      cudaMemcpyDefault));
  for (int i = 0; i < cnt; ++i) {
    EXPECT_EQ(
        static_cast<double>(myresults_h[i]),
        static_cast<double>(groundTruth_h[i]));
  }
}

REGISTER_TYPED_TEST_SUITE_P(ReduceScatterDdaTest, ddaReduceScatterIpcTest);
using TypesToTest = ::testing::Types<half, __nv_bfloat16>;
INSTANTIATE_TYPED_TEST_SUITE_P(
    ReduceScatterDdaTests,
    ReduceScatterDdaTest,
    TypesToTest);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
