// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <fmt/core.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include "comms/ctran/Ctran.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"
#include "meta/wrapper/DataTypeStrUtils.h"

struct ReduceScatterTestParams {
  enum NCCL_REDUCESCATTER_ALGO algo { NCCL_REDUCESCATTER_ALGO::orig };
  bool inplace{false};
  bool registFlag{false};
  MemAllocType memType{kMemCudaMalloc};
  size_t count{0};
  ncclRedOp_t op{ncclSum};
  ncclDataType_t datatype{ncclInt};

  std::string name() const {
    return fmt::format(
        "{}_{}_{}_{}_{}count_{}_{}",
        reduceScatterAlgoName(algo),
        inplace ? "Inplace" : "OutOfPlace",
        registFlag ? "Regist" : "NoRegist",
        testMemAllocTypeToStr(memType),
        count,
        getRedOpStr(op),
        getDatatypeStr(datatype));
  }
};

class ReduceScatterTest : public NcclxBaseTest {
 public:
  ReduceScatterTest() = default;
  void SetUp() override {
    NcclxBaseTest::SetUp();
    comm = createNcclComm(globalRank, numRanks, localRank);
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    NcclxBaseTest::TearDown();
  }

  // TODO: Add separate benchmark for performance testing
  template <typename T>
  void run(const ReduceScatterTestParams& param) {
    const auto algo = param.algo;
    const auto inplace = param.inplace;
    const auto registFlag = param.registFlag;
    const auto memType = param.memType;
    const auto count = param.count;
    const auto op = param.op;
    const auto datatype = param.datatype;

    // Validate supported reduction operations
    if (op != ncclSum && op != ncclAvg) {
      GTEST_SKIP() << "Only ncclSum and ncclAvg reduction ops are supported";
    }

    auto envGuard = EnvRAII(NCCL_REDUCESCATTER_ALGO, algo);

    if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
      GTEST_SKIP() << "CuMem not supported, skip test";
    }

#if !defined TEST_ENABLE_CTRAN
    if (algo != NCCL_REDUCESCATTER_ALGO::orig) {
      GTEST_SKIP() << "Ctran is disabled, skip test";
    }
#endif

    if (algo != NCCL_REDUCESCATTER_ALGO::orig &&
        !ctranReduceScatterSupport(comm->ctranComm_.get(), algo)) {
      GTEST_SKIP() << "Ctran algorithm is not supported, skip test";
    }

    if (memType == kMemCudaMalloc && algo != NCCL_REDUCESCATTER_ALGO::orig &&
        comm->ctranComm_->statex_->nLocalRanks() > 1) {
      GTEST_SKIP()
          << "Ctran does not support cudaMalloc-ed buffer with nLocalRanks > 1, skip test";
    }

    constexpr size_t elemSize = sizeof(T);
    size_t allocSize = count * numRanks * elemSize;
    allocSize = allocSize < 8192 ? 8192 : allocSize;

    T *sendBuf = nullptr, *recvBuf = nullptr;
    void *sendHandle = nullptr, *recvHandle = nullptr;

    if (memType == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaMalloc(&sendBuf, allocSize));
    } else {
      NCCLCHECK_TEST(ncclMemAlloc((void**)&sendBuf, allocSize));
    }

    if (inplace) {
      recvBuf = sendBuf + count * globalRank;
    } else {
      if (memType == kMemCudaMalloc) {
        CUDACHECK_TEST(cudaMalloc(&recvBuf, allocSize));
      } else {
        NCCLCHECK_TEST(ncclMemAlloc((void**)&recvBuf, allocSize));
      }
    }

    assignChunkValue<T>(recvBuf, count, static_cast<T>(-1));
    for (int r = 0; r < numRanks; r++) {
      T val = static_cast<T>(globalRank * numRanks + r);
      assignChunkValue<T>(sendBuf + r * count, count, val);
    }

    if (registFlag) {
      NCCLCHECK_TEST(ncclCommRegister(comm, sendBuf, allocSize, &sendHandle));
      if (!inplace) {
        NCCLCHECK_TEST(ncclCommRegister(comm, recvBuf, allocSize, &recvHandle));
      }
    }

    // Run communication
    auto res =
        ncclReduceScatter(sendBuf, recvBuf, count, datatype, op, comm, stream);
    ASSERT_EQ(res, ncclSuccess);

    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Check received chunk
    T expectedSum = static_cast<T>(0);
    for (int r = 0; r < numRanks; r++) {
      expectedSum += static_cast<T>(r * numRanks + globalRank);
    }
    if (op == ncclAvg) {
      expectedSum = expectedSum / static_cast<T>(numRanks);
    }

    std::optional<T> tolerance = std::nullopt;
    if constexpr (std::is_floating_point_v<T>) {
      tolerance = static_cast<T>(1e-5);
    }

    auto errs = checkChunkValue<T>(
        recvBuf, count, expectedSum, T{0}, globalRank, nullptr, tolerance);
    EXPECT_EQ(errs, 0) << "Rank " << globalRank << " checked chunk at "
                       << recvBuf << " with " << errs << " errors with inplace "
                       << inplace;

    // Deregister and free buffers
    if (registFlag) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
      if (!inplace) {
        NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
      }
    }

    if (memType == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaFree(sendBuf));
    } else {
      NCCLCHECK_TEST(ncclMemFree(sendBuf));
    }
    if (!inplace) {
      if (memType == kMemCudaMalloc) {
        CUDACHECK_TEST(cudaFree(recvBuf));
      } else {
        NCCLCHECK_TEST(ncclMemFree(recvBuf));
      }
    }
  }

 protected:
  ncclComm_t comm{nullptr};
  cudaStream_t stream{nullptr};
};

class ReduceScatterTestParam : public ReduceScatterTest,
                               public ::testing::WithParamInterface<std::tuple<
                                   enum NCCL_REDUCESCATTER_ALGO,
                                   bool,
                                   bool,
                                   MemAllocType,
                                   size_t>> {};

TEST_P(ReduceScatterTestParam, Test) {
  auto [algo, inplace, registFlag, memType, count] = GetParam();
  ReduceScatterTestParams param{
      .algo = algo,
      .inplace = inplace,
      .registFlag = registFlag,
      .memType = memType,
      .count = count,
      .op = ncclSum,
      .datatype = ncclInt,
  };

  run<int>(param);
}

std::string GetTestParamName(
    const testing::TestParamInfo<ReduceScatterTestParam::ParamType>& info) {
  ReduceScatterTestParams params{
      .algo = std::get<0>(info.param),
      .inplace = std::get<1>(info.param),
      .registFlag = std::get<2>(info.param),
      .memType = std::get<3>(info.param),
      .count = std::get<4>(info.param),
  };
  return params.name();
}

INSTANTIATE_TEST_SUITE_P(
    ReduceScatterTestInstance,
    ReduceScatterTestParam,
    ::testing::Combine(
        ::testing::Values(
            NCCL_REDUCESCATTER_ALGO::orig,
            NCCL_REDUCESCATTER_ALGO::ctran,
            NCCL_REDUCESCATTER_ALGO::ctrhd,
            NCCL_REDUCESCATTER_ALGO::ctring),
        ::testing::Values(true, false), // inplace
        ::testing::Values(true), // registFlag
        ::testing::Values(kMemCudaMalloc, kMemNcclMemAlloc), // memType
        ::testing::Values(1, 8192, 33554432) // count: small, medium, large
        ),
    GetTestParamName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
