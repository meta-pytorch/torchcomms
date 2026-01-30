// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <cuda_bf16.h>
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
  enum NCCL_REDUCESCATTER_ALGO algo;
  bool inplace;
  bool registFlag;
  MemAllocType memType;
  size_t count;
  ncclRedOp_t op = ncclSum;
  ncclDataType_t datatype = ncclInt;

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
    using Traits = DataTypeTraits<T>;
    using HostT = typename Traits::HostT;

    const auto algo = param.algo;
    const auto inplace = param.inplace;
    const auto registFlag = param.registFlag;
    const auto memType = param.memType;
    const auto count = param.count;
    const auto op = param.op;
    const auto datatype = param.datatype;

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

    // Initialize send buffer: each rank's chunk r has constant value
    // (globalRank * numRanks + r)
    for (int r = 0; r < numRanks; r++) {
      HostT val = static_cast<HostT>(globalRank * numRanks + r);
      assignChunkValue(sendBuf + r * count, count, Traits::toDevice(val));
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

    // Calculate expected value: sum of (r * numRanks + globalRank) for all r
    HostT expectedVal = static_cast<HostT>(0);
    for (int r = 0; r < numRanks; r++) {
      expectedVal += static_cast<HostT>(r * numRanks + globalRank);
    }
    if (op == ncclAvg) {
      expectedVal = expectedVal / static_cast<HostT>(numRanks);
    }

    // Verify results using checkChunkValue with type-appropriate tolerance
    size_t errs = checkChunkValue(
        recvBuf,
        count,
        Traits::toDevice(expectedVal),
        T{0},
        globalRank,
        stream,
        Traits::tolerance());
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
  ncclComm_t comm;
  cudaStream_t stream;
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
  };

  if (param.datatype == ncclInt) {
    run<int>(param);
  } else if (param.datatype == ncclFloat) {
    run<float>(param);
  }
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

// Parameters: inplace, count, datatype
// Tests native PAT AVG implementation with NCCL_ALGO=reducescatter:pat_postdiv
class ReduceScatterPatAvgTestParam
    : public ReduceScatterTest,
      public ::testing::WithParamInterface<
          std::tuple<bool, size_t, ncclDataType_t>> {};

TEST_P(ReduceScatterPatAvgTestParam, PatAvgTest) {
  auto [inplace, count, datatype] = GetParam();

  // Enable native PAT AVG mode using unified NCCL_ALGO syntax
  auto algoGuard = EnvRAII(NCCL_ALGO, std::string("reducescatter:pat_postdiv"));
  auto protoGuard = EnvRAII(NCCL_PROTO, std::string("Simple"));

  ReduceScatterTestParams param{
      .algo = NCCL_REDUCESCATTER_ALGO::orig, // Use orig algo, PAT is selected
                                             // via NCCL_ALGO cvar
      .inplace = inplace,
      .registFlag = false,
      .memType = kMemNcclMemAlloc,
      .count = count,
      .op = ncclAvg,
      .datatype = datatype,
  };

  if (datatype == ncclInt) {
    run<int>(param);
  } else if (datatype == ncclFloat) {
    run<float>(param);
  } else if (datatype == ncclDouble) {
    run<double>(param);
  } else if (datatype == ncclBfloat16) {
    run<__nv_bfloat16>(param);
  }
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

INSTANTIATE_TEST_SUITE_P(
    ReduceScatterPatAvgTestInstance,
    ReduceScatterPatAvgTestParam,
    ::testing::Combine(
        ::testing::Values(true, false), // inplace
        ::testing::Values(1, 95, 8000, 33554430), // count per rank
        ::testing::Values(
            ncclInt,
            ncclFloat,
            ncclDouble,
            ncclBfloat16)), // datatype
    [](const testing::TestParamInfo<std::tuple<bool, size_t, ncclDataType_t>>&
           info) {
      auto inplace = std::get<0>(info.param);
      auto count = std::get<1>(info.param);
      auto datatype = std::get<2>(info.param);
      std::string dtStr;
      switch (datatype) {
        case ncclInt:
          dtStr = "Int";
          break;
        case ncclFloat:
          dtStr = "Float";
          break;
        case ncclDouble:
          dtStr = "Double";
          break;
        case ncclBfloat16:
          dtStr = "Bfloat16";
          break;
        default:
          dtStr = "Unknown";
      }
      return fmt::format(
          "PatAvg_{}_{}_{}count",
          inplace ? "Inplace" : "OutOfPlace",
          dtStr,
          count);
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
