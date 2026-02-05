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
#include <optional>
#include <unordered_map>
#include "comms/ctran/Ctran.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"
#include "meta/colltrace/AlgoStats.h"
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
    // [META:PAT_AVG] Enable PAT before any comm creation to ensure the
    // NCCL_PARAM cache gets the right value. This is needed because NCCL_PARAM
    // uses static caching - once set, the value is never re-read.
    patEnableGuard_.emplace(NCCL_PAT_ENABLE, (int64_t)1);
    // Enable AlgoStats before comm creation for algorithm validation tests
    algoStatGuard_.emplace(
        NCCL_COLLTRACE, std::vector<std::string>{"algostat"});
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    NcclxBaseTest::TearDown();
  }

  // Verify that the expected algorithm was used via AlgoStats
  void verifyAlgoStats(
      const std::string& collective,
      const std::string& expectedAlgoSubstr) {
    std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>
        stats;
    ncclx::colltrace::dumpAlgoStat(comm, stats);

    auto it = stats.find(collective);
    ASSERT_NE(it, stats.end())
        << collective << " not found in AlgoStats. Stats may not be enabled.";

    bool foundExpectedAlgo = false;
    std::string foundAlgos;
    for (const auto& [algoName, callCount] : it->second) {
      if (!foundAlgos.empty()) {
        foundAlgos += ", ";
      }
      foundAlgos += fmt::format("{}({})", algoName, callCount);
      if (algoName.find(expectedAlgoSubstr) != std::string::npos &&
          callCount > 0) {
        foundExpectedAlgo = true;
      }
    }
    EXPECT_TRUE(foundExpectedAlgo)
        << "Expected algorithm containing '" << expectedAlgoSubstr
        << "' not found in " << collective << ". Found algorithms: ["
        << foundAlgos << "]";
  }

  // TODO: Add separate benchmark for performance testing
  // expectedAlgoSubstr: optional validation string; if set, verify algorithm
  // via AlgoStats after collective
  template <typename T>
  void run(
      const ReduceScatterTestParams& param,
      const std::string& expectedAlgoSubstr = "") {
    using Traits = DataTypeTraits<T>;
    using HostT = typename Traits::HostT;

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

    // Create comm after environment variables are set
    NcclCommRAII commGuard{globalRank, numRanks, localRank};
    comm = commGuard.get();

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

    // TODO: AlgoStats doesn't trace ctran algos yet, skip check for non-orig
    if (!expectedAlgoSubstr.empty() && algo == NCCL_REDUCESCATTER_ALGO::orig) {
      verifyAlgoStats("ReduceScatter", expectedAlgoSubstr);
    }

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
  std::optional<EnvRAII<int64_t>> patEnableGuard_;
  std::optional<EnvRAII<std::vector<std::string>>> algoStatGuard_;
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

// Parameters: inplace, count, datatype
// Tests native PAT AVG implementation with NCCL_REDUCESCATTER_PAT_AVG_ENABLE=1
class ReduceScatterPatAvgTestParam
    : public ReduceScatterTest,
      public ::testing::WithParamInterface<
          std::tuple<bool, size_t, ncclDataType_t>> {};

TEST_P(ReduceScatterPatAvgTestParam, PatAvgTest) {
  auto [inplace, count, datatype] = GetParam();

  auto algoGuard = EnvRAII<std::string>(NCCL_ALGO, "reducescatter:pat");
  auto protoGuard = EnvRAII<std::string>(NCCL_PROTO, "Simple");
  auto patAvgGuard = EnvRAII(NCCL_REDUCESCATTER_PAT_AVG_ENABLE, true);

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
    run<int>(param, "PAT");
  } else if (datatype == ncclFloat) {
    run<float>(param, "PAT");
  } else if (datatype == ncclDouble) {
    run<double>(param, "PAT");
  } else if (datatype == ncclBfloat16) {
    run<__nv_bfloat16>(param, "PAT");
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

std::string GetPatAvgTestParamName(
    const testing::TestParamInfo<ReduceScatterPatAvgTestParam::ParamType>&
        info) {
  ReduceScatterTestParams params{
      .algo = NCCL_REDUCESCATTER_ALGO::orig,
      .inplace = std::get<0>(info.param),
      .registFlag = false,
      .memType = kMemNcclMemAlloc,
      .count = std::get<1>(info.param),
      .op = ncclAvg,
      .datatype = std::get<2>(info.param),
  };
  return params.name();
}

INSTANTIATE_TEST_SUITE_P(
    ReduceScatterPatAvgTestInstance,
    ReduceScatterPatAvgTestParam,
    ::testing::Combine(
        ::testing::Values(true, false), // inplace
        ::testing::Values(1, 8000, 33554430), // count per rank
        ::testing::Values(
            ncclInt,
            ncclFloat,
            ncclDouble,
            ncclBfloat16)), // datatype
    GetPatAvgTestParamName);

// Test PAT algorithm selection with different PAT_AVG settings and ncclOp
// Also validates algorithm via AlgoStats
// Parameters: PAT_AVG_ENABLE (bool), ncclRedOp_t, expectedAlgoSubstr
class ReduceScatterPatAlgoSelectionTest
    : public ReduceScatterTest,
      public ::testing::WithParamInterface<
          std::tuple<bool, ncclRedOp_t, std::string>> {};

TEST_P(ReduceScatterPatAlgoSelectionTest, AlgoSelection) {
  auto [patAvgEnable, op, expectedAlgoSubstr] = GetParam();

  auto algoGuard = EnvRAII<std::string>(NCCL_ALGO, "reducescatter:pat");
  auto protoGuard = EnvRAII<std::string>(NCCL_PROTO, "Simple");
  auto patAvgGuard = EnvRAII(NCCL_REDUCESCATTER_PAT_AVG_ENABLE, patAvgEnable);

  ReduceScatterTestParams param{
      .algo = NCCL_REDUCESCATTER_ALGO::orig,
      .inplace = false,
      .registFlag = false,
      .memType = kMemNcclMemAlloc,
      .count = 8000,
      .op = op,
      .datatype = ncclFloat,
  };

  run<float>(param, expectedAlgoSubstr);
}

INSTANTIATE_TEST_SUITE_P(
    ReduceScatterPatAlgoSelectionTestInstance,
    ReduceScatterPatAlgoSelectionTest,
    ::testing::Values(
        // PAT_AVG_ENABLE=true: ALL operations use PAT (including Avg)
        // TODO: we test only SUM, AVG for now; we need extend the test to cover
        // other ops
        std::make_tuple(true, ncclSum, "PAT"),
        std::make_tuple(true, ncclAvg, "PAT"),
        // // PAT_AVG_ENABLE=false: Sum uses PAT, AVG fails
        std::make_tuple(false, ncclSum, "PAT")),
    [](const testing::TestParamInfo<std::tuple<bool, ncclRedOp_t, std::string>>&
           info) {
      auto patAvgEnable = std::get<0>(info.param);
      auto op = std::get<1>(info.param);
      auto expectedAlgo = std::get<2>(info.param);
      return fmt::format(
          "PAT_AVG_{}_{}_{}",
          patAvgEnable ? "on" : "off",
          getRedOpStr(op),
          expectedAlgo);
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
