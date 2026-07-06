// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Numerical canaries for NCCL AllReduce. These tests are intended to catch NCCL
// upgrade regressions by comparing forced-algorithm results against an
// independently computed FP64 host reference.

#include <comm.h>
#include <cuda_bf16.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "comms/ncclx/meta/algoconf/AlgoStrConv.h"
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/ncclx/meta/tests/ReductionNumericalTestUtils.h"
#include "comms/ncclx/meta/tests/VerifyAlgoStatsUtil.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/NcclxConfig.h"

namespace {

enum class AllReduceNumericalAlgo {
  Ring,
  Tree,
  CtranDirect,
};

struct AllReduceNumericalParam {
  AllReduceNumericalAlgo algo;
  size_t count;
  ncclDataType_t datatype;
  ncclx::test::numerics::InputDistribution distribution =
      ncclx::test::numerics::InputDistribution::Uniform;

  std::string name() const {
    std::string algoName;
    switch (algo) {
      case AllReduceNumericalAlgo::Ring:
        algoName = "Ring";
        break;
      case AllReduceNumericalAlgo::Tree:
        algoName = "Tree";
        break;
      case AllReduceNumericalAlgo::CtranDirect:
        algoName = "CtranDirect";
        break;
    }

    const std::string dtypeName =
        datatype == ncclFloat32 ? "Float32" : "Bfloat16";
    std::string base =
        algoName + "_" + dtypeName + "_Count_" + std::to_string(count);
    if (distribution == ncclx::test::numerics::InputDistribution::Normal) {
      base += "_Normal";
    }
    return base;
  }
};

class AllReduceNumericalTest
    : public NcclxBaseTestFixture,
      public ::testing::WithParamInterface<AllReduceNumericalParam> {
 public:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp({
        {"NCCL_CTRAN_ENABLE", "1"},
        {"NCCL_CTRAN_IPC_REGCACHE_ENABLE_ASYNC_SOCKET", "1"},
    });
    ncclAlgoStats_.enable();
    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream_));
    NcclxBaseTestFixture::TearDown();
  }

 protected:
  template <typename T>
  void run(const AllReduceNumericalParam& param) {
    T* originalDevice = nullptr;
    T* actualDevice = nullptr;
    NCCLCHECK_TEST(
        ncclMemAlloc((void**)&originalDevice, param.count * sizeof(T)));
    NCCLCHECK_TEST(
        ncclMemAlloc((void**)&actualDevice, param.count * sizeof(T)));

    std::vector<T> original;
    ncclx::test::numerics::appendRandomInputs<T>(
        original, param.count, globalRank, 0, param.distribution);
    CUDACHECK_TEST(cudaMemcpyAsync(
        originalDevice,
        original.data(),
        param.count * sizeof(T),
        cudaMemcpyDefault,
        stream_));
    std::vector<double> reference(param.count, 0.0);
    std::vector<T> bf16Hop(param.count);
    {
      ncclx::test::NcclCommRAII referenceComm{
          globalRank, numRanks, localRank, bootstrap_.get()};
      const auto gathered = ncclx::test::numerics::gatherInputs(
          originalDevice,
          param.count,
          param.datatype,
          referenceComm,
          stream_,
          numRanks);
      std::vector<double> contributions(numRanks);
      for (size_t i = 0; i < param.count; ++i) {
        for (int rank = 0; rank < numRanks; ++rank) {
          contributions[rank] = static_cast<double>(DataTypeTraits<T>::toHost(
              gathered[static_cast<size_t>(rank) * param.count + i]));
          reference[i] += contributions[rank];
        }
        bf16Hop[i] = ncclx::test::numerics::bf16HopReduce<T>(contributions);
      }
    }

    std::optional<SysEnvRAII> ncclAlgoGuard;
    if (param.algo == AllReduceNumericalAlgo::Ring) {
      ncclAlgoGuard.emplace("NCCL_ALGO", "RING");
    } else if (param.algo == AllReduceNumericalAlgo::Tree) {
      ncclAlgoGuard.emplace("NCCL_ALGO", "TREE");
    }

    ncclx::Hints hints;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    if (param.algo == AllReduceNumericalAlgo::Ring ||
        param.algo == AllReduceNumericalAlgo::Tree) {
      hints.set("allreduceAlgo", "orig");
    } else {
      hints.set(
          "allreduceAlgo",
          ncclx::algoconf::algoValToStr(NCCL_ALLREDUCE_ALGO::ctdirect));
    }
    config.hints = &hints;

    // NCCL_CONFIG_INITIALIZER sets versioned ncclConfig_t fields; Infer misses
    // this through NcclCommRAII.
    // @lint-ignore PULSE_UNINITIALIZED_VALUE
    ncclx::test::NcclCommRAII comm{
        globalRank, numRanks, localRank, bootstrap_.get(), false, &config};
    if (param.algo == AllReduceNumericalAlgo::CtranDirect) {
      EXPECT_EQ(
          NCCLX_CONFIG_FIELD(comm->config, allreduceAlgo),
          NCCL_ALLREDUCE_ALGO::ctdirect);
      ASSERT_NE(comm->ctranComm_, nullptr);
    }

    const auto result = ncclAllReduce(
        originalDevice,
        actualDevice,
        param.count,
        param.datatype,
        ncclSum,
        comm,
        stream_);
    ASSERT_EQ(result, ncclSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    ncclx::test::numerics::printActualOutputBytes(
        actualDevice,
        param.count,
        stream_,
        globalRank,
        "AllReduce",
        param.name());
    ncclx::test::numerics::printReferenceBytes(
        reference, globalRank, "AllReduce", param.name());
    if (param.datatype == ncclBfloat16) {
      ncclx::test::numerics::printBf16HopBytes(
          bf16Hop, globalRank, "AllReduce", param.name());
    }

    const size_t mismatches = ncclx::test::numerics::countMismatches(
        actualDevice, reference, stream_, globalRank, param.name());
    EXPECT_EQ(mismatches, 0) << param.name() << " rank=" << globalRank;

    if (param.algo == AllReduceNumericalAlgo::Ring) {
      ncclAlgoStats_.verify(comm, "AllReduce", "RING");
    } else if (param.algo == AllReduceNumericalAlgo::Tree) {
      ncclAlgoStats_.verify(comm, "AllReduce", "TREE");
    }

    NCCLCHECK_TEST(ncclMemFree(originalDevice));
    NCCLCHECK_TEST(ncclMemFree(actualDevice));
  }

  cudaStream_t stream_{nullptr};
  ncclx::test::VerifyAlgoStatsHelper ncclAlgoStats_;
};

TEST_P(AllReduceNumericalTest, MatchesFp64Reference) {
  const auto& param = GetParam();
  if (param.datatype == ncclFloat32) {
    run<float>(param);
  } else if (param.datatype == ncclBfloat16) {
    run<__nv_bfloat16>(param);
  } else {
    FAIL() << "Unhandled datatype in " << param.name();
  }
}

const auto kAllReduceNumericalParams = ::testing::Values(
    AllReduceNumericalParam{
        .algo = AllReduceNumericalAlgo::Ring,
        .count = 1,
        .datatype = ncclFloat32},
    AllReduceNumericalParam{
        .algo = AllReduceNumericalAlgo::Ring,
        .count = 8193,
        .datatype = ncclFloat32},
    AllReduceNumericalParam{
        .algo = AllReduceNumericalAlgo::Ring,
        .count = 1024,
        .datatype = ncclBfloat16},
    AllReduceNumericalParam{
        .algo = AllReduceNumericalAlgo::Tree,
        .count = 1,
        .datatype = ncclFloat32},
    AllReduceNumericalParam{
        .algo = AllReduceNumericalAlgo::Tree,
        .count = 8193,
        .datatype = ncclFloat32},
    AllReduceNumericalParam{
        .algo = AllReduceNumericalAlgo::Tree,
        .count = 1024,
        .datatype = ncclBfloat16},
    AllReduceNumericalParam{
        .algo = AllReduceNumericalAlgo::CtranDirect,
        .count = 1024,
        .datatype = ncclFloat32},
    AllReduceNumericalParam{
        .algo = AllReduceNumericalAlgo::CtranDirect,
        .count = 4099,
        .datatype = ncclBfloat16},
    AllReduceNumericalParam{
        .algo = AllReduceNumericalAlgo::Ring,
        .count = 8193,
        .datatype = ncclFloat32,
        .distribution = ncclx::test::numerics::InputDistribution::Normal},
    AllReduceNumericalParam{
        .algo = AllReduceNumericalAlgo::Ring,
        .count = 1024,
        .datatype = ncclBfloat16,
        .distribution = ncclx::test::numerics::InputDistribution::Normal}
#ifdef REDUCTION_NUMERICAL_LARGE_COUNT_TEST
    ,
    AllReduceNumericalParam{
        .algo = AllReduceNumericalAlgo::Ring,
        .count = 262144,
        .datatype = ncclFloat32},
    AllReduceNumericalParam{
        .algo = AllReduceNumericalAlgo::Tree,
        .count = 262144,
        .datatype = ncclFloat32},
    AllReduceNumericalParam{
        .algo = AllReduceNumericalAlgo::CtranDirect,
        .count = 65536,
        .datatype = ncclBfloat16}
#endif
);

INSTANTIATE_TEST_SUITE_P(
    AllReduce,
    AllReduceNumericalTest,
    kAllReduceNumericalParams,
    [](const ::testing::TestParamInfo<AllReduceNumericalParam>& info) {
      return info.param.name();
    });

} // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
