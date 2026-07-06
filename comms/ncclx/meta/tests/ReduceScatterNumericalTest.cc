// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Numerical canaries for NCCL ReduceScatter. These tests are intended to catch
// NCCL upgrade regressions by comparing forced-algorithm results against an
// independently computed FP64 host reference for each rank's output chunk.

#include <comm.h>
#include <cuda_bf16.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include <optional>
#include <string>

#include "comms/ctran/Ctran.h"
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/ncclx/meta/tests/ReductionNumericalTestUtils.h"
#include "comms/ncclx/meta/tests/VerifyAlgoStatsUtil.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace {

enum class ReduceScatterNumericalAlgo {
  Ring,
  Pat,
  CtranDirect,
};

struct ReduceScatterNumericalParam {
  ReduceScatterNumericalAlgo algo;
  size_t count;
  ncclDataType_t datatype;
  ncclx::test::numerics::InputDistribution distribution =
      ncclx::test::numerics::InputDistribution::Uniform;

  std::string name() const {
    std::string algoName;
    switch (algo) {
      case ReduceScatterNumericalAlgo::Ring:
        algoName = "Ring";
        break;
      case ReduceScatterNumericalAlgo::Pat:
        algoName = "Pat";
        break;
      case ReduceScatterNumericalAlgo::CtranDirect:
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

class ReduceScatterNumericalTest
    : public NcclxBaseTestFixture,
      public ::testing::WithParamInterface<ReduceScatterNumericalParam> {
 public:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp({
        {"NCCL_CTRAN_ENABLE", "1"},
        {"NCCL_CTRAN_IPC_REGCACHE_ENABLE_ASYNC_SOCKET", "1"},
        {"NCCL_PAT_ENABLE", "1"},
    });
    algoStats_.enable();
    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream_));
    NcclxBaseTestFixture::TearDown();
  }

 protected:
  template <typename T>
  void run(const ReduceScatterNumericalParam& param) {
    T* originalDevice = nullptr;
    T* actualDevice = nullptr;
    const size_t sendCount = param.count * static_cast<size_t>(numRanks);
    NCCLCHECK_TEST(
        ncclMemAlloc((void**)&originalDevice, sendCount * sizeof(T)));
    NCCLCHECK_TEST(
        ncclMemAlloc((void**)&actualDevice, param.count * sizeof(T)));

    std::vector<T> original;
    for (int outputRank = 0; outputRank < numRanks; ++outputRank) {
      ncclx::test::numerics::appendRandomInputs<T>(
          original, param.count, globalRank, outputRank, param.distribution);
    }
    CUDACHECK_TEST(cudaMemcpyAsync(
        originalDevice,
        original.data(),
        sendCount * sizeof(T),
        cudaMemcpyDefault,
        stream_));
    std::vector<double> reference(param.count, 0.0);
    std::vector<T> bf16Hop(param.count);
    {
      ncclx::test::NcclCommRAII referenceComm{
          globalRank, numRanks, localRank, bootstrap_.get()};
      const auto gathered = ncclx::test::numerics::gatherInputs(
          originalDevice,
          sendCount,
          param.datatype,
          referenceComm,
          stream_,
          numRanks);
      std::vector<double> contributions(numRanks);
      const size_t chunkOffset = static_cast<size_t>(globalRank) * param.count;
      for (size_t i = 0; i < param.count; ++i) {
        for (int rank = 0; rank < numRanks; ++rank) {
          const size_t rankOffset = static_cast<size_t>(rank) * sendCount;
          contributions[rank] = static_cast<double>(DataTypeTraits<T>::toHost(
              gathered[rankOffset + chunkOffset + i]));
          reference[i] += contributions[rank];
        }
        bf16Hop[i] = ncclx::test::numerics::bf16HopReduce<T>(contributions);
      }
    }

    std::optional<EnvRAII<enum NCCL_REDUCESCATTER_ALGO>> reduceScatterAlgoGuard;
    std::optional<SysEnvRAII> ncclAlgoGuard;
    std::optional<SysEnvRAII> ncclProtoGuard;
    if (param.algo == ReduceScatterNumericalAlgo::Ring) {
      reduceScatterAlgoGuard.emplace(
          NCCL_REDUCESCATTER_ALGO, NCCL_REDUCESCATTER_ALGO::orig);
      ncclAlgoGuard.emplace("NCCL_ALGO", "RING");
    } else if (param.algo == ReduceScatterNumericalAlgo::Pat) {
      reduceScatterAlgoGuard.emplace(
          NCCL_REDUCESCATTER_ALGO, NCCL_REDUCESCATTER_ALGO::orig);
      ncclAlgoGuard.emplace("NCCL_ALGO", "PAT");
      ncclProtoGuard.emplace("NCCL_PROTO", "Simple");
    } else {
      reduceScatterAlgoGuard.emplace(
          NCCL_REDUCESCATTER_ALGO, NCCL_REDUCESCATTER_ALGO::ctdirect);
    }

    ncclx::test::NcclCommRAII comm{
        globalRank, numRanks, localRank, bootstrap_.get()};
    if (param.algo == ReduceScatterNumericalAlgo::CtranDirect &&
        !ctranReduceScatterSupport(
            comm->ctranComm_.get(), NCCL_REDUCESCATTER_ALGO::ctdirect)) {
      NCCLCHECK_TEST(ncclMemFree(originalDevice));
      NCCLCHECK_TEST(ncclMemFree(actualDevice));
      GTEST_SKIP() << "Ctran ReduceScatter direct is not supported for this "
                      "topology";
    }

    const auto result = ncclReduceScatter(
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
        "ReduceScatter",
        param.name());
    ncclx::test::numerics::printReferenceBytes(
        reference, globalRank, "ReduceScatter", param.name());
    if (param.datatype == ncclBfloat16) {
      ncclx::test::numerics::printBf16HopBytes(
          bf16Hop, globalRank, "ReduceScatter", param.name());
    }

    const size_t mismatches = ncclx::test::numerics::countMismatches(
        actualDevice, reference, stream_, globalRank, param.name());
    EXPECT_EQ(mismatches, 0) << param.name() << " rank=" << globalRank;

    if (param.algo == ReduceScatterNumericalAlgo::Ring) {
      algoStats_.verify(comm, "ReduceScatter", "RING");
    } else if (param.algo == ReduceScatterNumericalAlgo::Pat) {
      algoStats_.verify(comm, "ReduceScatter", "PAT");
    }

    NCCLCHECK_TEST(ncclMemFree(originalDevice));
    NCCLCHECK_TEST(ncclMemFree(actualDevice));
  }

  cudaStream_t stream_{nullptr};
  ncclx::test::VerifyAlgoStatsHelper algoStats_;
};

TEST_P(ReduceScatterNumericalTest, MatchesFp64Reference) {
  const auto& param = GetParam();
  if (param.datatype == ncclFloat32) {
    run<float>(param);
  } else if (param.datatype == ncclBfloat16) {
    run<__nv_bfloat16>(param);
  } else {
    FAIL() << "Unhandled datatype in " << param.name();
  }
}

const auto kReduceScatterNumericalParams = ::testing::Values(
    ReduceScatterNumericalParam{
        .algo = ReduceScatterNumericalAlgo::Ring,
        .count = 1,
        .datatype = ncclFloat32},
    ReduceScatterNumericalParam{
        .algo = ReduceScatterNumericalAlgo::Ring,
        .count = 8193,
        .datatype = ncclFloat32},
    ReduceScatterNumericalParam{
        .algo = ReduceScatterNumericalAlgo::Ring,
        .count = 1024,
        .datatype = ncclBfloat16},
    ReduceScatterNumericalParam{
        .algo = ReduceScatterNumericalAlgo::Pat,
        .count = 1,
        .datatype = ncclFloat32},
    ReduceScatterNumericalParam{
        .algo = ReduceScatterNumericalAlgo::Pat,
        .count = 8193,
        .datatype = ncclFloat32},
    ReduceScatterNumericalParam{
        .algo = ReduceScatterNumericalAlgo::Pat,
        .count = 1024,
        .datatype = ncclBfloat16},
    ReduceScatterNumericalParam{
        .algo = ReduceScatterNumericalAlgo::CtranDirect,
        .count = 1024,
        .datatype = ncclFloat32},
    ReduceScatterNumericalParam{
        .algo = ReduceScatterNumericalAlgo::CtranDirect,
        .count = 4099,
        .datatype = ncclBfloat16},
    ReduceScatterNumericalParam{
        .algo = ReduceScatterNumericalAlgo::Ring,
        .count = 8193,
        .datatype = ncclFloat32,
        .distribution = ncclx::test::numerics::InputDistribution::Normal},
    ReduceScatterNumericalParam{
        .algo = ReduceScatterNumericalAlgo::Ring,
        .count = 1024,
        .datatype = ncclBfloat16,
        .distribution = ncclx::test::numerics::InputDistribution::Normal}
#ifdef REDUCTION_NUMERICAL_LARGE_COUNT_TEST
    ,
    ReduceScatterNumericalParam{
        .algo = ReduceScatterNumericalAlgo::Ring,
        .count = 65536,
        .datatype = ncclFloat32},
    ReduceScatterNumericalParam{
        .algo = ReduceScatterNumericalAlgo::Pat,
        .count = 65536,
        .datatype = ncclFloat32},
    ReduceScatterNumericalParam{
        .algo = ReduceScatterNumericalAlgo::CtranDirect,
        .count = 32768,
        .datatype = ncclBfloat16}
#endif
);

INSTANTIATE_TEST_SUITE_P(
    ReduceScatter,
    ReduceScatterNumericalTest,
    kReduceScatterNumericalParams,
    [](const ::testing::TestParamInfo<ReduceScatterNumericalParam>& info) {
      return info.param.name();
    });

} // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
