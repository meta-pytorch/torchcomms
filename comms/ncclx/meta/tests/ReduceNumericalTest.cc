// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Numerical canaries for NCCL Reduce. These tests are intended to catch NCCL
// upgrade regressions by comparing forced-algorithm root output against an
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

#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/ncclx/meta/tests/ReductionNumericalTestUtils.h"
#include "comms/ncclx/meta/tests/VerifyAlgoStatsUtil.h"
#include "comms/testinfra/TestUtils.h"

namespace {

enum class ReduceNumericalAlgo {
  Default,
  Ring,
  TreeUnsupported,
  PatUnsupported,
  CtranUnsupported,
};

struct ReduceNumericalParam {
  ReduceNumericalAlgo algo;
  size_t count;
  ncclDataType_t datatype;

  std::string name() const {
    std::string algoName;
    switch (algo) {
      case ReduceNumericalAlgo::Default:
        algoName = "Default";
        break;
      case ReduceNumericalAlgo::Ring:
        algoName = "Ring";
        break;
      case ReduceNumericalAlgo::TreeUnsupported:
        algoName = "TreeUnsupported";
        break;
      case ReduceNumericalAlgo::PatUnsupported:
        algoName = "PatUnsupported";
        break;
      case ReduceNumericalAlgo::CtranUnsupported:
        algoName = "CtranUnsupported";
        break;
    }

    const std::string dtypeName =
        datatype == ncclFloat32 ? "Float32" : "Bfloat16";
    return algoName + "_" + dtypeName + "_Count_" + std::to_string(count);
  }
};

class ReduceNumericalTest
    : public NcclxBaseTestFixture,
      public ::testing::WithParamInterface<ReduceNumericalParam> {
 public:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp();
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
  void run(const ReduceNumericalParam& param) {
    if (param.algo == ReduceNumericalAlgo::PatUnsupported) {
      GTEST_SKIP() << "PAT is not a selectable NCCL Reduce algorithm";
    }
    if (param.algo == ReduceNumericalAlgo::TreeUnsupported) {
      GTEST_SKIP() << "TREE is not a selectable NCCL Reduce algorithm";
    }
    if (param.algo == ReduceNumericalAlgo::CtranUnsupported) {
      GTEST_SKIP() << "NCCLX does not provide a CTRAN Reduce path";
    }

    constexpr int kRoot = 0;
    T* originalDevice = nullptr;
    T* actualDevice = nullptr;
    NCCLCHECK_TEST(
        ncclMemAlloc((void**)&originalDevice, param.count * sizeof(T)));
    NCCLCHECK_TEST(
        ncclMemAlloc((void**)&actualDevice, param.count * sizeof(T)));

    std::vector<T> original;
    ncclx::test::numerics::appendRandomInputs<T>(
        original, param.count, globalRank, kRoot);
    CUDACHECK_TEST(cudaMemcpyAsync(
        originalDevice,
        original.data(),
        param.count * sizeof(T),
        cudaMemcpyDefault,
        stream_));
    const auto reference = [&]() {
      ncclx::test::NcclCommRAII referenceComm{
          globalRank, numRanks, localRank, bootstrap_.get()};
      const auto gathered = ncclx::test::numerics::gatherInputs(
          originalDevice,
          param.count,
          param.datatype,
          referenceComm,
          stream_,
          numRanks);
      std::vector<double> result(param.count, 0.0);
      for (size_t i = 0; i < param.count; ++i) {
        for (int rank = 0; rank < numRanks; ++rank) {
          result[i] += static_cast<double>(DataTypeTraits<T>::toHost(
              gathered[static_cast<size_t>(rank) * param.count + i]));
        }
      }
      return result;
    }();

    std::optional<SysEnvRAII> ncclAlgoGuard;
    if (param.algo == ReduceNumericalAlgo::Ring) {
      ncclAlgoGuard.emplace("NCCL_ALGO", "RING");
    }

    ncclx::test::NcclCommRAII comm{
        globalRank, numRanks, localRank, bootstrap_.get()};

    const auto result = ncclReduce(
        originalDevice,
        actualDevice,
        param.count,
        param.datatype,
        ncclSum,
        kRoot,
        comm,
        stream_);
    ASSERT_EQ(result, ncclSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));

    if (globalRank == kRoot) {
      ncclx::test::numerics::printActualOutputBytes(
          actualDevice,
          param.count,
          stream_,
          globalRank,
          "Reduce",
          param.name());
      const size_t mismatches = ncclx::test::numerics::countMismatches(
          actualDevice, reference, stream_, globalRank, param.name());
      EXPECT_EQ(mismatches, 0) << param.name() << " rank=" << globalRank;
    }

    if (param.algo == ReduceNumericalAlgo::Ring) {
      algoStats_.verify(comm, "Reduce", "RING");
    }

    NCCLCHECK_TEST(ncclMemFree(originalDevice));
    NCCLCHECK_TEST(ncclMemFree(actualDevice));
  }

  cudaStream_t stream_{nullptr};
  ncclx::test::VerifyAlgoStatsHelper algoStats_;
};

TEST_P(ReduceNumericalTest, MatchesFp64Reference) {
  const auto& param = GetParam();
  if (param.datatype == ncclFloat32) {
    run<float>(param);
  } else if (param.datatype == ncclBfloat16) {
    run<__nv_bfloat16>(param);
  } else {
    FAIL() << "Unhandled datatype in " << param.name();
  }
}

const auto kReduceNumericalParams = ::testing::Values(
    ReduceNumericalParam{
        .algo = ReduceNumericalAlgo::Default,
        .count = 8193,
        .datatype = ncclFloat32},
    ReduceNumericalParam{
        .algo = ReduceNumericalAlgo::Default,
        .count = 1024,
        .datatype = ncclBfloat16},
    ReduceNumericalParam{
        .algo = ReduceNumericalAlgo::Ring,
        .count = 1,
        .datatype = ncclFloat32},
    ReduceNumericalParam{
        .algo = ReduceNumericalAlgo::Ring,
        .count = 8193,
        .datatype = ncclFloat32},
    ReduceNumericalParam{
        .algo = ReduceNumericalAlgo::Ring,
        .count = 1024,
        .datatype = ncclBfloat16},
    ReduceNumericalParam{
        .algo = ReduceNumericalAlgo::TreeUnsupported,
        .count = 1024,
        .datatype = ncclFloat32},
    ReduceNumericalParam{
        .algo = ReduceNumericalAlgo::PatUnsupported,
        .count = 1024,
        .datatype = ncclFloat32},
    ReduceNumericalParam{
        .algo = ReduceNumericalAlgo::CtranUnsupported,
        .count = 1024,
        .datatype = ncclFloat32}
#ifdef REDUCTION_NUMERICAL_LARGE_COUNT_TEST
    ,
    ReduceNumericalParam{
        .algo = ReduceNumericalAlgo::Ring,
        .count = 262144,
        .datatype = ncclFloat32},
    ReduceNumericalParam{
        .algo = ReduceNumericalAlgo::Ring,
        .count = 65536,
        .datatype = ncclBfloat16}
#endif
);

INSTANTIATE_TEST_SUITE_P(
    Reduce,
    ReduceNumericalTest,
    kReduceNumericalParams,
    [](const ::testing::TestParamInfo<ReduceNumericalParam>& info) {
      return info.param.name();
    });

} // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
