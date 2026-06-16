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
    return algoName + "_" + dtypeName + "_" +
        ncclx::test::numerics::countName(count);
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

    std::optional<SysEnvRAII> ncclAlgoGuard;
    if (param.algo == ReduceNumericalAlgo::Ring) {
      ncclAlgoGuard.emplace("NCCL_ALGO", "RING");
    }

    ncclx::test::NcclCommRAII comm{
        globalRank, numRanks, localRank, bootstrap_.get()};

    constexpr int kRoot = 0;
    T* sendBuf = nullptr;
    T* recvBuf = nullptr;
    NCCLCHECK_TEST(ncclMemAlloc((void**)&sendBuf, param.count * sizeof(T)));
    NCCLCHECK_TEST(ncclMemAlloc((void**)&recvBuf, param.count * sizeof(T)));

    const auto hostInput = ncclx::test::numerics::makeReduceInput<T>(
        globalRank, param.count, kRoot);
    CUDACHECK_TEST(cudaMemcpyAsync(
        sendBuf,
        hostInput.data(),
        param.count * sizeof(T),
        cudaMemcpyDefault,
        stream_));

    void* sendHandle = nullptr;
    void* recvHandle = nullptr;
    NCCLCHECK_TEST(
        ncclCommRegister(comm, sendBuf, param.count * sizeof(T), &sendHandle));
    NCCLCHECK_TEST(
        ncclCommRegister(comm, recvBuf, param.count * sizeof(T), &recvHandle));

    const auto result = ncclReduce(
        sendBuf,
        recvBuf,
        param.count,
        param.datatype,
        ncclSum,
        kRoot,
        comm,
        stream_);
    ASSERT_EQ(result, ncclSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));

    if (globalRank == kRoot) {
      const auto expected = ncclx::test::numerics::reduceExpected<T>(
          param.count, numRanks, kRoot);
      const size_t mismatches = ncclx::test::numerics::countMismatches(
          recvBuf, expected, stream_, globalRank, param.name());
      EXPECT_EQ(mismatches, 0) << param.name() << " rank=" << globalRank;
    }

    if (param.algo == ReduceNumericalAlgo::Ring) {
      algoStats_.verify(comm, "Reduce", "RING");
    }

    NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
    NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
    NCCLCHECK_TEST(ncclMemFree(sendBuf));
    NCCLCHECK_TEST(ncclMemFree(recvBuf));
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
        .datatype = ncclFloat32});

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
