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
    return algoName + "_" + dtypeName + "_" +
        ncclx::test::numerics::countName(count);
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
      GTEST_SKIP() << "Ctran ReduceScatter direct is not supported for this "
                      "topology";
    }

    T* sendBuf = nullptr;
    T* recvBuf = nullptr;
    const size_t sendCount = param.count * static_cast<size_t>(numRanks);
    NCCLCHECK_TEST(ncclMemAlloc((void**)&sendBuf, sendCount * sizeof(T)));
    NCCLCHECK_TEST(ncclMemAlloc((void**)&recvBuf, param.count * sizeof(T)));

    const auto hostInput = ncclx::test::numerics::makeReduceScatterInput<T>(
        globalRank, param.count, numRanks);
    CUDACHECK_TEST(cudaMemcpyAsync(
        sendBuf,
        hostInput.data(),
        sendCount * sizeof(T),
        cudaMemcpyDefault,
        stream_));

    void* sendHandle = nullptr;
    void* recvHandle = nullptr;
    NCCLCHECK_TEST(
        ncclCommRegister(comm, sendBuf, sendCount * sizeof(T), &sendHandle));
    NCCLCHECK_TEST(
        ncclCommRegister(comm, recvBuf, param.count * sizeof(T), &recvHandle));

    const auto result = ncclReduceScatter(
        sendBuf, recvBuf, param.count, param.datatype, ncclSum, comm, stream_);
    ASSERT_EQ(result, ncclSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));

    const auto expected = ncclx::test::numerics::reduceScatterExpected<T>(
        param.count, numRanks, globalRank);
    const size_t mismatches = ncclx::test::numerics::countMismatches(
        recvBuf, expected, stream_, globalRank, param.name());
    EXPECT_EQ(mismatches, 0) << param.name() << " rank=" << globalRank;

    if (param.algo == ReduceScatterNumericalAlgo::Ring) {
      algoStats_.verify(comm, "ReduceScatter", "RING");
    } else if (param.algo == ReduceScatterNumericalAlgo::Pat) {
      algoStats_.verify(comm, "ReduceScatter", "PAT");
    }

    NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
    NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
    NCCLCHECK_TEST(ncclMemFree(sendBuf));
    NCCLCHECK_TEST(ncclMemFree(recvBuf));
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
        .datatype = ncclBfloat16}
#ifdef REDUCTION_NUMERICAL_LARGE_TEST
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
