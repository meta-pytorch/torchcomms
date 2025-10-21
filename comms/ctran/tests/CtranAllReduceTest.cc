// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <future>
#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/synchronization/Baton.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/tests/CtranStandaloneUTUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::testing {

using AllReduceTestParam = std::tuple<std::string, enum NCCL_ALLREDUCE_ALGO>;

class CtranAllReduceTest
    : public CtranStandaloneMultiRankBaseTest,
      public ::testing::WithParamInterface<AllReduceTestParam> {
 protected:
  static constexpr int kNRanks = 4;
  static_assert(kNRanks % 2 == 0);
  static constexpr commRedOp_t kReduceOpType = commSum;
  static constexpr commDataType_t kDataType = commFloat32;
  static constexpr size_t kTypeSize = sizeof(float);
  static constexpr size_t kBufferNElem = kBufferSize / kTypeSize;

  void SetUp() override {
    setenv("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal", 1);
    setenv("NCCL_IGNORE_TOPO_LOAD_FAILURE", "1", 1);

    CtranStandaloneMultiRankBaseTest::SetUp();
  }
  void startWorkers(bool abortEnabled) {
    std::vector<std::shared_ptr<::ctran::utils::Abort>> aborts;
    if (abortEnabled) {
      for (int i = 0; i < kNRanks; ++i) {
        aborts.push_back(ctran::utils::createAbort(/*enabled=*/true));
      }
    }
    CtranStandaloneMultiRankBaseTest::startWorkers(kNRanks, /*aborts=*/aborts);
  }

  void validateConfigs(size_t nElem) {
    ASSERT_TRUE(nElem <= kBufferNElem);
  }

  void runAllReduce(
      size_t nElem,
      PerRankState& state,
      bool expectError = false,
      std::shared_ptr<folly::Baton<>> workEnqueued = nullptr,
      std::optional<std::chrono::milliseconds> timeout = std::nullopt) {
    validateConfigs(nElem);

    CLOGF(INFO, "rank {} allReduce with {} elems", state.rank, nElem);

    void* srcHandle;
    void* dstHandle;
    ASSERT_EQ(
        commSuccess,
        state.ctranComm->ctran_->commRegister(
            state.srcBuffer, kBufferSize, &srcHandle));
    ASSERT_EQ(
        commSuccess,
        state.ctranComm->ctran_->commRegister(
            state.dstBuffer, kBufferSize, &dstHandle));
    SCOPE_EXIT {
      // deregistering will happen after streamSync below
      state.ctranComm->ctran_->commDeregister(dstHandle);
      state.ctranComm->ctran_->commDeregister(srcHandle);
    };

    CLOGF(INFO, "rank {} allReduce completed registration", state.rank);

    EXPECT_EQ(
        commSuccess,
        ctranAllReduce(
            state.srcBuffer,
            state.dstBuffer,
            nElem,
            kDataType,
            kReduceOpType,
            state.ctranComm.get(),
            state.stream,
            std::nullopt,
            timeout));
    if (workEnqueued) {
      workEnqueued->post();
    }

    CLOGF(
        INFO,
        "rank {} allReduce scheduled, expecting {}",
        state.rank,
        expectError ? "error" : "success");

    // ensure async execution completion and no error
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(state.stream));
    if (!expectError) {
      EXPECT_EQ(commSuccess, state.ctranComm->getAsyncResult());
    } else {
      // TODO(T238821628): update error code for now we strictly check for
      // remote error, since this is the error type returned by network error
      // and temporarily for abort as well.
      EXPECT_EQ(commRemoteError, state.ctranComm->getAsyncResult());
    }

    CLOGF(INFO, "rank {} allReduce task completed", state.rank);
  }

  void runTestRanksAbsent(
      std::vector<int> ranksToRunCollective,
      std::vector<int> ranksAbsent,
      std::optional<std::chrono::milliseconds> timeout = std::nullopt);
};

TEST_P(CtranAllReduceTest, BasicRunAbortDisabled) {
  auto [algoName, algo] = GetParam();
  NCCL_ALLREDUCE_ALGO = algo;

  startWorkers(/*abortEnabled=*/false);
  for (int rank = 0; rank < kNRanks; ++rank) {
    run(rank,
        [this](PerRankState& state) { runAllReduce(kBufferNElem, state); });
  }
}

TEST_P(CtranAllReduceTest, BasicRunAbortEnabled) {
  auto [algoName, algo] = GetParam();
  NCCL_ALLREDUCE_ALGO = algo;

  startWorkers(/*abortEnabled=*/true);
  for (int rank = 0; rank < kNRanks; ++rank) {
    run(rank,
        [this](PerRankState& state) { runAllReduce(kBufferNElem, state); });
  }
}

void CtranAllReduceTest::runTestRanksAbsent(
    std::vector<int> ranksToRunCollective,
    std::vector<int> ranksAbsent,
    std::optional<std::chrono::milliseconds> timeout) {
  auto [algoName, algo] = GetParam();
  NCCL_ALLREDUCE_ALGO = algo;

  startWorkers(/*abortEnabled=*/true);

  for (auto rank : ranksToRunCollective) {
    run(rank, [this, timeout](PerRankState& state) {
      // warmup
      runAllReduce(kBufferNElem, state);

      state.getBootstrap()->barrierNamed(
          state.rank,
          state.nRanks,
          /*timeoutSeconds=*/4,
          "after healthy run");

      auto workEnqueued = std::make_shared<folly::Baton<>>();
      auto timer = std::async(std::launch::async, [&]() {
        workEnqueued->wait();

        auto timerWait = timeout.value_or(std::chrono::milliseconds(1000)) -
            std::chrono::milliseconds(100);
        std::this_thread::sleep_for(timerWait);

        EXPECT_EQ(cudaErrorNotReady, cudaStreamQuery(state.stream))
            << "rank " << state.rank;

        // Barrier to ensure ranks are not aborted before checking status above.
        // Generally speaking, ranks aborting may cause peers to see network
        // errors before local abort signal. This is an inherently racey
        // condition, so we want to avoid any cascading failures from network.
        state.getBootstrap()->barrierNamed(
            state.rank,
            state.nRanks,
            /*timeoutSeconds=*/4,
            "after verify hang");

        if (!timeout.has_value()) {
          state.ctranComm->setAbort();
        }
      });
      runAllReduce(
          kBufferNElem,
          state,
          /*expectError=*/true,
          workEnqueued,
          timeout);

      state.getBootstrap()->barrierNamed(
          state.rank, state.nRanks, /*timeoutSeconds=*/4, "exit");
    });
  }

  for (auto rank : ranksAbsent) {
    run(rank, [this](PerRankState& state) {
      // warmup
      runAllReduce(kBufferNElem, state);

      state.getBootstrap()->barrierNamed(
          state.rank,
          state.nRanks,
          /*timeoutSeconds=*/4,
          "after healthy run");

      state.getBootstrap()->barrierNamed(
          state.rank, state.nRanks, /*timeoutSeconds=*/4, "after verify hang");

      state.getBootstrap()->barrierNamed(
          state.rank, state.nRanks, /*timeoutSeconds=*/4, "exit");
    });
  }
}

TEST_P(CtranAllReduceTest, Rank1And3AbsentActiveAbort) {
  this->runTestRanksAbsent(
      /*ranksToRunCollective=*/{0, 2}, /*ranksAbsent=*/{1, 3});
}

TEST_P(CtranAllReduceTest, Rank1And3AbsentTimeout) {
  this->runTestRanksAbsent(
      /*ranksToRunCollective=*/{0, 2},
      /*ranksAbsent=*/{1, 3},
      /*timeout=*/std::chrono::milliseconds(2000));
}

TEST_P(CtranAllReduceTest, Rank2AbsentActiveAbort) {
  this->runTestRanksAbsent(
      /*ranksToRunCollective=*/{0, 1, 3}, /*ranksAbsent=*/{2});
}

TEST_P(CtranAllReduceTest, Rank2AbsentTimeout) {
  this->runTestRanksAbsent(
      /*ranksToRunCollective=*/{0, 1, 3},
      /*ranksAbsent=*/{2},
      /*timeout=*/std::chrono::milliseconds(2000));
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    CtranAllReduceTest,
    ::testing::Values(std::make_tuple("ctring", NCCL_ALLREDUCE_ALGO::ctring)),
    [](const ::testing::TestParamInfo<AllReduceTestParam>& info) {
      return std::get<0>(info.param);
    });

} // namespace ctran::testing
