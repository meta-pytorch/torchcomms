// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/synchronization/Baton.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"

namespace ctran::testing {

using AllReduceTestParam = std::tuple<std::string, enum NCCL_ALLREDUCE_ALGO>;
using AllReduceMinMsgSizeTestParam = std::tuple<size_t, commDataType_t>;

enum class CtranAllReduceRingMinSizeTestOpt {
  expect_sufficient,
  expect_insufficient,
};

class CtranAllReduceTest
    : public CtranIntraProcessFixture,
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

    CtranIntraProcessFixture::SetUp();
  }
  void startWorkers(bool abortEnabled) {
    std::vector<std::shared_ptr<::ctran::utils::Abort>> aborts;
    if (abortEnabled) {
      for (int i = 0; i < kNRanks; ++i) {
        aborts.push_back(ctran::utils::createAbort(/*enabled=*/true));
      }
    }
    CtranIntraProcessFixture::startWorkers(kNRanks, /*aborts=*/aborts);
  }

  void validateConfigs(size_t nElem) {
    ASSERT_TRUE(nElem <= kBufferNElem);
  }

  void runAllReduce(
      size_t nElem,
      PerRankState& state,
      enum NCCL_ALLREDUCE_ALGO algo,
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
            algo,
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

  startWorkers(/*abortEnabled=*/false);
  for (int rank = 0; rank < kNRanks; ++rank) {
    run(rank, [this, algo](PerRankState& state) {
      runAllReduce(kBufferNElem, state, algo);
    });
  }
}

TEST_P(CtranAllReduceTest, BasicRunAbortEnabled) {
  auto [algoName, algo] = GetParam();

  startWorkers(/*abortEnabled=*/true);
  for (int rank = 0; rank < kNRanks; ++rank) {
    run(rank, [this, algo](PerRankState& state) {
      runAllReduce(kBufferNElem, state, algo);
    });
  }
}

TEST_P(CtranAllReduceTest, SmallMessageSize) {
  auto [algoName, algo] = GetParam();

  startWorkers(/*abortEnabled=*/true);
  for (int rank = 0; rank < kNRanks; ++rank) {
    run(rank, [this, algo](PerRankState& state) {
      runAllReduce(/*nElem=*/1, state, algo);
    });
  }
}

void CtranAllReduceTest::runTestRanksAbsent(
    std::vector<int> ranksToRunCollective,
    std::vector<int> ranksAbsent,
    std::optional<std::chrono::milliseconds> timeout) {
  auto [algoName, algo] = GetParam();

  startWorkers(/*abortEnabled=*/true);

  for (auto rank : ranksToRunCollective) {
    run(rank, [this, timeout, algo](PerRankState& state) {
      // warmup
      runAllReduce(kBufferNElem, state, algo);

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
          algo,
          /*expectError=*/true,
          workEnqueued,
          timeout);

      state.getBootstrap()->barrierNamed(
          state.rank, state.nRanks, /*timeoutSeconds=*/4, "exit");
    });
  }

  for (auto rank : ranksAbsent) {
    run(rank, [this, algo](PerRankState& state) {
      // warmup
      runAllReduce(kBufferNElem, state, algo);

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
    ::testing::Values(
        std::make_tuple("ctring", NCCL_ALLREDUCE_ALGO::ctring),
        std::make_tuple("ctdirect", NCCL_ALLREDUCE_ALGO::ctdirect)),
    [](const ::testing::TestParamInfo<AllReduceTestParam>& info) {
      return std::get<0>(info.param);
    });

// Test fixture for ctring minimum message size validation
class CtranAllReduceRingMinSizeTest
    : public CtranIntraProcessFixture,
      public ::testing::WithParamInterface<AllReduceMinMsgSizeTestParam> {
 protected:
  static constexpr int kDefaultNumRanks = 4;
  static_assert(kDefaultNumRanks % 2 == 0);
  static constexpr commRedOp_t kReduceOpType = commSum;

  void SetUp() override {
    setenv("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal", 1);
    setenv("NCCL_IGNORE_TOPO_LOAD_FAILURE", "1", 1);
    CtranIntraProcessFixture::SetUp();
  }

  void startWorkers(int numRanks = kDefaultNumRanks) {
    std::vector<std::shared_ptr<::ctran::utils::Abort>> aborts;
    aborts.reserve(numRanks);
    for (int i = 0; i < numRanks; ++i) {
      aborts.push_back(ctran::utils::createAbort(/*enabled=*/true));
    }
    CtranIntraProcessFixture::startWorkers(numRanks, /*aborts=*/aborts);
  }

  void runTest(
      size_t count,
      commDataType_t dt,
      enum CtranAllReduceRingMinSizeTestOpt testOpt,
      int numRanks = kDefaultNumRanks) {
    startWorkers(numRanks);
    for (int rank = 0; rank < numRanks; ++rank) {
      run(rank, [this, count, dt, testOpt](PerRankState& state) {
        ASSERT_TRUE(ctranAllReduceSupport(
            state.ctranComm.get(), NCCL_ALLREDUCE_ALGO::ctring));

        size_t bufferSize = count * commTypeSize(dt);
        if (bufferSize < CTRAN_MIN_REGISTRATION_SIZE) {
          bufferSize = CTRAN_MIN_REGISTRATION_SIZE;
        }

        void* srcHandle;
        void* dstHandle;
        ASSERT_EQ(
            commSuccess,
            state.ctranComm->ctran_->commRegister(
                state.srcBuffer, bufferSize, &srcHandle));
        ASSERT_EQ(
            commSuccess,
            state.ctranComm->ctran_->commRegister(
                state.dstBuffer, bufferSize, &dstHandle));

        if (testOpt == CtranAllReduceRingMinSizeTestOpt::expect_sufficient) {
          // Should not throw when count >= nRanks
          EXPECT_NO_THROW({
            auto res = ctranAllReduceRing(
                state.srcBuffer,
                state.dstBuffer,
                count,
                dt,
                kReduceOpType,
                state.ctranComm.get(),
                state.stream);
            EXPECT_EQ(res, commSuccess);
          });
        } else {
          // Expect ctran::utils::Exception when count < nRanks
          EXPECT_THROW(
              {
                ctranAllReduceRing(
                    state.srcBuffer,
                    state.dstBuffer,
                    count,
                    dt,
                    kReduceOpType,
                    state.ctranComm.get(),
                    state.stream);
              },
              ctran::utils::Exception);
        }

        // ensure async execution completion and no error
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(state.stream));

        // deregistering will happen after streamSync below
        ASSERT_EQ(
            commSuccess, state.ctranComm->ctran_->commDeregister(dstHandle));
        ASSERT_EQ(
            commSuccess, state.ctranComm->ctran_->commDeregister(srcHandle));
      });
    }
  }
};

TEST_P(CtranAllReduceRingMinSizeTest, InsufficientElements_NRanksMinus1) {
  auto [numRanks, dt] = GetParam();
  ASSERT_GT(numRanks, 1) << "Need at least 2 ranks for this test";
  runTest(
      numRanks - 1,
      dt,
      CtranAllReduceRingMinSizeTestOpt::expect_insufficient,
      numRanks);
}

TEST_P(CtranAllReduceRingMinSizeTest, SufficientElements_ExactlyNRanks) {
  auto [numRanks, dt] = GetParam();
  runTest(
      numRanks,
      dt,
      CtranAllReduceRingMinSizeTestOpt::expect_sufficient,
      numRanks);
}

TEST_P(CtranAllReduceRingMinSizeTest, SufficientElements_NRanksPlus1) {
  auto [numRanks, dt] = GetParam();
  runTest(
      numRanks + 1,
      dt,
      CtranAllReduceRingMinSizeTestOpt::expect_sufficient,
      numRanks);
}

INSTANTIATE_TEST_SUITE_P(
    AllDataTypes,
    CtranAllReduceRingMinSizeTest,
    ::testing::Values(
        std::make_tuple<>(2, commFloat),
        std::make_tuple<>(2, commInt8),
        std::make_tuple<>(8, commFloat),
        std::make_tuple<>(8, commInt8)));

class CtranAllReduceRingOneRankTest : public CtranIntraProcessFixture {
 protected:
  static constexpr int kNRanks = 1;
  static constexpr commRedOp_t kReduceOpType = commSum;
  static constexpr commDataType_t kDataType = commInt32;
  static constexpr size_t kTypeSize = sizeof(int);
  static constexpr size_t kBufferNElem = kBufferSize / kTypeSize;

  void SetUp() override {
    CtranIntraProcessFixture::SetUp();
  }

  void runAllReduce(size_t nElem) {
    CtranIntraProcessFixture::startWorkers(
        kNRanks, /*aborts=*/{ctran::utils::createAbort(/*enabled=*/true)});

    run(/*rank=*/0, [this, nElem](PerRankState& state) {
      // set up src buffer to hold magic values, and zero out dst buffers
      int magic = 0xdeadbeef;
      int srcHost[kBufferNElem];
      int dstHost[kBufferNElem];
      for (int i = 0; i < kBufferNElem; ++i) {
        srcHost[i] = magic + i;
      }
      memset(dstHost, 0, kBufferSize);
      ASSERT_EQ(
          cudaSuccess,
          cudaMemcpy(
              state.srcBuffer, srcHost, kBufferSize, cudaMemcpyHostToDevice));
      ASSERT_EQ(cudaSuccess, cudaMemset(state.dstBuffer, 0, kBufferSize));

      // warmup
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
              NCCL_ALLREDUCE_ALGO::ctring,
              /*timeout=*/std::nullopt));

      CLOGF(INFO, "rank {} allReduce scheduled", state.rank);

      // ensure async execution completion and no error
      EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(state.stream));
      EXPECT_EQ(commSuccess, state.ctranComm->getAsyncResult());

      CLOGF(INFO, "rank {} allReduce task completed", state.rank);

      // validate results
      ASSERT_EQ(
          cudaSuccess,
          cudaMemcpy(
              dstHost, state.dstBuffer, kBufferSize, cudaMemcpyDeviceToHost));
      for (int i = 0; i < nElem; ++i) {
        EXPECT_EQ(srcHost[i], dstHost[i]);
      }
      for (int i = nElem; i < kBufferNElem; ++i) {
        EXPECT_EQ(dstHost[i], 0);
      }
    });
  }
};

TEST_F(CtranAllReduceRingOneRankTest, Basic) {
  this->runAllReduce(/*nElem=*/kBufferNElem);
}

TEST_F(CtranAllReduceRingOneRankTest, SmallMessageSize) {
  this->runAllReduce(/*nElem=*/1);
}

// Test fixture for AllReduceRing perftrace CVAR controls.
// Runs real multi-rank AllReduce ring operations and verifies:
// 1. AllReduce results are always correct
// 2. NCCL_CTRAN_ENABLE_PERFTRACE controls trace file production
// 3. NCCL_CTRAN_ALLREDUCE_RING_PERFTRACE_SKIP_OPS skips initial ops
// 4. NCCL_CTRAN_ALLREDUCE_RING_PERFTRACE_NUM_OPS limits traced ops
class AllReduceRingPerfTraceTest : public CtranIntraProcessFixture {
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

    // Create temp directory for trace output
    traceDir_ = std::filesystem::temp_directory_path() /
        ("ctran_perftrace_test_" + std::to_string(getpid()));
    std::filesystem::create_directories(traceDir_);
    setenv("NCCL_CTRAN_PERFTRACE_DIR", traceDir_.c_str(), 1);

    CtranIntraProcessFixture::SetUp();
  }

  void TearDown() override {
    CtranIntraProcessFixture::TearDown();

    // Clean up trace dir
    std::filesystem::remove_all(traceDir_);

    // Reset perftrace env vars
    unsetenv("NCCL_CTRAN_ENABLE_PERFTRACE");
    unsetenv("NCCL_CTRAN_ALLREDUCE_RING_PERFTRACE_SKIP_OPS");
    unsetenv("NCCL_CTRAN_ALLREDUCE_RING_PERFTRACE_NUM_OPS");
    unsetenv("NCCL_CTRAN_PERFTRACE_DIR");
    ncclCvarInit();
  }

  void startWorkers() {
    std::vector<std::shared_ptr<::ctran::utils::Abort>> aborts;
    for (int i = 0; i < kNRanks; ++i) {
      aborts.push_back(ctran::utils::createAbort(/*enabled=*/true));
    }
    CtranIntraProcessFixture::startWorkers(kNRanks, /*aborts=*/aborts);
  }

  void
  configurePerfTrace(bool enable, int64_t skipOps = 0, int64_t numOps = 0) {
    setenv("NCCL_CTRAN_ENABLE_PERFTRACE", enable ? "1" : "0", 1);
    setenv(
        "NCCL_CTRAN_ALLREDUCE_RING_PERFTRACE_SKIP_OPS",
        std::to_string(skipOps).c_str(),
        1);
    setenv(
        "NCCL_CTRAN_ALLREDUCE_RING_PERFTRACE_NUM_OPS",
        std::to_string(numOps).c_str(),
        1);
    ncclCvarInit();
  }

  // Run a single AllReduce operation on all ranks and verify correctness.
  // Each rank r fills srcBuffer with float(r+1). After commSum AllReduce,
  // every element should equal kNRanks*(kNRanks+1)/2.
  void runAllReduceAndVerify(PerRankState& state) {
    // Fill src buffer: each rank writes (rank + 1) to all elements
    float fillVal = static_cast<float>(state.rank + 1);
    std::vector<float> srcHost(kBufferNElem, fillVal);
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            state.srcBuffer,
            srcHost.data(),
            kBufferSize,
            cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(state.dstBuffer, 0, kBufferSize));

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
      state.ctranComm->ctran_->commDeregister(dstHandle);
      state.ctranComm->ctran_->commDeregister(srcHandle);
    };

    EXPECT_EQ(
        commSuccess,
        ctranAllReduce(
            state.srcBuffer,
            state.dstBuffer,
            kBufferNElem,
            kDataType,
            kReduceOpType,
            state.ctranComm.get(),
            state.stream,
            NCCL_ALLREDUCE_ALGO::ctring,
            /*timeout=*/std::nullopt));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(state.stream));
    EXPECT_EQ(commSuccess, state.ctranComm->getAsyncResult());

    // Verify correctness: expected sum = 1 + 2 + ... + kNRanks
    float expectedVal = static_cast<float>(kNRanks * (kNRanks + 1) / 2);
    std::vector<float> dstHost(kBufferNElem);
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            dstHost.data(),
            state.dstBuffer,
            kBufferSize,
            cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < kBufferNElem; ++i) {
      EXPECT_EQ(expectedVal, dstHost[i])
          << "Mismatch at index " << i << " on rank " << state.rank;
    }
  }

  size_t countTraceFiles() {
    size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(traceDir_)) {
      if (entry.path().extension() == ".json" &&
          entry.path().filename().string().find("ctran_trace_log") !=
              std::string::npos) {
        ++count;
      }
    }
    return count;
  }

  std::filesystem::path traceDir_;
};

TEST_F(AllReduceRingPerfTraceTest, Disabled) {
  configurePerfTrace(/*enable=*/false);
  startWorkers();
  for (int rank = 0; rank < kNRanks; ++rank) {
    run(rank, [this](PerRankState& state) {
      runAllReduceAndVerify(state);

      state.getBootstrap()->barrierNamed(
          state.rank, state.nRanks, /*timeoutSeconds=*/10, "after_allreduce");

      // 0 traced ops × kNRanks = 0 files
      EXPECT_EQ(0u, countTraceFiles())
          << "No trace files expected when perftrace is disabled"
          << " (rank " << state.rank << ")";
    });
  }
}

TEST_F(AllReduceRingPerfTraceTest, Enabled) {
  configurePerfTrace(/*enable=*/true);
  startWorkers();
  for (int rank = 0; rank < kNRanks; ++rank) {
    run(rank, [this](PerRankState& state) {
      runAllReduceAndVerify(state);

      state.getBootstrap()->barrierNamed(
          state.rank, state.nRanks, /*timeoutSeconds=*/10, "after_allreduce");

      // 1 traced op × kNRanks ranks = kNRanks files
      EXPECT_EQ(static_cast<size_t>(kNRanks), countTraceFiles())
          << "rank " << state.rank;
    });
  }
}

TEST_F(AllReduceRingPerfTraceTest, SkipOps) {
  // Run 3 ops total, skip the first 2: only opCount=2 is traced
  constexpr int kTotalOps = 3;
  constexpr int64_t kSkipOps = 2;
  constexpr int kTracedOps = kTotalOps - kSkipOps; // 1
  configurePerfTrace(/*enable=*/true, /*skipOps=*/kSkipOps);
  startWorkers();
  for (int rank = 0; rank < kNRanks; ++rank) {
    run(rank, [this](PerRankState& state) {
      for (int op = 0; op < kTotalOps; ++op) {
        runAllReduceAndVerify(state);
        state.getBootstrap()->barrierNamed(
            state.rank,
            state.nRanks,
            /*timeoutSeconds=*/10,
            "after_op_" + std::to_string(op));
      }

      // 1 traced op × kNRanks ranks = kNRanks files
      EXPECT_EQ(static_cast<size_t>(kTracedOps * kNRanks), countTraceFiles())
          << "rank " << state.rank;
    });
  }
}

TEST_F(AllReduceRingPerfTraceTest, NumOps) {
  // Run 3 ops total, trace only the first 1 (numOps=1, opCount=0 traced)
  constexpr int kTotalOps = 3;
  constexpr int64_t kNumOps = 1;
  configurePerfTrace(/*enable=*/true, /*skipOps=*/0, /*numOps=*/kNumOps);
  startWorkers();
  for (int rank = 0; rank < kNRanks; ++rank) {
    run(rank, [this](PerRankState& state) {
      for (int op = 0; op < kTotalOps; ++op) {
        runAllReduceAndVerify(state);
        state.getBootstrap()->barrierNamed(
            state.rank,
            state.nRanks,
            /*timeoutSeconds=*/10,
            "after_op_" + std::to_string(op));
      }

      // 1 traced op × kNRanks ranks = kNRanks files
      EXPECT_EQ(static_cast<size_t>(kNumOps * kNRanks), countTraceFiles())
          << "rank " << state.rank;
    });
  }
}

TEST_F(AllReduceRingPerfTraceTest, SkipAndNumOps) {
  // Run 5 ops total, skip 1, trace 2 (opCounts 1 and 2 are traced)
  constexpr int kTotalOps = 5;
  constexpr int64_t kSkipOps = 1;
  constexpr int64_t kNumOps = 2;
  configurePerfTrace(
      /*enable=*/true, /*skipOps=*/kSkipOps, /*numOps=*/kNumOps);
  startWorkers();
  for (int rank = 0; rank < kNRanks; ++rank) {
    run(rank, [this](PerRankState& state) {
      for (int op = 0; op < kTotalOps; ++op) {
        runAllReduceAndVerify(state);
        state.getBootstrap()->barrierNamed(
            state.rank,
            state.nRanks,
            /*timeoutSeconds=*/10,
            "after_op_" + std::to_string(op));
      }

      // 2 traced ops × kNRanks ranks = 2*kNRanks files
      EXPECT_EQ(static_cast<size_t>(kNumOps * kNRanks), countTraceFiles())
          << "rank " << state.rank;
    });
  }
}

} // namespace ctran::testing
