// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <folly/Synchronized.h>
#include <folly/synchronization/Baton.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/gpe/tests/CtranGpeUTKernels.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/utils/Abort.h"
#include "comms/utils/logger/LogUtils.h"

namespace ctran::fttesting {

#define ASSERT_CUDASUCCESS(cmd)                                     \
  do {                                                              \
    cudaError_t ret;                                                \
    ASSERT_EQ(cudaSuccess, ret = (cmd)) << cudaGetErrorString(ret); \
  } while (0)

struct FtTestSync {
  struct SyncData {
    std::optional<ctran::utils::Exception> exception{std::nullopt};
    std::optional<commResult_t> res{std::nullopt};
    bool timeout{false};
    bool blockUntilActiveAbort{false};
  };

  folly::Synchronized<SyncData> syncData_;
  folly::Baton<> baton_;

  void signal() {
    baton_.post();
  }

  void wait(std::chrono::milliseconds timeout) {
    bool signaled = baton_.try_wait_for(timeout);
    if (!signaled) {
      CLOGF(INFO, "wait timeout");
    }
  }

  std::optional<ctran::utils::Exception> getException() const {
    return syncData_.withRLock(
        [&](const auto& data) { return data.exception; });
  }

  std::optional<commResult_t> getResult() const {
    return syncData_.withRLock([&](const auto& data) { return data.res; });
  }

  bool getTimeout() const {
    return syncData_.withRLock([&](const auto& data) { return data.timeout; });
  }

  bool getBlockUntilActiveAbort() const {
    return syncData_.withRLock(
        [&](const auto& data) { return data.blockUntilActiveAbort; });
  }

  void setException(const std::optional<ctran::utils::Exception>& exc) {
    syncData_.withWLock([&](auto& data) { data.exception = exc; });
  }

  void setResult(const std::optional<commResult_t>& result) {
    syncData_.withWLock([&](auto& data) { data.res = result; });
  }

  void setTimeout() {
    syncData_.withWLock([&](auto& data) { data.timeout = true; });
  }

  void setBlockUntilActiveAbort() {
    syncData_.withWLock([&](auto& data) { data.blockUntilActiveAbort = true; });
  }
};

static const std::chrono::milliseconds kHostAlgoFnWait =
    std::chrono::milliseconds(2000);

commResult_t CtranGpeFtTestAlgoFn(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  auto* sync = reinterpret_cast<FtTestSync*>(
      const_cast<void*>(opGroup.front()->send.sendbuff));

  sync->wait(kHostAlgoFnWait);

  if (sync->getBlockUntilActiveAbort()) {
    while (!opGroup.front()->comm_->testAbort())
      ;
    throw ctran::utils::Exception("CtranGpe FT UT aborted: ", commRemoteError);
  }
  if (sync->getTimeout()) {
    while (!opGroup.front()->comm_->testAbort()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return commSuccess;
  }
  auto exception = sync->getException();
  if (exception.has_value()) {
    throw exception.value();
  }
  auto res = sync->getResult();
  if (res.has_value()) {
    return res.value();
  }

  return commSuccess;
}

class CtranGpeFaultToleranceTestBase : public ::ctran::CtranStandaloneFixture {
 protected:
  static constexpr int kNumBlocks = 4;
  static constexpr int kNumThreads = 64;
  volatile int* testFlag;
  CtranAlgoDeviceState* devState_d{nullptr};

  std::unique_ptr<CtranComm> ctranComm{nullptr};

  cudaStream_t stream{nullptr};

  int* oobKernelTerminateFlag;

  void SetUpInternal(bool abortEnabled) {
    CtranStandaloneFixture::SetUp();

    ctranComm = makeCtranComm(::ctran::utils::createAbort(abortEnabled));

    ASSERT_CUDASUCCESS(cudaStreamCreate(&stream));

    ASSERT_CUDASUCCESS(cudaHostAlloc(
        (void**)&testFlag, kNumBlocks * sizeof(int), cudaHostAllocDefault));
    for (int i = 0; i < kNumBlocks; i++) {
      testFlag[i] = KERNEL_UNSET;
    }
    ASSERT_CUDASUCCESS(cudaHostAlloc(
        (void**)&oobKernelTerminateFlag, sizeof(int), cudaHostAllocDefault));
    *oobKernelTerminateFlag = 0;

    ASSERT_CUDASUCCESS(cudaMalloc(&devState_d, sizeof(CtranAlgoDeviceState)));
    if (ctranComm->abortEnabled()) {
      CtranAlgoDeviceState devState_h;
      devState_h.enableCancellableWaits = true;
      ASSERT_CUDASUCCESS(cudaMemcpy(
          devState_d, &devState_h, sizeof(devState_h), cudaMemcpyHostToDevice));
    }
  }
  void TearDown() override {
    ASSERT_CUDASUCCESS(cudaFree(devState_d));
    ASSERT_CUDASUCCESS(cudaFreeHost((void*)oobKernelTerminateFlag));
    ASSERT_CUDASUCCESS(cudaFreeHost((void*)testFlag));
    ASSERT_CUDASUCCESS(cudaStreamDestroy(stream));
  }

  // util similar to cudaStreamSynchronize with timeout, makes tests fail fast
  std::chrono::milliseconds tryQueryStreamFor(
      cudaStream_t stream,
      std::chrono::milliseconds patience) {
    auto startTs = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    while (cudaErrorNotReady == cudaStreamQuery(stream) &&
           (now - startTs) < patience) {
      now = std::chrono::high_resolution_clock::now();
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - startTs);
  }

  void launchKernelFn(
      CtranGpe* gpe,
      void* kernelFn,
      cudaStream_t stream,
      FtTestSync* sync,
      std::optional<std::chrono::milliseconds> timeout = std::nullopt);
};

void CtranGpeFaultToleranceTestBase::launchKernelFn(
    CtranGpe* gpe,
    void* kernelFn,
    cudaStream_t stream,
    FtTestSync* sync,
    std::optional<std::chrono::milliseconds> timeout) {
  commResult_t res = commSuccess;

  uint64_t dummyOpCount = 100;
  std::vector<std::unique_ptr<struct OpElem>> ops;
  auto op = std::make_unique<struct OpElem>(
      OpElem::opType::SEND, stream, ctranComm.get(), dummyOpCount);
  // hack to pass sync to the test opFunc
  op->send.sendbuff = sync;
  op->send.count = 0;
  op->send.datatype = commInt8;
  op->send.peerRank = 0;
  ops.push_back(std::move(op));

  auto kernelConfig = KernelConfig(
      KernelConfig::KernelType::SEND, stream, "dummyAlgo", dummyOpCount);
  kernelConfig.numBlocks = kNumBlocks;
  kernelConfig.numThreads = kNumThreads;
  kernelConfig.args.devState_d = devState_d;
  CtranKernelFtArgs args;
  args.terminate = oobKernelTerminateFlag;
  kernelConfig.algoArgs = &args;

  res = gpe->submit(
      std::move(ops), &CtranGpeFtTestAlgoFn, kernelConfig, kernelFn, timeout);

  EXPECT_EQ(res, commSuccess);
}

class CtranGpeFTDisabledTest : public CtranGpeFaultToleranceTestBase {
 protected:
  void SetUp() override {
    SetUpInternal(/*abortEnabled=*/false);
  }

  void runTestNoAbortFtDisabled(cudaStream_t stream, FtTestSync* sync);
};
TEST_F(CtranGpeFTDisabledTest, SetupTeardown) {
  ASSERT_FALSE(ctranComm->abortEnabled());
}

TEST_F(CtranGpeFTDisabledTest, NoError) {
  ASSERT_FALSE(ctranComm->abortEnabled());
  FtTestSync sync;
  runTestNoAbortFtDisabled(stream, &sync);
  EXPECT_FALSE(ctranComm->testAbort());
}

TEST_F(CtranGpeFTDisabledTest, HostAlgoFnException) {
  ASSERT_FALSE(ctranComm->abortEnabled());
  FtTestSync sync;
  sync.setException(ctran::utils::Exception("test exception", commRemoteError));
  runTestNoAbortFtDisabled(stream, &sync);
  EXPECT_FALSE(ctranComm->testAbort());
}

TEST_F(CtranGpeFTDisabledTest, HostAlgoFnReturnError) {
  ASSERT_FALSE(ctranComm->abortEnabled());
  FtTestSync sync;
  sync.setResult(commRemoteError);
  runTestNoAbortFtDisabled(stream, &sync);
  EXPECT_FALSE(ctranComm->testAbort());
}

void CtranGpeFTDisabledTest::runTestNoAbortFtDisabled(
    cudaStream_t stream,
    FtTestSync* sync) {
  auto gpe = std::make_unique<CtranGpe>(cudaDev, ctranComm.get());

  this->launchKernelFn(
      gpe.get(), (void*)CtranGpeTestFtDisabledOobTerminateKernel, stream, sync);

  // wait a bit
  usleep(50 * 1000);

  // the kernel should be blocked at the moment
  EXPECT_EQ(cudaErrorNotReady, cudaStreamQuery(stream));
  EXPECT_EQ(ctranComm->getAsyncResult(), commSuccess)
      << "gpe thread should be blocked, and not reporting any errors";

  // gpe thread host AlgoFn unblock
  sync->signal();

  // kernel cannot terminate
  const auto patience = kHostAlgoFnWait + std::chrono::milliseconds(1000);
  auto waitMs = tryQueryStreamFor(stream, patience);
  ASSERT_GE(waitMs, patience)
      << "kernel aborted without FaultTolerance feature, stream query status: "
      << cudaGetErrorString(cudaStreamQuery(stream));

  // Terminate Oob kernel to avoid long blocking in test cases
  *oobKernelTerminateFlag = true;
}

class CtranGpeFTEnabledTest : public CtranGpeFaultToleranceTestBase {
 protected:
  void SetUp() override {
    SetUpInternal(/*abortEnabled=*/true);
  }

  void runTestNoAbortFtEnabled(cudaStream_t stream, FtTestSync* sync);
};

TEST_F(CtranGpeFTEnabledTest, SetupTeardown) {
  ASSERT_TRUE(ctranComm->abortEnabled());
}

TEST_F(CtranGpeFTEnabledTest, NoError) {
  ASSERT_TRUE(ctranComm->abortEnabled());
  FtTestSync sync;
  runTestNoAbortFtEnabled(stream, &sync);
  EXPECT_FALSE(ctranComm->testAbort());
}

TEST_F(CtranGpeFTEnabledTest, HostAlgoFnExceptionErrorChecking) {
  ASSERT_TRUE(ctranComm->abortEnabled());
  FtTestSync sync;
  sync.setException(ctran::utils::Exception("test exception", commRemoteError));
  ASSERT_FALSE(sync.getTimeout());
  runTestNoAbortFtEnabled(stream, &sync);
  EXPECT_TRUE(ctranComm->testAbort());
}

TEST_F(CtranGpeFTEnabledTest, HostAlgoFnReturnErrorErrorChecking) {
  ASSERT_TRUE(ctranComm->abortEnabled());
  FtTestSync sync;
  sync.setResult(commRemoteError);
  ASSERT_FALSE(sync.getTimeout());
  runTestNoAbortFtEnabled(stream, &sync);
  EXPECT_TRUE(ctranComm->testAbort());
}

void CtranGpeFTEnabledTest::runTestNoAbortFtEnabled(
    cudaStream_t stream,
    FtTestSync* sync) {
  ASSERT_TRUE(ctranComm->abortEnabled()) << "feature is not enabled";

  auto gpe = std::make_unique<CtranGpe>(cudaDev, ctranComm.get());

  this->launchKernelFn(
      gpe.get(), (void*)CtranGpeTestFtEnabledOobTerminateKernel, stream, sync);

  // wait a bit
  usleep(50 * 1000);

  // the kernel should be blocked at the moment
  EXPECT_EQ(cudaErrorNotReady, cudaStreamQuery(stream));
  EXPECT_EQ(ctranComm->getAsyncResult(), commSuccess)
      << "gpe thread should be blocked, and not reporting any errors";

  // For FT enabled test cases, let OobKernel terminate. These are used for
  // asyncEx report checking only.
  //
  // This kernel still calls KernelWaitGpeTerminate to ensure logic there is
  // working correctly.
  *oobKernelTerminateFlag = true;

  // gpe thread host AlgoFn unblock
  sync->signal();

  // ensure kernel complete
  tryQueryStreamFor(stream, kHostAlgoFnWait + std::chrono::milliseconds(1000));
  auto ok = cudaStreamQuery(stream) == cudaSuccess; // kernel terminated
  EXPECT_TRUE(ok) << "kernel did not terminate";
  // fast termination on failed tests instead of hang
  if (!ok) {
    abort();
  }

  // For kernels terminated on kernelFlag, cudaKernel terminate
  // indicates GpeThread reported host AlgoFn error already.
  if (sync->getResult().has_value() || sync->getException().has_value()) {
    // If we have an error, we should have an async error
    EXPECT_NE(ctranComm->getAsyncResult(), commSuccess);
  }
}

class CtranGpeFTEnabledAbortTest
    : public CtranGpeFTEnabledTest,
      public ::testing::WithParamInterface<std::tuple<std::string, void*>> {
 protected:
  void runTestWillAbort(
      void* kernelFn,
      cudaStream_t stream,
      FtTestSync* sync,
      bool activeAbort = false,
      std::chrono::milliseconds statusCheckDelay = kHostAlgoFnWait -
          std::chrono::milliseconds(1000),
      std::optional<std::chrono::milliseconds> timeout = std::nullopt);
};

void CtranGpeFTEnabledAbortTest::runTestWillAbort(
    void* kernelFn,
    cudaStream_t stream,
    FtTestSync* sync,
    bool activeAbort,
    std::chrono::milliseconds statusCheckDelay,
    std::optional<std::chrono::milliseconds> timeout) {
  ASSERT_TRUE(ctranComm->abortEnabled());

  auto gpe = std::make_unique<CtranGpe>(cudaDev, ctranComm.get());

  this->launchKernelFn(gpe.get(), kernelFn, stream, sync, timeout);

  // wait a bit
  usleep(50 * 1000);

  // the kernel should be blocked at the moment
  EXPECT_EQ(cudaErrorNotReady, cudaStreamQuery(stream));
  EXPECT_EQ(ctranComm->getAsyncResult(), commSuccess)
      << "gpe thread should be blocked, and not reporting any errors";

  // gpe thread host AlgoFn unblock, allow kernel terminate if not Oob
  sync->signal();

  auto waitMs = tryQueryStreamFor(stream, /*patience=*/statusCheckDelay);

  if (activeAbort) {
    CLOGF(INFO, "host active abort");
    ctranComm->setAbort();
    ASSERT_TRUE(ctranComm->testAbort());
    EXPECT_GE(waitMs, statusCheckDelay) << "kernel unblocked too early";
    // spin extra 1s to allow terminate from active abort
    tryQueryStreamFor(stream, /*patience=*/std::chrono::milliseconds(1000));
  }

  // ensure kernel complete
  auto ok = cudaStreamQuery(stream) == cudaSuccess; // kernel terminated
  EXPECT_TRUE(ok) << "kernel did not terminate";
  // fast termination on failed tests instead of hang
  if (!ok) {
    abort();
  }

  // For kernels terminated on kernelFlag, cudaKernel terminate
  // indicates GpeThread reported host AlgoFn error already.
  if (sync->getResult().has_value() || sync->getException().has_value()) {
    // If we have an error, we should have an async error
    EXPECT_NE(ctranComm->getAsyncResult(), commSuccess);
  }
}

TEST_P(CtranGpeFTEnabledAbortTest, HostDetectedTimeout) {
  auto [name, kernelFn] = GetParam();

  ASSERT_TRUE(ctranComm->abortEnabled());
  FtTestSync sync;
  ASSERT_FALSE(sync.getException().has_value());
  ASSERT_FALSE(sync.getResult().has_value());
  ASSERT_FALSE(sync.getBlockUntilActiveAbort());
  sync.setTimeout();
  runTestWillAbort(
      kernelFn,
      stream,
      &sync,
      /*activeAbort=*/false,
      /*statusCheckDelay=*/kHostAlgoFnWait + std::chrono::milliseconds(1000),
      /*timeout=*/kHostAlgoFnWait - std::chrono::milliseconds(500));
  EXPECT_TRUE(ctranComm->testAbort());
}

TEST_P(CtranGpeFTEnabledAbortTest, HostActiveAbort) {
  auto [name, kernelFn] = GetParam();

  ASSERT_TRUE(ctranComm->abortEnabled());
  FtTestSync sync;
  ASSERT_FALSE(sync.getException().has_value());
  ASSERT_FALSE(sync.getResult().has_value());
  ASSERT_FALSE(sync.getTimeout());
  sync.setBlockUntilActiveAbort();
  // no error + no exception + no timeout indicates active abort
  runTestWillAbort(
      kernelFn,
      stream,
      &sync,
      /*activeAbort=*/true,
      /*statusCheckDelay=*/kHostAlgoFnWait - std::chrono::milliseconds(1000));
  EXPECT_TRUE(ctranComm->testAbort());
}

TEST_P(
    CtranGpeFTEnabledAbortTest,
    HostActiveAbortKernelAfterHostAlgoFnTerminate) {
  auto [name, kernelFn] = GetParam();

  if (kernelFn == (void*)CtranGpeTestFtKernelSkipGpeStart) {
    CLOGF(
        INFO,
        "CtranGpeTestKernelSkipGpeStart here is equivalent to HostActiveAbort "
        "test, since it will block before starting HostAlgoFn");
  }

  ASSERT_TRUE(ctranComm->abortEnabled());
  FtTestSync sync;
  ASSERT_FALSE(sync.getException().has_value());
  ASSERT_FALSE(sync.getResult().has_value());
  ASSERT_FALSE(sync.getTimeout());
  ASSERT_FALSE(sync.getBlockUntilActiveAbort());
  // no error + no exception + no timeout indicates active abort
  runTestWillAbort(
      kernelFn,
      stream,
      &sync,
      /*activeAbort=*/true,
      /*statusCheckDelay=*/kHostAlgoFnWait + std::chrono::milliseconds(1000));
  EXPECT_TRUE(ctranComm->testAbort());
}

INSTANTIATE_TEST_SUITE_P(
    CtranGpeFTEnabledAbortTest,
    CtranGpeFTEnabledAbortTest,
    ::testing::Values(
        std::make_tuple(
            "CtranGpeTestFtBaseKernel",
            (void*)CtranGpeTestFtBaseKernel),
        std::make_tuple(
            "CtranGpeTestFtShmAbortKernel",
            (void*)CtranGpeTestFtShmAbortKernel),
        std::make_tuple(
            "CtranGpeTestFtKernelSkipGpeStart",
            (void*)CtranGpeTestFtKernelSkipGpeStart),
        std::make_tuple(
            "CtranGpeTestFtKernelSkipGpeTerminate",
            (void*)CtranGpeTestFtKernelSkipGpeTerminate)),
    [](const ::testing::TestParamInfo<CtranGpeFTEnabledAbortTest::ParamType>&
           info) { return std::get<0>(info.param); });

class CtranGpeFTEnabledAbortFromErrorTest : public CtranGpeFTEnabledAbortTest {
  // parametrized test just to enable different set of testcases
};

TEST_P(CtranGpeFTEnabledAbortFromErrorTest, HostAlgoFnExceptionFtAbortKernel) {
  auto [name, kernelFn] = GetParam();

  ASSERT_TRUE(ctranComm->abortEnabled());
  FtTestSync sync;
  sync.setException(ctran::utils::Exception("test exception", commRemoteError));
  ASSERT_FALSE(sync.getTimeout());
  runTestWillAbort(kernelFn, stream, &sync);
  EXPECT_TRUE(ctranComm->testAbort());
}

TEST_P(
    CtranGpeFTEnabledAbortFromErrorTest,
    HostAlgoFnReturnErrorFtAbortKernel) {
  auto [name, kernelFn] = GetParam();

  ASSERT_TRUE(ctranComm->abortEnabled());
  FtTestSync sync;
  sync.setResult(commRemoteError);
  ASSERT_FALSE(sync.getTimeout());
  runTestWillAbort(kernelFn, stream, &sync);
  EXPECT_TRUE(ctranComm->testAbort());
}

INSTANTIATE_TEST_SUITE_P(
    CtranGpeFTEnabledAbortFromErrorTest,
    CtranGpeFTEnabledAbortFromErrorTest,
    ::testing::Values(
        std::make_tuple(
            "CtranGpeTestFtBaseKernel",
            (void*)CtranGpeTestFtBaseKernel),
        std::make_tuple(
            "CtranGpeTestFtShmAbortKernel",
            (void*)CtranGpeTestFtShmAbortKernel),
        // SkipGpeStart must be aborted by active abort or timeout. Hangs will
        // happen before injected error/exceptions which only happens in host
        // AlgoFn.
        std::make_tuple(
            "CtranGpeFtKernelSkipGpeTerminate",
            (void*)CtranGpeTestFtKernelSkipGpeTerminate)),
    [](const ::testing::TestParamInfo<
        CtranGpeFTEnabledAbortFromErrorTest::ParamType>& info) {
      return std::get<0>(info.param);
    });

} // namespace ctran::fttesting
