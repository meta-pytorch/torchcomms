// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/gpe/tests/CtranGpeFaultToleranceUTBase.h"

namespace ctran::fttesting {

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

// Submit with no per-op timeout; comm-level default fires the abort.
TEST_P(CtranGpeFTEnabledAbortTest, HostDetectedTimeoutFromDefault) {
  auto [name, kernelFn] = GetParam();

  ASSERT_TRUE(ctranComm->abortEnabled());
  FtTestSync sync;
  sync.setTimeout();
  ctranComm->getAbort()->SetDefaultTimeoutDuration(
      kHostAlgoFnWait - std::chrono::milliseconds(500));
  runTestWillAbort(
      kernelFn,
      stream,
      &sync,
      /*activeAbort=*/false,
      /*statusCheckDelay=*/kHostAlgoFnWait + std::chrono::milliseconds(1000),
      /*timeout=*/std::nullopt);
  EXPECT_TRUE(ctranComm->testAbort());
}

// Per-op timeout wins when both per-op and comm-level default are set.
// The default is set far longer than the test window; abort within window
// proves the per-op value was used.
TEST_P(CtranGpeFTEnabledAbortTest, HostDetectedTimeoutPerOpOverridesDefault) {
  auto [name, kernelFn] = GetParam();

  ASSERT_TRUE(ctranComm->abortEnabled());
  FtTestSync sync;
  sync.setTimeout();
  ctranComm->getAbort()->SetDefaultTimeoutDuration(
      kHostAlgoFnWait + std::chrono::milliseconds(5000));
  runTestWillAbort(
      kernelFn,
      stream,
      &sync,
      /*activeAbort=*/false,
      /*statusCheckDelay=*/kHostAlgoFnWait + std::chrono::milliseconds(1000),
      /*timeout=*/kHostAlgoFnWait - std::chrono::milliseconds(500));
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
            (void*)CtranGpeTestFtShmAbortKernel)),
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
            (void*)CtranGpeTestFtShmAbortKernel)),
    [](const ::testing::TestParamInfo<
        CtranGpeFTEnabledAbortFromErrorTest::ParamType>& info) {
      return std::get<0>(info.param);
    });

// Impl function that sets a global flag to prove it was called.
static std::atomic<bool> g_secondImplCalled{false};

commResult_t CtranGpeFtTestRecordCallAlgoFn(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  g_secondImplCalled.store(true);
  return commSuccess;
}

// Test: after abort from first collective, second collective's impl is skipped.
// This verifies the fix for the double-complete bug where progressInternal()
// would access stale VC queue entries from the previously aborted collective.
TEST_F(CtranGpeFTEnabledTest, SecondCollectiveSkippedAfterAbort) {
  ASSERT_TRUE(ctranComm->abortEnabled());
  g_secondImplCalled.store(false);

  auto gpe = std::make_unique<CtranGpe>(cudaDev, ctranComm.get());

  // Collective 1: will throw, causing abort
  FtTestSync sync1;
  sync1.setException(
      ctran::utils::Exception("test abort exception", commRemoteError));

  this->launchKernelFn(
      gpe.get(),
      (void*)CtranGpeTestFtEnabledOobTerminateKernel,
      stream,
      &sync1);

  // Let OobKernel terminate and unblock collective 1's impl
  *oobKernelTerminateFlag = true;
  sync1.signal();

  // Wait for collective 1 to complete and abort to be set
  tryQueryStreamFor(stream, kHostAlgoFnWait + std::chrono::milliseconds(1000));
  ASSERT_TRUE(ctranComm->testAbort()) << "comm should be aborted after coll 1";

  // Submit collective 2 with a different impl — should be skipped
  this->launchKernelFn(
      gpe.get(),
      (void*)CtranGpeTestFtEnabledOobTerminateKernel,
      stream,
      nullptr,
      std::nullopt,
      &CtranGpeFtTestRecordCallAlgoFn);

  // Wait for GPE thread to process collective 2.
  // Busy-wait on kernel flags rather than sleeping to avoid flakiness.
  while (gpe->numInUseKernelFlags() > 0) {
  }

  EXPECT_FALSE(g_secondImplCalled.load())
      << "second collective's impl should be skipped after abort";
}

} // namespace ctran::fttesting
