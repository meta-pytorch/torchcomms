// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <folly/Synchronized.h>
#include <folly/synchronization/Baton.h>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/gpe/tests/CtranGpeUTKernels.h"
#include "comms/ctran/tests/CtranTestUtils.h"
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

inline constexpr std::chrono::milliseconds kHostAlgoFnWait{2000};

inline commResult_t CtranGpeFtTestAlgoFn(
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

    ctranComm =
        makeCtranComm(::comms::fault_tolerance::createAbort(abortEnabled));

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
      std::optional<std::chrono::milliseconds> timeout = std::nullopt,
      opFunc func = &CtranGpeFtTestAlgoFn,
      const void* sendbuff = nullptr) {
    commResult_t res = commSuccess;

    uint64_t dummyOpCount = 100;
    std::vector<std::unique_ptr<struct OpElem>> ops;
    auto op = std::make_unique<struct OpElem>(
        OpElem::opType::SEND, stream, ctranComm.get(), dummyOpCount);
    // hack to pass sync/data to the test opFunc via sendbuff
    op->send.sendbuff = sendbuff ? sendbuff : sync;
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

    res = gpe->submit(std::move(ops), func, kernelConfig, kernelFn, timeout);

    EXPECT_EQ(res, commSuccess);
  }
};

} // namespace ctran::fttesting
