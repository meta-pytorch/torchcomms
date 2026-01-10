// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/algos/tests/CtranDevWaitUTKernels.cuh"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"

namespace ctran::testing {

class CtranDeviceWaitUT : public CtranStandaloneFixture {
 protected:
  void SetUp() override {
    CtranStandaloneFixture::SetUp();

    FB_CUDACHECKTHROW(
        cudaHostAlloc(&flag_, kNBlocks * sizeof(int), cudaHostAllocDefault));
    for (int i = 0; i < kNBlocks; i++) {
      *flag_ = KERNEL_STARTED;
    }
    FB_COMMCHECKTHROW_EX(
        ctran::utils::commCudaMalloc(
            &devState_, 1, /*logMetaData=*/nullptr, "CtranDeviceWaitUT"),
        /*rank=*/0,
        /*commHash=*/0,
        /*commDesc=*/std::string(""));

    memset(&args_, 0, sizeof(args_));
    FB_CUDACHECKTHROW(
        cudaHostAlloc(&args_.h2d, sizeof(*args_.h2d), cudaHostAllocDefault));
    args_.h2d->intSync = 0;
    args_.h2d->elem.ngroups = kNBlocks;
    args_.h2d->elem.setStatus(KernelElem::INUSE);
    args_.h2d->gpeKernelSync.nworkers = kNBlocks;
    args_.h2d->gpeKernelSync.resetStatus();

    FB_COMMCHECKTHROW_EX(
        ctran::utils::commCudaMalloc(
            &args_.d2h, 1, /*logMetaData=*/nullptr, "CtranDeviceWaitUT"),
        /*rank=*/0,
        /*commHash=*/0,
        /*commDesc=*/std::string(""));
    FB_CUDACHECKTHROW(cudaMemset(args_.d2h, 0, sizeof(*args_.d2h)));

    d2h_.revoked = false;
    d2h_.warpTested = false;
  }
  void TearDown() override {
    FB_COMMCHECKTHROW_EX(
        ctran::utils::commCudaFree(args_.d2h),
        /*rank=*/0,
        /*commHash=*/0,
        /*commDesc=*/std::string(""));
    FB_CUDACHECKTHROW(cudaFreeHost(args_.h2d));
    FB_COMMCHECKTHROW_EX(
        ctran::utils::commCudaFree(devState_),
        /*rank=*/0,
        /*commHash=*/0,
        /*commDesc=*/std::string(""));
    FB_CUDACHECKTHROW(cudaFreeHost(flag_));
  }

  void waitStreamFor(bool expectFinish, std::chrono::seconds secs) {
    auto start = std::chrono::high_resolution_clock::now();
    while (cudaErrorNotReady == cudaStreamQuery(stream_)) {
      auto now = std::chrono::high_resolution_clock::now();
      if (now - start > std::chrono::seconds(2)) {
        break;
      }
    }
    if (expectFinish) {
      EXPECT_EQ(cudaStreamQuery(stream_), cudaSuccess);
    } else {
      EXPECT_EQ(cudaStreamQuery(stream_), cudaErrorNotReady);
    }
  }

  void verifyKernelUnblockBehavior(FnName fnName, bool enableCancellableWaits) {
    args_.h2d->fnName = fnName;

    FB_CUDACHECKTHROW(cudaMemcpy(
        &devState_->enableCancellableWaits,
        &enableCancellableWaits,
        sizeof(bool),
        cudaMemcpyHostToDevice));

    std::array<void*, 3> kernelArgs;
    kernelArgs.at(0) = (void*)&flag_;
    kernelArgs.at(1) = (void*)&devState_;
    kernelArgs.at(2) = (void*)&args_;
    dim3 grid = {kNBlocks, 1, 1};
    dim3 block = {kNThreads, 1, 1};
    const void* kernel = (void*)&ncclKernelTestDeviceWait;
    constexpr auto dynamicShmSize = sizeof(CtranAlgoDeviceState);
    ASSERT_EQ(
        cudaSuccess,
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamicShmSize));
    ASSERT_EQ(
        cudaSuccess,
        cudaLaunchKernel(
            kernel, grid, block, kernelArgs.data(), dynamicShmSize, stream_));

    waitStreamFor(/*expectFinish=*/false, std::chrono::seconds(2));

    flag_[0] = KERNEL_HOST_ABORT;

    waitStreamFor(/*expectFinish=*/false, std::chrono::seconds(2));

    for (int i = 1; i < kNBlocks; ++i) {
      flag_[i] = KERNEL_HOST_ABORT;
    }

    waitStreamFor(
        /*expectFinish=*/enableCancellableWaits, std::chrono::seconds(2));

    if (!enableCancellableWaits) {
      // terminate the kernels in the normal way
      switch (fnName) {
        case FnName::elemWaitPostOrRevoke:
          args_.h2d->elem.setStatus(KernelElem::ElemStatus::POSTED);
          break;
        case FnName::devSyncWaitStep:
          // device side is also expecting the step value 1
          args_.h2d->intSync = 1;
          break;
        case FnName::devSyncSetNotify:
          for (int i = 0; i < kNBlocks; ++i) {
            args_.h2d->devSync.syncs[i].stepOnSameBlockIdx =
                CTRAN_ALGO_NOTIFY_RESET;
          }
          break;
        case FnName::devSyncWaitNotify:
          for (int i = 0; i < kNBlocks; ++i) {
            args_.h2d->devSync.syncs[i].stepOnSameBlockIdx =
                CTRAN_ALGO_NOTIFY_SET;
          }
          break;
        case FnName::waitPost:
        case FnName::waitPostWarp:
          args_.h2d->gpeKernelSync.post(/*step=*/1);
          break;
      }
    }
    cudaMemcpy(&d2h_, args_.d2h, sizeof(d2h_), cudaMemcpyDeviceToHost);
  }

  int* flag_;
  CtranAlgoDeviceState* devState_;
  CtranTestDeviceWaitArgs args_;

  CtranTestDeviceWaitArgs::D2h d2h_;

  cudaStream_t stream_ = 0;

  static constexpr int kNBlocks = 2;
  static_assert(kNBlocks > 1);
  static constexpr int kNThreads = 512;
  static_assert(kNThreads > 2 * /*warpsize=*/32);
};

class CtranDeviceWaitUTCancelEnabled : public CtranDeviceWaitUT {
 protected:
  void runTestBody(FnName fnName) {
    return verifyKernelUnblockBehavior(fnName, /*enableCancellableWaits=*/true);
  }
};

TEST_F(CtranDeviceWaitUTCancelEnabled, elemWaitPostOrRevoke) {
  runTestBody(FnName::elemWaitPostOrRevoke);

  EXPECT_TRUE(d2h_.revoked);
}

TEST_F(CtranDeviceWaitUTCancelEnabled, devSyncWaitStep) {
  runTestBody(FnName::devSyncWaitStep);
}
TEST_F(CtranDeviceWaitUTCancelEnabled, devSyncSetNotify) {
  runTestBody(FnName::devSyncSetNotify);
}
TEST_F(CtranDeviceWaitUTCancelEnabled, devSyncWaitNotify) {
  runTestBody(FnName::devSyncWaitNotify);
}
TEST_F(CtranDeviceWaitUTCancelEnabled, GpeKernelSyncWaitPost) {
  runTestBody(FnName::waitPost);
}
TEST_F(CtranDeviceWaitUTCancelEnabled, GpeKernelSyncWaitPostWarp) {
  runTestBody(FnName::waitPostWarp);

  EXPECT_TRUE(d2h_.warpTested);
}

class CtranDeviceWaitUTCancelDisabled : public CtranDeviceWaitUT {
 protected:
  void runTestBody(FnName fnName) {
    return verifyKernelUnblockBehavior(
        fnName, /*enableCancellableWaits=*/false);
  }
};

TEST_F(CtranDeviceWaitUTCancelDisabled, elemWaitPostOrRevoke) {
  runTestBody(FnName::elemWaitPostOrRevoke);

  EXPECT_FALSE(d2h_.revoked);
}

TEST_F(CtranDeviceWaitUTCancelDisabled, devSyncWaitStep) {
  runTestBody(FnName::devSyncWaitStep);
}
TEST_F(CtranDeviceWaitUTCancelDisabled, devSyncSetNotify) {
  runTestBody(FnName::devSyncSetNotify);
}
TEST_F(CtranDeviceWaitUTCancelDisabled, devSyncWaitNotify) {
  runTestBody(FnName::devSyncWaitNotify);
}
TEST_F(CtranDeviceWaitUTCancelDisabled, GpeKernelSyncWaitPost) {
  runTestBody(FnName::waitPost);
}
TEST_F(CtranDeviceWaitUTCancelDisabled, GpeKernelSyncWaitPostWarp) {
  runTestBody(FnName::waitPostWarp);
}

} // namespace ctran::testing
