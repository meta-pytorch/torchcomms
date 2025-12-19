// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAllvDedup/ExecFwd.cuh"
#include "comms/ctran/algos/AllToAllvDedup/ExecRecvCopy.cuh"
#include "comms/ctran/algos/AllToAllvDedup/ExecSendCopy.cuh"

namespace ctran::alltoallvdedup {

template <typename T>
__global__ void ncclKernelAllToAllvDedup(
    int* flag,
    CtranAlgoDeviceState* devState,
    ExecKernArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  if (threadIdx.x == 0) {
    // TODO: move it into a generic routine for all kernels
    devState->opCount = args.opCount;
  }

  devStateLoadToShm(devState);
  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE("loaded devState %p\n", devState);
  }

  WorkerGroup sendG, fwdG, recvG, intraFwdG, intraRecvG;
  assignWorkerGroups(args, sendG, fwdG, recvG, intraFwdG, intraRecvG);

  if (sendG.contains(blockIdx.x)) {
    progressSendCopy<T>(args, sendG);
  } else if (fwdG.contains(blockIdx.x)) {
    progressFwd<T>(args, fwdG);
  } else if (recvG.contains(blockIdx.x)) {
    progressRecvCopy<T>(args, recvG);
  } else if (intraFwdG.contains(blockIdx.x)) {
    progressIntraFwd<T>(args, intraFwdG);
  } else if (intraRecvG.contains(blockIdx.x)) {
    progressIntraRecvCopy<T>(args, intraRecvG);
  }

  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE("Finished progress loop\n");
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
  if (threadIdx.x == 0) {
    CTRAN_DEV_TRACE("exit kernel\n");
  }
}

#define DECL_ALLTOALLVDEDUP_KERN(T)                     \
  template __global__ void ncclKernelAllToAllvDedup<T>( \
      int* flag, CtranAlgoDeviceState* devState, ExecKernArgs args)

DECL_ALLTOALLVDEDUP_KERN(int8_t);
DECL_ALLTOALLVDEDUP_KERN(uint8_t);
DECL_ALLTOALLVDEDUP_KERN(int32_t);
DECL_ALLTOALLVDEDUP_KERN(uint32_t);
DECL_ALLTOALLVDEDUP_KERN(int64_t);
DECL_ALLTOALLVDEDUP_KERN(uint64_t);
DECL_ALLTOALLVDEDUP_KERN(half);
DECL_ALLTOALLVDEDUP_KERN(float);
DECL_ALLTOALLVDEDUP_KERN(double);
#if defined(__CUDA_BF16_TYPES_EXIST__)
DECL_ALLTOALLVDEDUP_KERN(__nv_bfloat16);
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
DECL_ALLTOALLVDEDUP_KERN(__nv_fp8_e4m3);
DECL_ALLTOALLVDEDUP_KERN(__nv_fp8_e5m2);
#endif

} // namespace ctran::alltoallvdedup
