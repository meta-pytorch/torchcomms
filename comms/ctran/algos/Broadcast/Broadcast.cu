// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <stdio.h>
#include <cstddef>
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

#ifndef CTRAN_DISABLE_TCPDM
#include "comms/tcp_devmem/unpack/batch_unpack_kernel.cuh"
#include "comms/tcp_devmem/unpack/batch_unpack_kernel.h"
#include "comms/tcp_devmem/unpack/batch_unpack_kernel_device.h"

__shared__ UnpackBlockState broadcastUnpack;
#endif

template <bool UNPACK>
__global__ void __launch_bounds__(1024, 1) ncclKernelBroadcast(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelBroadcastArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState);
#ifndef CTRAN_DISABLE_TCPDM
  if (UNPACK) {
    waitUnpack(
        args.unpack.sq[blockIdx.x], &broadcastUnpack, flag, KERNEL_TERMINATE);
  }
#endif

  // For now kernel always ack GPE thread once remote notify arrives for
  // timepoint tracing
  if (args.waitNotifyList != nullptr) {
    ctranKernMultiWaitNotify<true /* Complete */, false /* Free */>(
        args.waitNotifyList);
  }

  // For now kernel always ack GPE thread once put completes for timepoint
  // tracing
  if (args.putNotifyList != nullptr) {
    ctranKernMultiPutNotify<true /* Complete */, false /* Free */>(
        args.putNotifyList);
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

#define DECL_BROADCAST_KERN(UNPACK)                     \
  template __global__ void ncclKernelBroadcast<UNPACK>( \
      int* flag,                                        \
      CtranAlgoDeviceState* devState,                   \
      CtranKernelBroadcastArgs args)

DECL_BROADCAST_KERN(/*UNPACK=*/false);
DECL_BROADCAST_KERN(/*UNPACK=*/true);
