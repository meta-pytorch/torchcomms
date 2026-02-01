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

__shared__ UnpackBlockState sendRecvUnpack;
#endif

__global__ void __launch_bounds__(1024, 1) ncclKernelSend(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernelSendArgs args) {
  const auto tId = threadIdx.x;
  const auto bId = blockIdx.x;

  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  devStateLoadToShm(&flag[bId], devState);

  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(&flag[bId]);
  }

  // For now kernel always ack GPE thread once put completes for timepoint
  // tracing
  if (args.putNotifyList != nullptr) {
    ctranKernMultiPutNotify<true /* Complete */, false /* Free */>(
        args.putNotifyList);
  }

  if (flag && tId == 0) {
    ctran::device::KernelWaitGpeTerminate(&flag[bId]);
  }
}

// Receive kernel might need non-trivial number of blocks because
// it can potentially require unpack with TCP DM backend.
template <bool UNPACK>
__global__ void __launch_bounds__(1024, 1) ncclKernelRecv(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernelRecvArgs args) {
  const auto tId = threadIdx.x;
  const auto bId = blockIdx.x;

  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  devStateLoadToShm(&flag[bId], devState);

  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(&flag[bId]);
  }

#ifndef CTRAN_DISABLE_TCPDM
  if (UNPACK) {
    waitUnpack(
        args.unpack.sq[bId], &sendRecvUnpack, &flag[bId], KERNEL_TERMINATE);
  }
#endif

  // For now kernel always ack GPE thread once remote notify arrives for
  // timepoint tracing
  if (args.waitNotifyList != nullptr) {
    ctranKernMultiWaitNotify<true /* Complete */, false /* Free */>(
        args.waitNotifyList);
  }

  if (flag && tId == 0) {
    ctran::device::KernelWaitGpeTerminate(&flag[bId]);
  }
}

template <bool UNPACK>
__global__ void __launch_bounds__(1024, 1) ncclKernelSendRecv(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernelSendRecvArgs args) {
  const auto tId = threadIdx.x;
  const auto bId = blockIdx.x;

  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  devStateLoadToShm(&flag[bId], devState);

  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(&flag[bId]);
  }

#ifndef CTRAN_DISABLE_TCPDM
  if (UNPACK) {
    waitUnpack(
        args.unpack.sq[bId], &sendRecvUnpack, &flag[bId], KERNEL_TERMINATE);
  }
#endif

  // For now kernel always ack GPE thread once put completes for timepoint
  // tracing
  if (args.putNotifyList != nullptr) {
    ctranKernMultiPutNotify<true /* Complete */, false /* Free */>(
        args.putNotifyList);
  }

  // For now kernel always ack GPE thread once remote notify arrives for
  // timepoint tracing
  if (args.waitNotifyList != nullptr) {
    ctranKernMultiWaitNotify<true /* Complete */, false /* Free */>(
        args.waitNotifyList);
  }

  if (flag && tId == 0) {
    ctran::device::KernelWaitGpeTerminate(&flag[bId]);
  }
}

__global__ void __launch_bounds__(1024, 1) ncclKernelSendRecvNotifyOnly(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernelSendRecvArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState);

  if (args.putNotifyList != nullptr) {
    ctranKernMultiNotifyOnly<false /* Complete */, true /* Free */>(
        args.putNotifyList);
  }

  if (args.waitNotifyList != nullptr) {
    ctranKernMultiWaitNotifyOnly<false /* Complete */, true /* Free */>(
        args.waitNotifyList);
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

__global__ void __launch_bounds__(1024, 1) ncclKernelSendNotifyOnly(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernelSendArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState);

  if (args.putNotifyList != nullptr) {
    ctranKernMultiNotifyOnly<false /* Complete */, true /* Free */>(
        args.putNotifyList);
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

__global__ void __launch_bounds__(1, 1) ncclKernelRecvNotifyOnly(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernelRecvArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState);

  if (args.waitNotifyList != nullptr) {
    ctranKernMultiWaitNotifyOnly<false /* Complete */, true /* Free */>(
        args.waitNotifyList);
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

#define DECL_SENDRECV_KERN(UNPACK)                     \
  template __global__ void ncclKernelRecv<UNPACK>(     \
      int* flag,                                       \
      CtranAlgoDeviceState* devState,                  \
      ctran::sendrecv::KernelRecvArgs args);           \
  template __global__ void ncclKernelSendRecv<UNPACK>( \
      int* flag,                                       \
      CtranAlgoDeviceState* devState,                  \
      ctran::sendrecv::KernelSendRecvArgs args)

DECL_SENDRECV_KERN(/*UNPACK=*/false);
DECL_SENDRECV_KERN(/*UNPACK=*/true);
