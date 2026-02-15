// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/common/AtomicUtils.cuh"
#include "comms/ctran/algos/AllGather/Types.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/common/GpeKernelDev.cuh"
// FIXME [REBASE]: update the path once moved to fbcode/comms
#include "comms/ctran/gpe/tests/CtranGpeUTKernels.h"

__global__ void CtranGpeTestKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    ctran::allgather::KernelArgs args) {
  int* a = const_cast<int*>(reinterpret_cast<const int*>(args.sendbuff));
  int* expValInt = reinterpret_cast<int*>(args.recvbuff);
  // Assume data type = commInt8
  size_t count = args.count;

  if (flag) {
    ctran::device::KernelStartGpe(flag);
  }

  for (int i = 0; i < count; i++) {
    a[i] = *expValInt;
  }

  if (flag) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

__global__ void CtranGpeTestCustomArgsKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelCustomArgs args) {
  const auto gtId = blockIdx.x * blockDim.x + threadIdx.x;

  for (auto i = gtId; i < args.numElems; i += gridDim.x * blockDim.x) {
    args.data[i] *= args.scaleFactor;
  }
  __syncthreads();
}

__global__ void CtranGpeTestTerminateKernel(int* flag) {
  ctran::device::KernelStartGpe(flag);
  ctran::device::KernelWaitGpeTerminate(flag);
}

__global__ void CtranGpeTestStartAndExitKernel(int* flag) {
  ctran::device::KernelStartGpeAndExit(flag);
}

__global__ void CtranGpeTestKElemsKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    ctran::allgather::KernelArgs args) {
  KernelElem* elemList = const_cast<KernelElem*>(
      reinterpret_cast<const KernelElem*>(args.sendbuff));
  KernelElem* elem = elemList;
  int numElems = numKElems;
  // consume only numKElems amount of objects
  while (numElems > 0) {
    elemFree(elem, blockIdx.x);
    elem = elem->next;
    numElems--;
  }
}

__global__ void CtranGpeTestOneFlagKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    ctran::allgather::KernelArgs args) {
  auto gtIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (flag && gtIdx == 0) {
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState_d);

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

__global__ void CtranGpeTestPerBlockFlagKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    ctran::allgather::KernelArgs args) {
  auto tId = threadIdx.x;
  auto bId = blockIdx.x;
  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(&flag[bId]);
  }

  devStateLoadToShm(devState_d);

  __syncthreads();
  if (flag && tId == 0) {
    ctran::device::KernelWaitGpeTerminate(&flag[bId]);
  }
}

__global__ void CtranGpeTestFtDisabledOobTerminateKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args) {
  const auto tId = threadIdx.x;
  const auto bId = blockIdx.x;
  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(&flag[bId]);
  }

  devStateLoadToShm(devState_d);

  while (comms::device::ld_volatile_global(args.terminate) == 0)
    ;

  // FtDisabled: GpeThread will terminate after setting AsyncEx_, and will not
  // be able to terminate Kernel, so we don't wait here.
}

__global__ void CtranGpeTestFtEnabledOobTerminateKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args) {
  const auto tId = threadIdx.x;
  const auto bId = blockIdx.x;
  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(&flag[bId]);
  }

  devStateLoadToShm(devState_d);

  while (comms::device::ld_volatile_global(args.terminate) == 0)
    ;

  // FtEnabled: GpeThread will abort kernel after setting AsyncEx_.
  //
  // For kernel terminated on kernelFlag, cudaKernel terminate indicates
  // GpeThread reported host AlgoFn error already. Hence, this helps with
  // consistent error checking.
  __syncthreads();
  if (flag && tId == 0) {
    ctran::device::KernelWaitGpeTerminate(&flag[bId]);
  }
}

__global__ void CtranGpeTestFtBaseKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args) {
  const auto tId = threadIdx.x;
  const auto bId = blockIdx.x;
  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(&flag[bId]);
  }

  devStateLoadToShm(&flag[bId], devState_d);

  while (!ctran::device::KernelTestHostAbort(&flag[bId]))
    ;

  __syncthreads();
  if (flag && tId == 0) {
    ctran::device::KernelWaitGpeTerminate(&flag[bId]);
  }
}

__global__ void CtranGpeTestFtShmAbortKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args) {
  const auto tId = threadIdx.x;
  const auto bId = blockIdx.x;
  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(&flag[bId]);
  }

  devStateLoadToShm(&flag[bId], devState_d);

  while (!ctran::device::KernelTestHostAbortBlock(&flag[bId]))
    ;

  __syncthreads();
  if (flag && tId == 0) {
    ctran::device::KernelWaitGpeTerminate(&flag[bId]);
  }
}
