// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/common/AtomicUtils.cuh"
#include "comms/ctran/algos/AllGather/Types.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/common/GpeKernelDev.cuh"
// FIXME [REBASE]: update the path once moved to fbcode/comms
#include "comms/ctran/gpe/tests/CtranGpeUTKernels.h"

__global__ void CtranGpeTestKernel(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState_d,
    ctran::allgather::KernelArgs args) {
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  int* a = const_cast<int*>(reinterpret_cast<const int*>(args.sendbuff));
  int* expValInt = reinterpret_cast<int*>(args.recvbuff);
  // Assume data type = commInt8
  size_t count = args.count;

  if (flag) {
    ctran::device::KernelStartGpe(f);
  }

  for (int i = 0; i < count; i++) {
    a[i] = *expValInt;
  }

  if (flag) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

__global__ void CtranGpeTestCustomArgsKernel(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState_d,
    CtranKernelCustomArgs args) {
  const auto gtId = blockIdx.x * blockDim.x + threadIdx.x;

  for (auto i = gtId; i < args.numElems; i += gridDim.x * blockDim.x) {
    args.data[i] *= args.scaleFactor;
  }
  __syncthreads();
}

__global__ void CtranGpeTestTerminateKernel(ctran::gpe::KernelFlagDev* f) {
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  ctran::device::KernelStartGpe(f);
  ctran::device::KernelWaitGpeTerminate(flag);
}

__global__ void CtranGpeTestStartAndExitKernel(ctran::gpe::KernelFlagDev* f) {
  ctran::device::KernelStartGpeAndExit(f);
}

__global__ void CtranGpeTestKElemsKernel(
    ctran::gpe::KernelFlagDev* f,
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
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState_d,
    ctran::allgather::KernelArgs args) {
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  auto gtIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (flag && gtIdx == 0) {
    ctran::device::KernelStartGpe(f);
  }

  devStateLoadToShm(devState_d);

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

__global__ void CtranGpeTestPerBlockFlagKernel(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState_d,
    ctran::allgather::KernelArgs args) {
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  auto tId = threadIdx.x;
  auto bId = blockIdx.x;
  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(f, bId);
  }

  devStateLoadToShm(devState_d);

  __syncthreads();
  if (flag && tId == 0) {
    ctran::device::KernelWaitGpeTerminate(&flag[bId]);
  }
}

__global__ void CtranGpeTestFtDisabledOobTerminateKernel(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args) {
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  const auto tId = threadIdx.x;
  const auto bId = blockIdx.x;
  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(f, bId);
  }

  devStateLoadToShm(devState_d);

  while (comms::device::ld_volatile_global(args.terminate) == 0)
    ;

  __syncthreads();
  if (flag && tId == 0) {
    comms::device::st_volatile_global(&flag[bId], KERNEL_UNSET);
  }
}

__global__ void CtranGpeTestFtEnabledOobTerminateKernel(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args) {
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  const auto tId = threadIdx.x;
  const auto bId = blockIdx.x;
  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(f, bId);
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
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args) {
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  const auto tId = threadIdx.x;
  const auto bId = blockIdx.x;
  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(f, bId);
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
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args) {
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  const auto tId = threadIdx.x;
  const auto bId = blockIdx.x;
  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(f, bId);
  }

  devStateLoadToShm(&flag[bId], devState_d);

  while (!ctran::device::KernelTestHostAbortBlock(&flag[bId]))
    ;

  __syncthreads();
  if (flag && tId == 0) {
    ctran::device::KernelWaitGpeTerminate(&flag[bId]);
  }
}
