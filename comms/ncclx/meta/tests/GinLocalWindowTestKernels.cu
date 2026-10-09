// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ncclx/meta/tests/GinLocalWindowTestKernels.h"

namespace {

__global__ void putFromLocalWindowKernel(
    ncclWindow_t srcWin,
    ncclWindow_t dstWin,
    size_t bytes,
    ncclDevComm devComm) {
  ncclGin gin{devComm, 0};
  const unsigned int signalIndex = 0;
  const uint64_t signalValue = gin.readSignal(signalIndex);

  ncclGinBarrierSession<ncclCoopCta> bar{
      ncclCoopCta(), gin, ncclTeamTagWorld(), 0};
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);

  if (threadIdx.x == 0) {
    const int peer = (devComm.rank + 1) % devComm.nRanks;
    gin.put(
        ncclTeamWorld(devComm),
        peer,
        dstWin,
        devComm.rank * bytes,
        srcWin,
        0,
        bytes,
        ncclGin_WeakSignalInc{signalIndex});
  }
  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + 1);
  gin.flush(ncclCoopCta());
  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::None);
}

} // namespace

cudaError_t launchPutFromLocalWindow(
    ncclWindow_t srcWin,
    ncclWindow_t dstWin,
    size_t bytes,
    const ncclDevComm& devComm,
    cudaStream_t stream) {
  putFromLocalWindowKernel<<<1, 128, 0, stream>>>(
      srcWin, dstWin, bytes, devComm);
  return cudaGetLastError();
}
