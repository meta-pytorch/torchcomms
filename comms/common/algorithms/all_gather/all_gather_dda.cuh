// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "comms/common/IpcGpuBarrier.cuh"

namespace meta::comms {

template <typename T, int NRANKS, bool hasAcc>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
    __global__ void ddaAllGatherIpc(
        T* const* __restrict__ ipcbuffs,
        T* __restrict__ recvbuff,
        size_t count,
        const T* __restrict__ sendbuff,
        int selfRank,
        IpcGpuBarrier barrier,
        const T* __restrict__ acc) {
  // use uint4 to do 16-byte loads to maximize memory efficiency
  // We assume that count % countPerThread == 0. This assumption is enforced
  // before kernel launch
  // TODO: we should be able to deal with left over as well
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = count;

  // It is expensive to launch hipMemcpyAsync on ROCm
  // Move data copy here. Each block copies part of sendbuff data
  T* ipcbuff = ipcbuffs[selfRank];
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    *reinterpret_cast<uint4*>(&ipcbuff[idx]) =
        reinterpret_cast<const uint4*>(&sendbuff[idx])[0];
  }

  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      true /* hasSubsequentMemAccess */>();

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    // Store to the destination buffer.
#pragma unroll NRANKS
    for (int r = 0; r < NRANKS; ++r) {
      int srcRank = (selfRank + r) % NRANKS;
      int srcIdx = idx + srcRank * idxEnd;
      *reinterpret_cast<uint4*>(&recvbuff[srcIdx]) =
          reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][idx])[0];
    }
  }

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<
      true /* hasPreviousMemAccess */,
      false /* hasSubsequentMemAccess */>();
}

} // namespace meta::comms
