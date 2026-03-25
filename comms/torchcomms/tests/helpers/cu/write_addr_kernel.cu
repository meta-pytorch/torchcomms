// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstdint>

__global__ void writeAddrKernel(const void* buf, int64_t* probe) {
  probe[0] = reinterpret_cast<int64_t>(buf);
}

void launchWriteAddr(const void* buf, int64_t* probe, cudaStream_t stream) {
  writeAddrKernel<<<1, 1, 0, stream>>>(buf, probe);
}
