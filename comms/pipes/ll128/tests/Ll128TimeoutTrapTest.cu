// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/TimeoutUtils.h"
#include "comms/pipes/ll128/Ll128Ops.cuh"
#include "comms/pipes/ll128/Ll128Packet.cuh"
#include "comms/pipes/ll128/tests/Ll128TimeoutTrapTest.cuh"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::test {

__global__ void ll128_send_no_recv_kernel(
    const char* src,
    size_t nbytes,
    Ll128Packet* remote_ll128_buf,
    Timeout timeout) {
  auto group = make_warp_group();
  timeout.start();
  // Send data — the sender will poll for ACK (kLl128ReadyToWrite) which
  // never arrives because there is no receiver, causing timeout + __trap().
  ll128_send(group, src, nbytes, remote_ll128_buf, timeout);
}

void launch_ll128_send_no_recv_timeout(int device, uint32_t timeout_ms) {
  PIPES_CUDA_CHECK(cudaSetDevice(device));

  constexpr size_t nbytes = 4096;

  char* src_d;
  PIPES_CUDA_CHECK(cudaMalloc(&src_d, nbytes));
  PIPES_CUDA_CHECK(cudaMemset(src_d, 0x42, nbytes));

  // Allocate LL128 buffer and set all flags to 1 (NOT kLl128ReadyToWrite).
  // This simulates a buffer that never gets ACKed by a receiver.
  size_t buf_size = ll128_buffer_size(nbytes);
  Ll128Packet* ll128_buf;
  PIPES_CUDA_CHECK(cudaMalloc(&ll128_buf, buf_size));
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf, 0, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  auto timeout = makeTimeout(timeout_ms, device);

  ll128_send_no_recv_kernel<<<1, 256>>>(src_d, nbytes, ll128_buf, timeout);
  // Don't check launch — the kernel will trap.

  cudaDeviceSynchronize();
  // Caller checks cudaGetLastError() for trap error.

  // Cleanup is best-effort since device may be corrupted after trap.
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaFree(src_d);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaFree(ll128_buf);
}

// =============================================================================
// Undersized buffer kernel — buffer_num_packets < kLl128PacketsPerWarp
// should trigger PIPES_DEVICE_CHECK and __trap().
// =============================================================================

__global__ void ll128_send_recv_undersized_buffer_kernel(
    const char* src,
    char* dst,
    size_t nbytes,
    Ll128Packet* ll128_buf,
    size_t buffer_num_packets) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(2);
  Timeout timeout;
  timeout.start();

  if (partition_id == 0) {
    ll128_send(subgroup, src, nbytes, ll128_buf, timeout, buffer_num_packets);
  } else {
    ll128_recv(subgroup, dst, nbytes, ll128_buf, timeout, buffer_num_packets);
  }
}

void launch_ll128_send_recv_undersized_buffer(int device) {
  PIPES_CUDA_CHECK(cudaSetDevice(device));

  constexpr size_t nbytes = 4096;
  constexpr size_t buffer_num_packets = 2;

  char* src_d;
  PIPES_CUDA_CHECK(cudaMalloc(&src_d, nbytes));
  PIPES_CUDA_CHECK(cudaMemset(src_d, 0x42, nbytes));

  char* dst_d;
  PIPES_CUDA_CHECK(cudaMalloc(&dst_d, nbytes));
  PIPES_CUDA_CHECK(cudaMemset(dst_d, 0, nbytes));

  size_t buf_size = buffer_num_packets * kLl128PacketSize;
  Ll128Packet* ll128_buf;
  PIPES_CUDA_CHECK(cudaMalloc(&ll128_buf, buf_size));
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  ll128_send_recv_undersized_buffer_kernel<<<2, 256>>>(
      src_d, dst_d, nbytes, ll128_buf, buffer_num_packets);

  cudaDeviceSynchronize();

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaFree(src_d);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaFree(dst_d);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaFree(ll128_buf);
}

} // namespace comms::pipes::test
