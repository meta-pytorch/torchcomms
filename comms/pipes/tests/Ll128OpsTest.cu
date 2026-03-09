// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/ll128/Ll128Ops.cuh"
#include "comms/pipes/ll128/Ll128Packet.cuh"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::test {

using namespace comms::pipes;

// =============================================================================
// Forward kernel — reads from local LL128, forwards to remote, copies to dst
// =============================================================================

__global__ void ll128_forward_kernel(
    char* dst,
    size_t nbytes,
    Ll128Packet* local_ll128_buf,
    Ll128Packet* remote_ll128_buf,
    int64_t flag_value) {
  auto group = make_warp_group();
  Timeout timeout;
  timeout.start();
  ll128_forward(
      group,
      dst,
      nbytes,
      local_ll128_buf,
      remote_ll128_buf,
      flag_value,
      timeout);
}

// =============================================================================
// Multi-step combined kernel — send and recv in a single launch via
// partition_interleaved(2) for warp-level role assignment.
// Even-indexed warps are senders, odd-indexed warps are receivers.
// This avoids the deadlock that occurs when two separate kernels on different
// streams are serialized by the GPU scheduler.
// =============================================================================

__global__ void ll128_multi_step_combined_kernel(
    const char* src,
    char* dst,
    size_t nbytes,
    Ll128Packet* ll128_buf,
    int64_t start_step_id,
    int num_steps) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(2);

  Timeout timeout;
  timeout.start();

  if (partition_id == 0) {
    // Sender warps
    for (int i = 0; i < num_steps; i++) {
      ll128_send(subgroup, src, nbytes, ll128_buf, start_step_id + i, timeout);
    }
  } else {
    // Receiver warps
    for (int i = 0; i < num_steps; i++) {
      ll128_recv(subgroup, dst, nbytes, ll128_buf, start_step_id + i, timeout);
    }
  }
}

// =============================================================================
// Host-callable wrappers
// =============================================================================

void test_ll128_multi_step_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    Ll128Packet* ll128_buf,
    int64_t start_step_id,
    int num_steps,
    int num_blocks,
    int block_size) {
  // Initialize LL128 buffer flags to READY_TO_WRITE
  size_t buf_size = ll128_buffer_size(nbytes);
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  // Launch combined kernel: 2 * num_blocks total blocks so each role
  // (sender/receiver) gets num_blocks * warps_per_block warps via
  // partition_interleaved(2).
  int total_blocks = 2 * num_blocks;
  ll128_multi_step_combined_kernel<<<total_blocks, block_size>>>(
      src_d, dst_d, nbytes, ll128_buf, start_step_id, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

void test_ll128_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    Ll128Packet* ll128_buf,
    int num_blocks,
    int block_size) {
  // Delegate to combined kernel to avoid SM-hogging deadlock
  // when send/recv are launched as separate kernels.
  test_ll128_multi_step_send_recv(
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
      /*start_step_id=*/1,
      /*num_steps=*/1,
      num_blocks,
      block_size);
}

void test_ll128_forward(
    char* dst_d,
    size_t nbytes,
    Ll128Packet* local_ll128_buf,
    Ll128Packet* remote_ll128_buf,
    int num_blocks,
    int block_size) {
  constexpr int64_t flag_value = 1;

  // remote_ll128_buf should be initialized to READY_TO_WRITE for the
  // forward→recv chain. The local_ll128_buf is pre-populated by the caller.
  size_t remote_buf_size = ll128_buffer_size(nbytes);
  PIPES_CUDA_CHECK(
      cudaMemset(remote_ll128_buf, kLl128MemsetInitByte, remote_buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  ll128_forward_kernel<<<num_blocks, block_size>>>(
      dst_d, nbytes, local_ll128_buf, remote_ll128_buf, flag_value);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace comms::pipes::test
