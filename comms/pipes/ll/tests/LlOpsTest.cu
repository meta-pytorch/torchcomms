// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/TimeoutUtils.h"
#include "comms/pipes/ll/LlOps.cuh"
#include "comms/pipes/ll/LlPacket.cuh"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::test {

using namespace comms::pipes;

// =============================================================================
// Forward kernel
// =============================================================================

__global__ void ll_forward_kernel(
    char* dst,
    size_t nbytes,
    LlLine* local_ll_buf,
    LlLine* remote_ll_buf) {
  auto group = make_warp_group();
  Timeout timeout;
  timeout.start();
  ll_forward(group, dst, nbytes, local_ll_buf, remote_ll_buf, timeout);
}

// =============================================================================
// Combined send/recv kernel via partition_interleaved(2)
// =============================================================================

__global__ void ll_multi_step_combined_kernel(
    const char* src,
    char* dst,
    size_t nbytes,
    LlLine* ll_buf,
    int num_steps) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(2);

  Timeout timeout;
  timeout.start();

  if (partition_id == 0) {
    for (int i = 0; i < num_steps; i++) {
      ll_send(subgroup, src, nbytes, ll_buf, timeout);
    }
  } else {
    for (int i = 0; i < num_steps; i++) {
      ll_recv(subgroup, dst, nbytes, ll_buf, timeout);
    }
  }
}

// =============================================================================
// 3-role kernel: send → forward → recv
// =============================================================================

__global__ void ll_multi_step_send_forward_recv_kernel(
    const char* src,
    char* fwd_dst,
    char* recv_dst,
    size_t nbytes,
    LlLine* ll_buf_a,
    LlLine* ll_buf_b,
    int num_steps) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(3);
  Timeout timeout;
  timeout.start();

  if (partition_id == 0) {
    for (int i = 0; i < num_steps; i++) {
      ll_send(subgroup, src, nbytes, ll_buf_a, timeout);
    }
  } else if (partition_id == 1) {
    for (int i = 0; i < num_steps; i++) {
      ll_forward(subgroup, fwd_dst, nbytes, ll_buf_a, ll_buf_b, timeout);
    }
  } else {
    for (int i = 0; i < num_steps; i++) {
      ll_recv(subgroup, recv_dst, nbytes, ll_buf_b, timeout);
    }
  }
}

// =============================================================================
// Chunked combined kernel with buffer_num_lines
// =============================================================================

__global__ void ll_chunked_combined_kernel(
    const char* src,
    char* dst,
    size_t nbytes,
    LlLine* ll_buf,
    size_t buffer_num_lines,
    int num_steps,
    Timeout timeout) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(2);

  timeout.start();

  if (partition_id == 0) {
    for (int i = 0; i < num_steps; i++) {
      ll_send(subgroup, src, nbytes, ll_buf, timeout, buffer_num_lines);
    }
  } else {
    for (int i = 0; i < num_steps; i++) {
      ll_recv(subgroup, dst, nbytes, ll_buf, timeout, buffer_num_lines);
    }
  }
}

// =============================================================================
// Varying-data multi-step combined kernel
// =============================================================================

__global__ void ll_varying_data_multi_step_combined_kernel(
    const char* src,
    char* dst,
    size_t nbytes,
    LlLine* ll_buf,
    size_t buffer_num_lines,
    int num_steps) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(2);

  Timeout timeout;
  timeout.start();

  if (partition_id == 0) {
    for (int i = 0; i < num_steps; i++) {
      ll_send(
          subgroup,
          src + i * nbytes,
          nbytes,
          ll_buf,
          timeout,
          buffer_num_lines);
    }
  } else {
    for (int i = 0; i < num_steps; i++) {
      ll_recv(
          subgroup,
          dst + i * nbytes,
          nbytes,
          ll_buf,
          timeout,
          buffer_num_lines);
    }
  }
}

// =============================================================================
// Varying-data multi-step 3-role kernel
// =============================================================================

__global__ void ll_varying_data_multi_step_send_forward_recv_kernel(
    const char* src,
    char* fwd_dst,
    char* recv_dst,
    size_t nbytes,
    LlLine* ll_buf_a,
    LlLine* ll_buf_b,
    size_t buffer_num_lines,
    int num_steps) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(3);
  Timeout timeout;
  timeout.start();

  if (partition_id == 0) {
    for (int i = 0; i < num_steps; i++) {
      ll_send(
          subgroup,
          src + i * nbytes,
          nbytes,
          ll_buf_a,
          timeout,
          buffer_num_lines);
    }
  } else if (partition_id == 1) {
    for (int i = 0; i < num_steps; i++) {
      ll_forward(
          subgroup,
          fwd_dst + i * nbytes,
          nbytes,
          ll_buf_a,
          ll_buf_b,
          timeout,
          buffer_num_lines);
    }
  } else {
    for (int i = 0; i < num_steps; i++) {
      ll_recv(
          subgroup,
          recv_dst + i * nbytes,
          nbytes,
          ll_buf_b,
          timeout,
          buffer_num_lines);
    }
  }
}

// =============================================================================
// Host-callable wrappers
// =============================================================================

void test_ll_multi_step_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    LlLine* ll_buf,
    int num_steps,
    int num_blocks,
    int block_size) {
  size_t buf_size = ll_buffer_size(nbytes);
  PIPES_CUDA_CHECK(cudaMemset(ll_buf, kLlMemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  int total_blocks = 2 * num_blocks;
  ll_multi_step_combined_kernel<<<total_blocks, block_size>>>(
      src_d, dst_d, nbytes, ll_buf, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

void test_ll_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    LlLine* ll_buf,
    int num_blocks,
    int block_size) {
  test_ll_multi_step_send_recv(
      src_d, dst_d, nbytes, ll_buf, /*num_steps=*/1, num_blocks, block_size);
}

void test_ll_forward(
    char* dst_d,
    size_t nbytes,
    LlLine* local_ll_buf,
    LlLine* remote_ll_buf,
    int num_blocks,
    int block_size) {
  size_t remote_buf_size = ll_buffer_size(nbytes);
  PIPES_CUDA_CHECK(
      cudaMemset(remote_ll_buf, kLlMemsetInitByte, remote_buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  ll_forward_kernel<<<num_blocks, block_size>>>(
      dst_d, nbytes, local_ll_buf, remote_ll_buf);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

void test_ll_multi_step_forward(
    const char* src_d,
    char* fwd_dst_d,
    char* recv_dst_d,
    size_t nbytes,
    LlLine* ll_buf_a,
    LlLine* ll_buf_b,
    int num_steps,
    int num_blocks,
    int block_size) {
  size_t buf_size = ll_buffer_size(nbytes);
  PIPES_CUDA_CHECK(cudaMemset(ll_buf_a, kLlMemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaMemset(ll_buf_b, kLlMemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  int total_blocks = 3 * num_blocks;
  ll_multi_step_send_forward_recv_kernel<<<total_blocks, block_size>>>(
      src_d, fwd_dst_d, recv_dst_d, nbytes, ll_buf_a, ll_buf_b, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

void test_ll_multi_step_send_recv_chunked(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    LlLine* ll_buf,
    size_t buffer_num_lines,
    int num_steps,
    int num_blocks,
    int block_size) {
  size_t buf_size = buffer_num_lines * kLlLineSize;
  PIPES_CUDA_CHECK(cudaMemset(ll_buf, kLlMemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  auto timeout = makeTimeout(20000);

  int total_blocks = 2 * num_blocks;
  ll_chunked_combined_kernel<<<total_blocks, block_size>>>(
      src_d, dst_d, nbytes, ll_buf, buffer_num_lines, num_steps, timeout);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

void test_ll_send_recv_chunked(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    LlLine* ll_buf,
    size_t buffer_num_lines,
    int num_blocks,
    int block_size) {
  test_ll_multi_step_send_recv_chunked(
      src_d,
      dst_d,
      nbytes,
      ll_buf,
      buffer_num_lines,
      /*num_steps=*/1,
      num_blocks,
      block_size);
}

void test_ll_varying_data_multi_step_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    LlLine* ll_buf,
    size_t buffer_num_lines,
    int num_steps,
    int num_blocks,
    int block_size) {
  size_t buf_size = (buffer_num_lines > 0) ? buffer_num_lines * kLlLineSize
                                           : ll_buffer_size(nbytes);
  PIPES_CUDA_CHECK(cudaMemset(ll_buf, kLlMemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  int total_blocks = 2 * num_blocks;
  ll_varying_data_multi_step_combined_kernel<<<total_blocks, block_size>>>(
      src_d, dst_d, nbytes, ll_buf, buffer_num_lines, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

void test_ll_varying_data_multi_step_forward(
    const char* src_d,
    char* fwd_dst_d,
    char* recv_dst_d,
    size_t nbytes,
    LlLine* ll_buf_a,
    LlLine* ll_buf_b,
    size_t buffer_num_lines,
    int num_steps,
    int num_blocks,
    int block_size) {
  size_t buf_size = (buffer_num_lines > 0) ? buffer_num_lines * kLlLineSize
                                           : ll_buffer_size(nbytes);
  PIPES_CUDA_CHECK(cudaMemset(ll_buf_a, kLlMemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaMemset(ll_buf_b, kLlMemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  int total_blocks = 3 * num_blocks;
  ll_varying_data_multi_step_send_forward_recv_kernel<<<
      total_blocks,
      block_size>>>(
      src_d,
      fwd_dst_d,
      recv_dst_d,
      nbytes,
      ll_buf_a,
      ll_buf_b,
      buffer_num_lines,
      num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace comms::pipes::test
