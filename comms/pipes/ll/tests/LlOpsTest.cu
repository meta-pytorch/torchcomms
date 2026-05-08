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

} // namespace comms::pipes::test
