// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/TimeoutUtils.h"
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
    Ll128Packet* remote_ll128_buf) {
  auto group = make_warp_group();
  Timeout timeout;
  timeout.start();
  ll128_forward(group, dst, nbytes, local_ll128_buf, remote_ll128_buf, timeout);
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
    int num_steps) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(2);

  Timeout timeout;
  timeout.start();

  if (partition_id == 0) {
    // Sender warps
    for (int i = 0; i < num_steps; i++) {
      ll128_send(subgroup, src, nbytes, ll128_buf, timeout);
    }
  } else {
    // Receiver warps
    for (int i = 0; i < num_steps; i++) {
      ll128_recv(subgroup, dst, nbytes, ll128_buf, timeout);
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
      src_d, dst_d, nbytes, ll128_buf, num_steps);
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
  // remote_ll128_buf should be initialized to READY_TO_WRITE for the
  // forward→recv chain. The local_ll128_buf is pre-populated by the caller.
  size_t remote_buf_size = ll128_buffer_size(nbytes);
  PIPES_CUDA_CHECK(
      cudaMemset(remote_ll128_buf, kLl128MemsetInitByte, remote_buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  ll128_forward_kernel<<<num_blocks, block_size>>>(
      dst_d, nbytes, local_ll128_buf, remote_ll128_buf);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

// =============================================================================
// Multi-step 3-role kernel — send → forward → recv in a single launch via
// partition_interleaved(3) for warp-level role assignment.
// Partition 0: senders, partition 1: forwarders, partition 2: receivers.
// The receiver is necessary because ll128_forward polls remote_ll128_buf for
// READY_TO_WRITE before each store. Without a receiver ACKing remote_ll128_buf,
// the forwarder deadlocks on step 2+.
// =============================================================================

__global__ void ll128_multi_step_send_forward_recv_kernel(
    const char* src,
    char* fwd_dst,
    char* recv_dst,
    size_t nbytes,
    Ll128Packet* ll128_buf_a,
    Ll128Packet* ll128_buf_b,
    int num_steps) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(3);
  Timeout timeout;
  timeout.start();

  if (partition_id == 0) {
    // Sender warps: write src → ll128_buf_a
    for (int i = 0; i < num_steps; i++) {
      ll128_send(subgroup, src, nbytes, ll128_buf_a, timeout);
    }
  } else if (partition_id == 1) {
    // Forwarder warps: read ll128_buf_a → ll128_buf_b + copy to fwd_dst
    for (int i = 0; i < num_steps; i++) {
      ll128_forward(
          subgroup, fwd_dst, nbytes, ll128_buf_a, ll128_buf_b, timeout);
    }
  } else {
    // Receiver warps: read ll128_buf_b → recv_dst (ACKs ll128_buf_b)
    for (int i = 0; i < num_steps; i++) {
      ll128_recv(subgroup, recv_dst, nbytes, ll128_buf_b, timeout);
    }
  }
}

// =============================================================================
// Host-callable wrappers (continued)
// =============================================================================

void test_ll128_multi_step_forward(
    const char* src_d,
    char* fwd_dst_d,
    char* recv_dst_d,
    size_t nbytes,
    Ll128Packet* ll128_buf_a,
    Ll128Packet* ll128_buf_b,
    int num_steps,
    int num_blocks,
    int block_size) {
  // Initialize both LL128 buffers to READY_TO_WRITE
  size_t buf_size = ll128_buffer_size(nbytes);
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf_a, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf_b, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  // Launch with 3 * num_blocks total blocks so each role gets
  // num_blocks * warps_per_block warps via partition_interleaved(3).
  int total_blocks = 3 * num_blocks;
  ll128_multi_step_send_forward_recv_kernel<<<total_blocks, block_size>>>(
      src_d,
      fwd_dst_d,
      recv_dst_d,
      nbytes,
      ll128_buf_a,
      ll128_buf_b,
      num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

// =============================================================================
// Chunked send/recv — combined kernel with buffer_num_packets
// =============================================================================

__global__ void ll128_chunked_combined_kernel(
    const char* src,
    char* dst,
    size_t nbytes,
    Ll128Packet* ll128_buf,
    size_t buffer_num_packets,
    int num_steps,
    Timeout timeout) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(2);

  timeout.start();

  if (partition_id == 0) {
    for (int i = 0; i < num_steps; i++) {
      ll128_send(subgroup, src, nbytes, ll128_buf, timeout, buffer_num_packets);
    }
  } else {
    for (int i = 0; i < num_steps; i++) {
      ll128_recv(subgroup, dst, nbytes, ll128_buf, timeout, buffer_num_packets);
    }
  }
}

// =============================================================================
// Chunked send→forward→recv — 3-role kernel with buffer_num_packets
// =============================================================================

__global__ void ll128_chunked_send_forward_recv_kernel(
    const char* src,
    char* fwd_dst,
    char* recv_dst,
    size_t nbytes,
    Ll128Packet* ll128_buf_a,
    Ll128Packet* ll128_buf_b,
    size_t buffer_num_packets,
    int num_steps) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(3);
  Timeout timeout;
  timeout.start();

  if (partition_id == 0) {
    for (int i = 0; i < num_steps; i++) {
      ll128_send(
          subgroup, src, nbytes, ll128_buf_a, timeout, buffer_num_packets);
    }
  } else if (partition_id == 1) {
    for (int i = 0; i < num_steps; i++) {
      ll128_forward(
          subgroup,
          fwd_dst,
          nbytes,
          ll128_buf_a,
          ll128_buf_b,
          timeout,
          buffer_num_packets);
    }
  } else {
    for (int i = 0; i < num_steps; i++) {
      ll128_recv(
          subgroup, recv_dst, nbytes, ll128_buf_b, timeout, buffer_num_packets);
    }
  }
}

// =============================================================================
// Chunked host-callable wrappers
// =============================================================================

void test_ll128_multi_step_send_recv_chunked(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    Ll128Packet* ll128_buf,
    size_t buffer_num_packets,
    int num_steps,
    int num_blocks,
    int block_size) {
  size_t buf_size = buffer_num_packets * kLl128PacketSize;
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  // 20s debug timeout — generous upper bound for a test completing in <100ms.
  // On timeout, TIMEOUT_TRAP_IF_EXPIRED_SINGLE in ll128_send/ll128_recv prints
  // which side is stuck, which packet, which buf_idx, and the current flag
  // value.
  auto timeout = makeTimeout(20000);

  int total_blocks = 2 * num_blocks;
  ll128_chunked_combined_kernel<<<total_blocks, block_size>>>(
      src_d, dst_d, nbytes, ll128_buf, buffer_num_packets, num_steps, timeout);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

void test_ll128_send_recv_chunked(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    Ll128Packet* ll128_buf,
    size_t buffer_num_packets,
    int num_blocks,
    int block_size) {
  test_ll128_multi_step_send_recv_chunked(
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
      buffer_num_packets,
      /*num_steps=*/1,
      num_blocks,
      block_size);
}

void test_ll128_multi_step_forward_chunked(
    const char* src_d,
    char* fwd_dst_d,
    char* recv_dst_d,
    size_t nbytes,
    Ll128Packet* ll128_buf_a,
    Ll128Packet* ll128_buf_b,
    size_t buffer_num_packets,
    int num_steps,
    int num_blocks,
    int block_size) {
  size_t buf_size = buffer_num_packets * kLl128PacketSize;
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf_a, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf_b, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  int total_blocks = 3 * num_blocks;
  ll128_chunked_send_forward_recv_kernel<<<total_blocks, block_size>>>(
      src_d,
      fwd_dst_d,
      recv_dst_d,
      nbytes,
      ll128_buf_a,
      ll128_buf_b,
      buffer_num_packets,
      num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

// From D95387114

// =============================================================================
// Windowed send/recv kernel — buffer smaller than message, uses modular
// indexing via max_ll128_packets parameter.
// =============================================================================

__global__ void ll128_windowed_combined_kernel(
    const char* src,
    char* dst,
    size_t nbytes,
    Ll128Packet* ll128_buf,
    size_t max_ll128_packets) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(2);

  Timeout timeout;
  timeout.start();

  if (partition_id == 0) {
    ll128_send(subgroup, src, nbytes, ll128_buf, timeout, max_ll128_packets);
  } else {
    ll128_recv(subgroup, dst, nbytes, ll128_buf, timeout, max_ll128_packets);
  }
}

void test_ll128_windowed_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    Ll128Packet* ll128_buf,
    size_t max_ll128_packets,
    int num_blocks,
    int block_size) {
  // Initialize the windowed buffer to READY_TO_WRITE
  size_t buf_size = max_ll128_packets * kLl128PacketSize;
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  // Launch combined kernel: 2 * num_blocks so each role gets num_blocks warps
  int total_blocks = 2 * num_blocks;
  ll128_windowed_combined_kernel<<<total_blocks, block_size>>>(
      src_d, dst_d, nbytes, ll128_buf, max_ll128_packets);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

// =============================================================================
// Varying-data multi-step kernels — each step uses a different src/dst offset
// to detect stale buffer contents leaking across steps.
// =============================================================================

__global__ void ll128_varying_data_multi_step_combined_kernel(
    const char* src,
    char* dst,
    size_t nbytes,
    Ll128Packet* ll128_buf,
    size_t buffer_num_packets,
    int num_steps) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(2);

  Timeout timeout;
  timeout.start();

  if (partition_id == 0) {
    for (int i = 0; i < num_steps; i++) {
      ll128_send(
          subgroup,
          src + i * nbytes,
          nbytes,
          ll128_buf,
          timeout,
          buffer_num_packets);
    }
  } else {
    for (int i = 0; i < num_steps; i++) {
      ll128_recv(
          subgroup,
          dst + i * nbytes,
          nbytes,
          ll128_buf,
          timeout,
          buffer_num_packets);
    }
  }
}

__global__ void ll128_varying_data_multi_step_send_forward_recv_kernel(
    const char* src,
    char* fwd_dst,
    char* recv_dst,
    size_t nbytes,
    Ll128Packet* ll128_buf_a,
    Ll128Packet* ll128_buf_b,
    size_t buffer_num_packets,
    int num_steps) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(3);
  Timeout timeout;
  timeout.start();

  if (partition_id == 0) {
    for (int i = 0; i < num_steps; i++) {
      ll128_send(
          subgroup,
          src + i * nbytes,
          nbytes,
          ll128_buf_a,
          timeout,
          buffer_num_packets);
    }
  } else if (partition_id == 1) {
    for (int i = 0; i < num_steps; i++) {
      ll128_forward(
          subgroup,
          fwd_dst + i * nbytes,
          nbytes,
          ll128_buf_a,
          ll128_buf_b,
          timeout,
          buffer_num_packets);
    }
  } else {
    for (int i = 0; i < num_steps; i++) {
      ll128_recv(
          subgroup,
          recv_dst + i * nbytes,
          nbytes,
          ll128_buf_b,
          timeout,
          buffer_num_packets);
    }
  }
}

// =============================================================================
// Varying-data host-callable wrappers
// =============================================================================

void test_ll128_varying_data_multi_step_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    Ll128Packet* ll128_buf,
    size_t buffer_num_packets,
    int num_steps,
    int num_blocks,
    int block_size) {
  size_t buf_size = (buffer_num_packets > 0)
      ? buffer_num_packets * kLl128PacketSize
      : ll128_buffer_size(nbytes);
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  int total_blocks = 2 * num_blocks;
  ll128_varying_data_multi_step_combined_kernel<<<total_blocks, block_size>>>(
      src_d, dst_d, nbytes, ll128_buf, buffer_num_packets, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

void test_ll128_varying_data_multi_step_forward(
    const char* src_d,
    char* fwd_dst_d,
    char* recv_dst_d,
    size_t nbytes,
    Ll128Packet* ll128_buf_a,
    Ll128Packet* ll128_buf_b,
    size_t buffer_num_packets,
    int num_steps,
    int num_blocks,
    int block_size) {
  size_t buf_size = (buffer_num_packets > 0)
      ? buffer_num_packets * kLl128PacketSize
      : ll128_buffer_size(nbytes);
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf_a, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf_b, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  int total_blocks = 3 * num_blocks;
  ll128_varying_data_multi_step_send_forward_recv_kernel<<<
      total_blocks,
      block_size>>>(
      src_d,
      fwd_dst_d,
      recv_dst_d,
      nbytes,
      ll128_buf_a,
      ll128_buf_b,
      buffer_num_packets,
      num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace comms::pipes::test
