// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/ll128/Ll128Ops.cuh"
#include "comms/pipes/ll128/Ll128Packet.cuh"
#include "comms/pipes/ll128/tests/Ll128OpsNvlinkTest.cuh"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::test {

using namespace comms::pipes;

// =============================================================================
// Kernels — consolidated send/recv/forward for cross-GPU tests
// =============================================================================

__global__ void ll128_nvlink_send_kernel(
    const char* src,
    size_t nbytes,
    Ll128Packet* remote_ll128_buf,
    size_t buffer_num_packets,
    int num_steps) {
  auto group = make_warp_group();
  Timeout timeout;
  timeout.start();
  for (int i = 0; i < num_steps; i++) {
    ll128_send(
        group, src, nbytes, remote_ll128_buf, timeout, buffer_num_packets);
  }
}

__global__ void ll128_nvlink_recv_kernel(
    char* dst,
    size_t nbytes,
    Ll128Packet* local_ll128_buf,
    size_t buffer_num_packets,
    int num_steps) {
  auto group = make_warp_group();
  Timeout timeout;
  timeout.start();
  for (int i = 0; i < num_steps; i++) {
    ll128_recv(
        group, dst, nbytes, local_ll128_buf, timeout, buffer_num_packets);
  }
}

__global__ void ll128_nvlink_forward_kernel(
    char* fwd_dst,
    size_t nbytes,
    Ll128Packet* local_ll128_buf,
    Ll128Packet* remote_ll128_buf,
    size_t buffer_num_packets,
    int num_steps) {
  auto group = make_warp_group();
  Timeout timeout;
  timeout.start();
  for (int i = 0; i < num_steps; i++) {
    ll128_forward(
        group,
        fwd_dst,
        nbytes,
        local_ll128_buf,
        remote_ll128_buf,
        timeout,
        buffer_num_packets);
  }
}

// =============================================================================
// Combined send+recv kernel for forward tests.
// When sender and receiver share the same GPU, separate kernels on different
// streams may not be scheduled concurrently, causing deadlock when buffer
// reuse requires the receiver to ACK before the forwarder/sender can proceed.
// partition_interleaved(2) guarantees both roles execute concurrently.
// =============================================================================

__global__ void ll128_nvlink_send_recv_kernel(
    const char* src,
    char* dst,
    size_t nbytes,
    Ll128Packet* remote_send_buf,
    Ll128Packet* local_recv_buf,
    size_t buffer_num_packets,
    int num_steps) {
  auto group = make_warp_group();
  auto [partition_id, subgroup] = group.partition_interleaved(2);
  Timeout timeout;
  timeout.start();
  if (partition_id == 0) {
    for (int i = 0; i < num_steps; i++) {
      ll128_send(
          subgroup, src, nbytes, remote_send_buf, timeout, buffer_num_packets);
    }
  } else {
    for (int i = 0; i < num_steps; i++) {
      ll128_recv(
          subgroup, dst, nbytes, local_recv_buf, timeout, buffer_num_packets);
    }
  }
}

// =============================================================================
// Host-callable wrappers
// =============================================================================

void test_ll128_nvlink_send_recv(
    int sender_gpu,
    int receiver_gpu,
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    Ll128Packet* ll128_buf,
    size_t buffer_num_packets,
    int num_steps,
    int num_blocks,
    int block_size) {
  // Compute buffer size and init LL128 buffer on receiver GPU
  size_t buf_size = (buffer_num_packets > 0)
      ? buffer_num_packets * kLl128PacketSize
      : ll128_buffer_size(nbytes);
  PIPES_CUDA_CHECK(cudaSetDevice(receiver_gpu));
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  // Launch sender on sender_gpu
  cudaStream_t send_stream;
  PIPES_CUDA_CHECK(cudaSetDevice(sender_gpu));
  PIPES_CUDA_CHECK(cudaStreamCreate(&send_stream));
  ll128_nvlink_send_kernel<<<num_blocks, block_size, 0, send_stream>>>(
      src_d, nbytes, ll128_buf, buffer_num_packets, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Launch receiver on receiver_gpu
  cudaStream_t recv_stream;
  PIPES_CUDA_CHECK(cudaSetDevice(receiver_gpu));
  PIPES_CUDA_CHECK(cudaStreamCreate(&recv_stream));
  ll128_nvlink_recv_kernel<<<num_blocks, block_size, 0, recv_stream>>>(
      dst_d, nbytes, ll128_buf, buffer_num_packets, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Wait for both to complete
  PIPES_CUDA_CHECK(cudaSetDevice(sender_gpu));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(send_stream));
  PIPES_CUDA_CHECK(cudaStreamDestroy(send_stream));
  PIPES_CUDA_CHECK(cudaSetDevice(receiver_gpu));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(recv_stream));
  PIPES_CUDA_CHECK(cudaStreamDestroy(recv_stream));
}

void test_ll128_nvlink_forward(
    int sender_gpu,
    int forwarder_gpu,
    int receiver_gpu,
    const char* src_d,
    char* fwd_dst_d,
    char* recv_dst_d,
    size_t nbytes,
    Ll128Packet* ll128_buf_a,
    Ll128Packet* ll128_buf_b,
    int num_blocks,
    int block_size,
    size_t buffer_num_packets,
    int num_steps) {
  // The combined send+recv kernel requires sender and receiver on the same GPU.
  // All current forward tests use sender_gpu == receiver_gpu == 0.
  PIPES_CUDA_CHECK(
      sender_gpu == receiver_gpu ? cudaSuccess : cudaErrorInvalidValue);

  size_t buf_size = (buffer_num_packets > 0)
      ? buffer_num_packets * kLl128PacketSize
      : ll128_buffer_size(nbytes);

  // Init both LL128 buffers on their respective GPUs
  PIPES_CUDA_CHECK(cudaSetDevice(forwarder_gpu));
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf_a, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  PIPES_CUDA_CHECK(cudaSetDevice(receiver_gpu));
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf_b, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  // Combined sender+receiver on sender_gpu.
  // Uses partition_interleaved(2): needs 2x blocks so each role gets
  // num_blocks effective warps.
  cudaStream_t send_recv_stream;
  PIPES_CUDA_CHECK(cudaSetDevice(sender_gpu));
  PIPES_CUDA_CHECK(cudaStreamCreate(&send_recv_stream));
  ll128_nvlink_send_recv_kernel<<<
      2 * num_blocks,
      block_size,
      0,
      send_recv_stream>>>(
      src_d,
      recv_dst_d,
      nbytes,
      ll128_buf_a,
      ll128_buf_b,
      buffer_num_packets,
      num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Forwarder on forwarder_gpu: reads local buf_a, writes buf_b on
  // receiver_gpu via NVLink, copies to fwd_dst
  cudaStream_t fwd_stream;
  PIPES_CUDA_CHECK(cudaSetDevice(forwarder_gpu));
  PIPES_CUDA_CHECK(cudaStreamCreate(&fwd_stream));
  ll128_nvlink_forward_kernel<<<num_blocks, block_size, 0, fwd_stream>>>(
      fwd_dst_d,
      nbytes,
      ll128_buf_a,
      ll128_buf_b,
      buffer_num_packets,
      num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Wait for both to complete
  PIPES_CUDA_CHECK(cudaSetDevice(sender_gpu));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(send_recv_stream));
  PIPES_CUDA_CHECK(cudaStreamDestroy(send_recv_stream));
  PIPES_CUDA_CHECK(cudaSetDevice(forwarder_gpu));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(fwd_stream));
  PIPES_CUDA_CHECK(cudaStreamDestroy(fwd_stream));
}

void test_ll128_nvlink_bidirectional(
    const char* src0_d,
    char* dst0_d,
    const char* src1_d,
    char* dst1_d,
    size_t nbytes,
    Ll128Packet* ll128_buf_on_gpu0,
    Ll128Packet* ll128_buf_on_gpu1,
    int num_blocks,
    int block_size,
    size_t buffer_num_packets,
    int num_steps) {
  size_t buf_size = (buffer_num_packets > 0)
      ? buffer_num_packets * kLl128PacketSize
      : ll128_buffer_size(nbytes);

  // Init both buffers
  PIPES_CUDA_CHECK(cudaSetDevice(0));
  PIPES_CUDA_CHECK(
      cudaMemset(ll128_buf_on_gpu0, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());
  PIPES_CUDA_CHECK(cudaSetDevice(1));
  PIPES_CUDA_CHECK(
      cudaMemset(ll128_buf_on_gpu1, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  // Stream A (GPU0): send src0 → ll128_buf_on_gpu1
  cudaStream_t stream_a;
  PIPES_CUDA_CHECK(cudaSetDevice(0));
  PIPES_CUDA_CHECK(cudaStreamCreate(&stream_a));
  ll128_nvlink_send_kernel<<<num_blocks, block_size, 0, stream_a>>>(
      src0_d, nbytes, ll128_buf_on_gpu1, buffer_num_packets, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Stream B (GPU1): recv from ll128_buf_on_gpu1 → dst1
  cudaStream_t stream_b;
  PIPES_CUDA_CHECK(cudaSetDevice(1));
  PIPES_CUDA_CHECK(cudaStreamCreate(&stream_b));
  ll128_nvlink_recv_kernel<<<num_blocks, block_size, 0, stream_b>>>(
      dst1_d, nbytes, ll128_buf_on_gpu1, buffer_num_packets, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Stream C (GPU1): send src1 → ll128_buf_on_gpu0
  cudaStream_t stream_c;
  PIPES_CUDA_CHECK(cudaStreamCreate(&stream_c));
  ll128_nvlink_send_kernel<<<num_blocks, block_size, 0, stream_c>>>(
      src1_d, nbytes, ll128_buf_on_gpu0, buffer_num_packets, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Stream D (GPU0): recv from ll128_buf_on_gpu0 → dst0
  cudaStream_t stream_d;
  PIPES_CUDA_CHECK(cudaSetDevice(0));
  PIPES_CUDA_CHECK(cudaStreamCreate(&stream_d));
  ll128_nvlink_recv_kernel<<<num_blocks, block_size, 0, stream_d>>>(
      dst0_d, nbytes, ll128_buf_on_gpu0, buffer_num_packets, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Synchronize all streams
  PIPES_CUDA_CHECK(cudaSetDevice(0));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(stream_a));
  PIPES_CUDA_CHECK(cudaStreamDestroy(stream_a));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(stream_d));
  PIPES_CUDA_CHECK(cudaStreamDestroy(stream_d));
  PIPES_CUDA_CHECK(cudaSetDevice(1));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(stream_b));
  PIPES_CUDA_CHECK(cudaStreamDestroy(stream_b));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(stream_c));
  PIPES_CUDA_CHECK(cudaStreamDestroy(stream_c));
}

void test_ll128_nvlink_forward_3gpu(
    int sender_gpu,
    int forwarder_gpu,
    int receiver_gpu,
    const char* src_d,
    char* fwd_dst_d,
    char* recv_dst_d,
    size_t nbytes,
    Ll128Packet* ll128_buf_a,
    Ll128Packet* ll128_buf_b,
    int num_blocks,
    int block_size,
    size_t buffer_num_packets,
    int num_steps) {
  size_t buf_size = (buffer_num_packets > 0)
      ? buffer_num_packets * kLl128PacketSize
      : ll128_buffer_size(nbytes);

  // Init both LL128 buffers on their respective GPUs
  PIPES_CUDA_CHECK(cudaSetDevice(forwarder_gpu));
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf_a, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  PIPES_CUDA_CHECK(cudaSetDevice(receiver_gpu));
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf_b, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  // Sender on sender_gpu: writes to ll128_buf_a on forwarder_gpu
  cudaStream_t send_stream;
  PIPES_CUDA_CHECK(cudaSetDevice(sender_gpu));
  PIPES_CUDA_CHECK(cudaStreamCreate(&send_stream));
  ll128_nvlink_send_kernel<<<num_blocks, block_size, 0, send_stream>>>(
      src_d, nbytes, ll128_buf_a, buffer_num_packets, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Forwarder on forwarder_gpu: reads local buf_a, writes buf_b on
  // receiver_gpu, copies to fwd_dst
  cudaStream_t fwd_stream;
  PIPES_CUDA_CHECK(cudaSetDevice(forwarder_gpu));
  PIPES_CUDA_CHECK(cudaStreamCreate(&fwd_stream));
  ll128_nvlink_forward_kernel<<<num_blocks, block_size, 0, fwd_stream>>>(
      fwd_dst_d,
      nbytes,
      ll128_buf_a,
      ll128_buf_b,
      buffer_num_packets,
      num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Receiver on receiver_gpu: reads local buf_b to recv_dst
  cudaStream_t recv_stream;
  PIPES_CUDA_CHECK(cudaSetDevice(receiver_gpu));
  PIPES_CUDA_CHECK(cudaStreamCreate(&recv_stream));
  ll128_nvlink_recv_kernel<<<num_blocks, block_size, 0, recv_stream>>>(
      recv_dst_d, nbytes, ll128_buf_b, buffer_num_packets, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();

  // Wait for all to complete
  PIPES_CUDA_CHECK(cudaSetDevice(sender_gpu));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(send_stream));
  PIPES_CUDA_CHECK(cudaStreamDestroy(send_stream));
  PIPES_CUDA_CHECK(cudaSetDevice(forwarder_gpu));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(fwd_stream));
  PIPES_CUDA_CHECK(cudaStreamDestroy(fwd_stream));
  PIPES_CUDA_CHECK(cudaSetDevice(receiver_gpu));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(recv_stream));
  PIPES_CUDA_CHECK(cudaStreamDestroy(recv_stream));
}

// =============================================================================
// Varying-data NVLink kernels — each step offsets src/dst by i * nbytes
// =============================================================================

__global__ void ll128_nvlink_varying_send_kernel(
    const char* src,
    size_t nbytes,
    Ll128Packet* remote_ll128_buf,
    size_t buffer_num_packets,
    int num_steps) {
  auto group = make_warp_group();
  Timeout timeout;
  timeout.start();
  for (int i = 0; i < num_steps; i++) {
    ll128_send(
        group,
        src + i * nbytes,
        nbytes,
        remote_ll128_buf,
        timeout,
        buffer_num_packets);
  }
}

__global__ void ll128_nvlink_varying_recv_kernel(
    char* dst,
    size_t nbytes,
    Ll128Packet* local_ll128_buf,
    size_t buffer_num_packets,
    int num_steps) {
  auto group = make_warp_group();
  Timeout timeout;
  timeout.start();
  for (int i = 0; i < num_steps; i++) {
    ll128_recv(
        group,
        dst + i * nbytes,
        nbytes,
        local_ll128_buf,
        timeout,
        buffer_num_packets);
  }
}

// =============================================================================
// Varying-data NVLink host-callable wrapper
// =============================================================================

void test_ll128_nvlink_varying_send_recv(
    int sender_gpu,
    int receiver_gpu,
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
  PIPES_CUDA_CHECK(cudaSetDevice(receiver_gpu));
  PIPES_CUDA_CHECK(cudaMemset(ll128_buf, kLl128MemsetInitByte, buf_size));
  PIPES_CUDA_CHECK(cudaDeviceSynchronize());

  cudaStream_t send_stream;
  PIPES_CUDA_CHECK(cudaSetDevice(sender_gpu));
  PIPES_CUDA_CHECK(cudaStreamCreate(&send_stream));
  ll128_nvlink_varying_send_kernel<<<num_blocks, block_size, 0, send_stream>>>(
      src_d, nbytes, ll128_buf, buffer_num_packets, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();

  cudaStream_t recv_stream;
  PIPES_CUDA_CHECK(cudaSetDevice(receiver_gpu));
  PIPES_CUDA_CHECK(cudaStreamCreate(&recv_stream));
  ll128_nvlink_varying_recv_kernel<<<num_blocks, block_size, 0, recv_stream>>>(
      dst_d, nbytes, ll128_buf, buffer_num_packets, num_steps);
  PIPES_KERNEL_LAUNCH_CHECK();

  PIPES_CUDA_CHECK(cudaSetDevice(sender_gpu));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(send_stream));
  PIPES_CUDA_CHECK(cudaStreamDestroy(send_stream));
  PIPES_CUDA_CHECK(cudaSetDevice(receiver_gpu));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(recv_stream));
  PIPES_CUDA_CHECK(cudaStreamDestroy(recv_stream));
}

} // namespace comms::pipes::test
