// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Benchmark-only CUDA kernel: zero-copy RDMA put with BLOCK-scope ops.
// Not part of the library — used only by IbSendRecvBenchmark.

#include "comms/pipes/collectives/benchmarks/IbSendRecvBenchmarkKernels.cuh"

#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLX.cuh"

namespace comms::pipes::ib::benchmark {

using torchcomms::device::CmpOp;
using torchcomms::device::CoopScope;
using torchcomms::device::DeviceWindowNCCL;
using torchcomms::device::RegisteredBufferNCCL;
using torchcomms::device::SignalOp;

__global__ void put_bw_kernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    size_t total_bytes,
    int dst_rank,
    int src_rank,
    int signal_base,
    int iterations) {
  auto pid = blockIdx.x;
  auto num_blks = gridDim.x;
  size_t tile_bytes = total_bytes / num_blks;
  size_t tile_offset = pid * tile_bytes;

  int data_signal = pid;
  int ack_signal = num_blks + pid;

  for (int iter = 0; iter < iterations; iter++) {
    uint64_t expected = static_cast<uint64_t>(signal_base + iter + 1);

    // Put tile to remote, piggyback DATA_READY signal
    win->put(
        tile_offset,
        src_buf,
        tile_offset,
        dst_rank,
        tile_bytes,
        data_signal,
        -1,
        CoopScope::BLOCK);
    win->flush(CoopScope::BLOCK);

    // Wait for remote's data to arrive (their put to us)
    win->wait_signal_from(
        src_rank, data_signal, CmpOp::GE, expected, CoopScope::BLOCK);

    // ACK to remote sender: we received their data
    win->signal(src_rank, ack_signal, SignalOp::ADD, 1, CoopScope::BLOCK);

    // Wait for ACK from our receiver: they received our data
    win->wait_signal_from(
        dst_rank, ack_signal, CmpOp::GE, expected, CoopScope::BLOCK);
  }
}

void launch_put_bw_kernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    size_t total_bytes,
    int dst_rank,
    int src_rank,
    int num_blocks,
    int signal_base,
    int iterations,
    cudaStream_t stream) {
  put_bw_kernel<<<num_blocks, 256, 0, stream>>>(
      win, src_buf, total_bytes, dst_rank, src_rank, signal_base, iterations);
  cudaError_t err = cudaGetLastError();
  assert(err == cudaSuccess && "put_bw_kernel launch failed");
}

} // namespace comms::pipes::ib::benchmark
