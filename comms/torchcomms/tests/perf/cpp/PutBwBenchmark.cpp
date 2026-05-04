// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Put BW microbenchmark: measures raw RDMA put bandwidth using the
// TorchComm device API (NCCLx/GIN backend).
//
// Two tests:
//   SingleBlock — one block puts the full buffer, sweep over sizes.
//   MultiBlock  — N blocks each put total/N bytes, sweep over block counts.
//
// Zero-copy (no staging buffer). BLOCK-scope cooperative ops.
// Receiver ACKs each iteration so sender knows the put landed.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/MemPool.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "PutBwBenchmarkKernels.cuh"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/device/TorchCommDeviceWindow.hpp"
#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

using namespace torchcomms::device;
using namespace torchcomms::device::test;

namespace {

constexpr int kWarmupIters = 20;
constexpr int kMeasureIters = 100;
constexpr size_t KB = 1024;
constexpr size_t MB = 1024 * 1024;

std::string formatSize(size_t bytes) {
  if (bytes >= MB) {
    return std::to_string(bytes / MB) + "MB";
  }
  if (bytes >= KB) {
    return std::to_string(bytes / KB) + "KB";
  }
  return std::to_string(bytes) + "B";
}

struct WindowSetup {
  std::unique_ptr<at::cuda::MemPool> mem_pool;
  at::Tensor win_tensor;
  at::Tensor src_tensor;
  std::shared_ptr<torch::comms::TorchCommWindow> win;
  DeviceWindowNCCL* dev_win{nullptr};
  RegisteredBufferNCCL src_buf{};
};

WindowSetup createWindowSetup(
    std::shared_ptr<torch::comms::TorchComm>& torchcomm,
    std::shared_ptr<c10::Allocator>& allocator,
    int device_index,
    size_t total_bytes,
    int signal_count) {
  WindowSetup s;
  size_t count = total_bytes / sizeof(float);

  s.mem_pool = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          allocator));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      s.mem_pool->device(), s.mem_pool->id(), [](cudaStream_t) {
        return true;
      });

  auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index);
  s.win_tensor = at::zeros({static_cast<int64_t>(count)}, options);

  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      s.mem_pool->device(), s.mem_pool->id());

  // Allocate src outside pool for proper alignment (see DeviceApiTest comment).
  s.src_tensor = at::zeros({static_cast<int64_t>(count)}, options);

  torchcomm->barrier(false);
  s.win = torchcomm->new_window();
  s.win->tensor_register(s.win_tensor);
  torchcomm->barrier(false);

  s.dev_win = static_cast<DeviceWindowNCCL*>(
      s.win->get_device_window(signal_count, -1, 2));
  s.src_buf = s.win->register_local_buffer(s.src_tensor);

  torchcomm->barrier(false);
  cudaDeviceSynchronize();

  return s;
}

void teardownWindow(
    WindowSetup& s,
    std::shared_ptr<torch::comms::TorchComm>& torchcomm) {
  s.win->deregister_local_buffer(s.src_buf);
  s.win->tensor_deregister();
  s.win.reset();
  s.mem_pool.reset();
  torchcomm->barrier(false);
}

} // namespace

class PutBwBenchmark : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* env = std::getenv("RUN_PUT_BW_BENCHMARK");
    if (!env || (std::string(env) != "1" && std::string(env) != "true")) {
      GTEST_SKIP() << "Set RUN_PUT_BW_BENCHMARK=true to run";
    }

    wrapper_ = std::make_unique<TorchCommTestWrapper>();
    torchcomm_ = wrapper_->getTorchComm();
    rank_ = torchcomm_->getRank();
    num_ranks_ = torchcomm_->getSize();
    device_index_ = rank_ % at::cuda::device_count();
    allocator_ = torch::comms::get_mem_allocator(torchcomm_->getBackend());
  }

  void TearDown() override {
    torchcomm_.reset();
    wrapper_.reset();
  }

  struct BenchResult {
    size_t total_bytes;
    int num_blocks;
    int iterations;
    float elapsed_ms;
    double bw_gbps;
  };

  BenchResult runBenchmark(size_t total_bytes, int num_blocks) {
    int signal_count = 2 * num_blocks;
    auto s = createWindowSetup(
        torchcomm_, allocator_, device_index_, total_bytes, signal_count);

    int dst_rank = (rank_ + 1) % num_ranks_;
    int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

    auto stream = at::cuda::getStreamFromPool(false, device_index_);

    // Warmup
    {
      c10::cuda::CUDAStreamGuard guard(stream);
      launchPutBwKernel(
          s.dev_win,
          s.src_buf,
          total_bytes,
          dst_rank,
          src_rank,
          num_blocks,
          0,
          kWarmupIters,
          stream.stream());
    }
    stream.synchronize();
    torchcomm_->barrier(false);

    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    {
      c10::cuda::CUDAStreamGuard guard(stream);
      cudaEventRecord(start, stream.stream());
      launchPutBwKernel(
          s.dev_win,
          s.src_buf,
          total_bytes,
          dst_rank,
          src_rank,
          num_blocks,
          kWarmupIters,
          kMeasureIters,
          stream.stream());
      cudaEventRecord(stop, stream.stream());
    }
    cudaEventSynchronize(stop);

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    double total_data = static_cast<double>(total_bytes) * kMeasureIters;
    double bw = total_data / (elapsed_ms / 1000.0) / 1e9;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    teardownWindow(s, torchcomm_);

    return {total_bytes, num_blocks, kMeasureIters, elapsed_ms, bw};
  }

  BenchResult runSendRecvBenchmark(size_t total_bytes, int num_blocks) {
    int signal_count = 2 * num_blocks;
    int counter_count = num_blocks;
    size_t count = total_bytes / sizeof(float);

    // MemPool for window tensor
    auto mem_pool = std::make_unique<at::cuda::MemPool>(
        std::static_pointer_cast<
            c10::cuda::CUDACachingAllocator::CUDAAllocator>(allocator_));
    c10::cuda::CUDACachingAllocator::beginAllocateToPool(
        mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

    auto options =
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index_);
    auto win_tensor = at::zeros({static_cast<int64_t>(count)}, options);

    c10::cuda::CUDACachingAllocator::endAllocateToPool(
        mem_pool->device(), mem_pool->id());

    // Allocate src, staging, dst outside pool
    auto src_tensor = at::zeros({static_cast<int64_t>(count)}, options);
    auto staging_tensor = at::zeros({static_cast<int64_t>(count)}, options);
    auto dst_tensor = at::zeros({static_cast<int64_t>(count)}, options);

    torchcomm_->barrier(false);
    auto win = torchcomm_->new_window();
    win->tensor_register(win_tensor);
    torchcomm_->barrier(false);

    auto dev_win = static_cast<DeviceWindowNCCL*>(
        win->get_device_window(signal_count, counter_count, 2));
    auto staging_buf = win->register_local_buffer(staging_tensor);

    torchcomm_->barrier(false);
    cudaDeviceSynchronize();

    int dst_rank = (rank_ + 1) % num_ranks_;
    int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

    auto stream = at::cuda::getStreamFromPool(false, device_index_);

    float* src_ptr = src_tensor.data_ptr<float>();
    float* staging_ptr = staging_tensor.data_ptr<float>();
    float* win_ptr = win_tensor.data_ptr<float>();
    float* dst_ptr = dst_tensor.data_ptr<float>();

    // Warmup
    {
      c10::cuda::CUDAStreamGuard guard(stream);
      launchSendRecvBwKernel(
          dev_win,
          staging_buf,
          src_ptr,
          staging_ptr,
          win_ptr,
          dst_ptr,
          total_bytes,
          dst_rank,
          src_rank,
          num_blocks,
          0,
          kWarmupIters,
          stream.stream());
    }
    stream.synchronize();
    torchcomm_->barrier(false);

    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    {
      c10::cuda::CUDAStreamGuard guard(stream);
      cudaEventRecord(start, stream.stream());
      launchSendRecvBwKernel(
          dev_win,
          staging_buf,
          src_ptr,
          staging_ptr,
          win_ptr,
          dst_ptr,
          total_bytes,
          dst_rank,
          src_rank,
          num_blocks,
          kWarmupIters,
          kMeasureIters,
          stream.stream());
      cudaEventRecord(stop, stream.stream());
    }
    cudaEventSynchronize(stop);

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    double total_data = static_cast<double>(total_bytes) * kMeasureIters;
    double bw = total_data / (elapsed_ms / 1000.0) / 1e9;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Teardown
    win->deregister_local_buffer(staging_buf);
    win->tensor_deregister();
    win.reset();
    mem_pool.reset();
    torchcomm_->barrier(false);

    return {total_bytes, num_blocks, kMeasureIters, elapsed_ms, bw};
  }

  BenchResult runSendRecvPipelinedBenchmark(
      size_t total_bytes,
      size_t section_bytes,
      int pipeline_depth,
      int num_blocks) {
    int signal_count = 2 * num_blocks;
    int counter_count = num_blocks;
    int total_steps = total_bytes / section_bytes;

    // Staging and window are ring buffers: pipeline_depth * section_bytes
    size_t ring_bytes = pipeline_depth * section_bytes;
    size_t ring_count = ring_bytes / sizeof(float);
    size_t total_count = total_bytes / sizeof(float);

    // MemPool for window tensor (recv staging ring buffer)
    auto mem_pool = std::make_unique<at::cuda::MemPool>(
        std::static_pointer_cast<
            c10::cuda::CUDACachingAllocator::CUDAAllocator>(allocator_));
    c10::cuda::CUDACachingAllocator::beginAllocateToPool(
        mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

    auto options =
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index_);
    auto win_tensor = at::zeros({static_cast<int64_t>(ring_count)}, options);

    c10::cuda::CUDACachingAllocator::endAllocateToPool(
        mem_pool->device(), mem_pool->id());

    // Allocate src, staging, dst outside pool
    auto src_tensor = at::zeros({static_cast<int64_t>(total_count)}, options);
    auto staging_tensor =
        at::zeros({static_cast<int64_t>(ring_count)}, options);
    auto dst_tensor = at::zeros({static_cast<int64_t>(total_count)}, options);

    torchcomm_->barrier(false);
    auto win = torchcomm_->new_window();
    win->tensor_register(win_tensor);
    torchcomm_->barrier(false);

    auto dev_win = static_cast<DeviceWindowNCCL*>(
        win->get_device_window(signal_count, counter_count, 2));
    auto staging_buf = win->register_local_buffer(staging_tensor);

    torchcomm_->barrier(false);
    cudaDeviceSynchronize();

    int dst_rank = (rank_ + 1) % num_ranks_;
    int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

    auto stream = at::cuda::getStreamFromPool(false, device_index_);

    float* src_ptr = src_tensor.data_ptr<float>();
    float* staging_ptr = staging_tensor.data_ptr<float>();
    float* win_ptr = win_tensor.data_ptr<float>();
    float* dst_ptr = dst_tensor.data_ptr<float>();

    // Warmup
    {
      c10::cuda::CUDAStreamGuard guard(stream);
      launchSendRecvPipelinedBwKernel(
          dev_win,
          staging_buf,
          src_ptr,
          staging_ptr,
          win_ptr,
          dst_ptr,
          total_bytes,
          section_bytes,
          pipeline_depth,
          dst_rank,
          src_rank,
          num_blocks,
          0,
          kWarmupIters,
          stream.stream());
    }
    stream.synchronize();
    torchcomm_->barrier(false);

    // Timed run
    int measure_signal_base = kWarmupIters * total_steps;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    {
      c10::cuda::CUDAStreamGuard guard(stream);
      cudaEventRecord(start, stream.stream());
      launchSendRecvPipelinedBwKernel(
          dev_win,
          staging_buf,
          src_ptr,
          staging_ptr,
          win_ptr,
          dst_ptr,
          total_bytes,
          section_bytes,
          pipeline_depth,
          dst_rank,
          src_rank,
          num_blocks,
          measure_signal_base,
          kMeasureIters,
          stream.stream());
      cudaEventRecord(stop, stream.stream());
    }
    cudaEventSynchronize(stop);

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    double total_data = static_cast<double>(total_bytes) * kMeasureIters;
    double bw = total_data / (elapsed_ms / 1000.0) / 1e9;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Teardown
    win->deregister_local_buffer(staging_buf);
    win->tensor_deregister();
    win.reset();
    mem_pool.reset();
    torchcomm_->barrier(false);

    return {total_bytes, num_blocks, kMeasureIters, elapsed_ms, bw};
  }

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  std::shared_ptr<c10::Allocator> allocator_;
  int rank_{0};
  int num_ranks_{0};
  int device_index_{0};
};

TEST_F(PutBwBenchmark, SingleBlock) {
  std::vector<size_t> sizes = {
      256 * KB, 1 * MB, 4 * MB, 16 * MB, 64 * MB, 128 * MB, 512 * MB};

  if (rank_ == 0) {
    printf("\n=== Put BW: Single Block (BLOCK scope, zero-copy) ===\n");
    printf(
        "%-12s  %5s  %10s  %10s\n", "PutSize", "Iters", "Lat(us)", "BW(GB/s)");
    printf(
        "%-12s  %5s  %10s  %10s\n", "-------", "-----", "-------", "--------");
  }

  for (size_t sz : sizes) {
    auto r = runBenchmark(sz, 1);
    if (rank_ == 0) {
      double lat_us = r.elapsed_ms * 1000.0 / r.iterations;
      printf(
          "%-12s  %5d  %10.1f  %10.2f\n",
          formatSize(r.total_bytes).c_str(),
          r.iterations,
          lat_us,
          r.bw_gbps);
    }
  }
}

TEST_F(PutBwBenchmark, MultiBlock) {
  size_t total = 512 * MB;
  std::vector<int> block_counts = {1, 2, 4, 8, 16, 32, 64, 128};

  if (rank_ == 0) {
    printf(
        "\n=== Put BW: Multi Block (512MB total, BLOCK scope, zero-copy) ===\n");
    printf(
        "%-8s  %-12s  %5s  %10s  %10s\n",
        "Blocks",
        "TileSize",
        "Iters",
        "Lat(us)",
        "BW(GB/s)");
    printf(
        "%-8s  %-12s  %5s  %10s  %10s\n",
        "------",
        "--------",
        "-----",
        "-------",
        "--------");
  }

  for (int nblk : block_counts) {
    auto r = runBenchmark(total, nblk);
    if (rank_ == 0) {
      double lat_us = r.elapsed_ms * 1000.0 / r.iterations;
      printf(
          "%-8d  %-12s  %5d  %10.1f  %10.2f\n",
          nblk,
          formatSize(total / nblk).c_str(),
          r.iterations,
          lat_us,
          r.bw_gbps);
    }
  }
}

TEST_F(PutBwBenchmark, SendRecvMultiBlock) {
  size_t total = 512 * MB;
  std::vector<int> block_counts = {1, 2, 4, 8, 16, 32, 64, 128};

  if (rank_ == 0) {
    printf(
        "\n=== SendRecv BW: Multi Block (512MB total, BLOCK scope, copy-based) ===\n");
    printf(
        "%-8s  %-12s  %5s  %10s  %10s\n",
        "Blocks",
        "TileSize",
        "Iters",
        "Lat(us)",
        "BW(GB/s)");
    printf(
        "%-8s  %-12s  %5s  %10s  %10s\n",
        "------",
        "--------",
        "-----",
        "-------",
        "--------");
  }

  for (int nblk : block_counts) {
    auto r = runSendRecvBenchmark(total, nblk);
    if (rank_ == 0) {
      double lat_us = r.elapsed_ms * 1000.0 / r.iterations;
      printf(
          "%-8d  %-12s  %5d  %10.1f  %10.2f\n",
          nblk,
          formatSize(total / nblk).c_str(),
          r.iterations,
          lat_us,
          r.bw_gbps);
    }
  }
}

TEST_F(PutBwBenchmark, SendRecvPipelinedMultiBlock) {
  // 128 blocks, sweep tile sizes by varying total (steps=1, PD=1).
  // Tile = total / 128. Larger tile → less per-tile overhead.
  int nblk = 128;
  std::vector<size_t> totals = {512 * MB, 1024 * MB, 2048 * MB, 4096UL * MB};

  if (rank_ == 0) {
    printf(
        "\n=== SendRecv BW: 128 Blocks, Tile Size Sweep (steps=1, PD=1) ===\n");
    printf("%-12s  %-10s  %10s\n", "Total", "Tile", "BW(GB/s)");
    printf("%-12s  %-10s  %10s\n", "-----", "----", "--------");
  }

  for (size_t total : totals) {
    size_t section = total; // steps=1
    int pd = 1;
    size_t tile = section / nblk;

    auto r = runSendRecvPipelinedBenchmark(total, section, pd, nblk);
    if (rank_ == 0) {
      printf(
          "%-12s  %-10s  %10.2f\n",
          formatSize(total).c_str(),
          formatSize(tile).c_str(),
          r.bw_gbps);
    }
  }
}

TEST_F(PutBwBenchmark, SendRecvPipelinedParamSweep) {
  // Comprehensive sweep: 128 blocks, tile sizes 1-16MB, pd sweep.
  // Matches Triton sweep configs exactly for direct comparison.
  size_t total = 2048UL * MB;
  std::vector<int> block_counts = {128};
  std::vector<size_t> section_sizes = {
      128 * MB, 256 * MB, 512 * MB, 1024 * MB, 2048 * MB};
  std::vector<int> pd_values = {1, 2, 4, 8, 16};

  if (rank_ == 0) {
    printf(
        "\n=== SendRecv Pipelined: Full Parameter Sweep (2GB total, 128 blocks) ===\n");
    printf(
        "%-22s | %-6s | %-10s | %-4s | %-10s | %-6s | %-10s | %-12s\n",
        "Name",
        "Blocks",
        "Section",
        "PD",
        "Tile",
        "Steps",
        "Staging",
        "BW(GB/s)");
    printf(
        "----------------------------------------------------------------------"
        "------------------------------------------------------\n");
  }

  for (int nblk : block_counts) {
    for (size_t sec : section_sizes) {
      if (sec > total)
        continue;
      int total_steps = total / sec;
      for (int pd : pd_values) {
        if (pd > total_steps)
          continue;
        // Skip configs that need >2GB staging per dir to avoid OOM
        size_t staging_per_dir = static_cast<size_t>(pd) * sec;
        if (staging_per_dir > 1024UL * MB)
          continue;
        size_t tile = sec / nblk;
        size_t staging = static_cast<size_t>(pd) * sec * 2; // send + recv

        char name[64];
        snprintf(
            name,
            sizeof(name),
            "b%d_s%s_p%d",
            nblk,
            formatSize(sec).c_str(),
            pd);

        auto r = runSendRecvPipelinedBenchmark(total, sec, pd, nblk);
        if (rank_ == 0) {
          printf(
              "%-22s | %-6d | %-10s | %-4d | %-10s | %-6d | %-10s | %-12.2f\n",
              name,
              nblk,
              formatSize(sec).c_str(),
              pd,
              formatSize(tile).c_str(),
              total_steps,
              formatSize(staging).c_str(),
              r.bw_gbps);
        }
      }
    }
  }
}
