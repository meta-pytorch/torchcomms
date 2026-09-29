// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// IB SendRecv benchmark: measures RDMA put and pipelined send/recv bandwidth
// using the TorchComm device API (NCCLx/GIN backend).
//
// Tests:
//   PutSingleBlock   — one block puts the full buffer, sweep over sizes.
//   PutMultiBlock    — N blocks each put total/N bytes, sweep over block
//   counts. SendRecvMultiBlock — pipelined sendrecv with section=total
//   (steps=1, PD=1). SendRecvTileSweep  — 128 blocks, tile size sweep (steps=1,
//   PD=1). SendRecvParamSweep — blocks x section x PD sweep for Pareto
//   analysis. SendRecvLargeTileSweep — large total (1-4GB), tile sweep, PD
//   sweep.

#include <gtest/gtest.h>
#include <nccl.h> // @manual

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/MemPool.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual=//caffe2:torch-cpp-cpu

#include "comms/prims/collectives/benchmarks/IbSendRecvBenchmarkKernels.cuh"
#include "comms/prims/collectives/ib/SendRecv.cuh"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/device/TorchCommDeviceWindow.hpp"
#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

using namespace torchcomms::device;

#define NCCL_CHECK(cmd)                                                       \
  do {                                                                        \
    ncclResult_t res = cmd;                                                   \
    ASSERT_EQ(res, ncclSuccess) << "NCCL error: " << ncclGetErrorString(res); \
  } while (0)

namespace {

constexpr int kWarmupIters = 20;
constexpr int kMeasureIters = 100;
constexpr size_t KB = 1024;
constexpr size_t MB = 1024 * 1024;
constexpr size_t GB = 1024UL * 1024 * 1024;
constexpr size_t kMismatchCopyChunkBytes = 4 * MB;

bool expect_cuda_success(cudaError_t result, const char* expr) {
  if (result == cudaSuccess) {
    return true;
  }
  ADD_FAILURE() << expr << " failed: " << cudaGetErrorString(result);
  return false;
}

#define CUDA_EXPECT_SUCCESS(cmd)            \
  do {                                      \
    (void)expect_cuda_success((cmd), #cmd); \
  } while (0)

namespace ib_bench = comms::prims::ib::benchmark;

template <typename T>
class DeviceScratch {
 public:
  explicit DeviceScratch(size_t count) {
    CUDA_EXPECT_SUCCESS(cudaMalloc(&ptr_, count * sizeof(T)));
  }

  ~DeviceScratch() {
    if (ptr_ != nullptr) {
      CUDA_EXPECT_SUCCESS(cudaFree(ptr_));
    }
  }

  DeviceScratch(const DeviceScratch&) = delete;
  DeviceScratch& operator=(const DeviceScratch&) = delete;

  T* get() const {
    return ptr_;
  }

  T* at(size_t index) const {
    return ptr_ + index;
  }

 private:
  T* ptr_{nullptr};
};

class IterationEvents {
 public:
  explicit IterationEvents(int iterations)
      : starts_(iterations), stops_(iterations) {
    for (int i = 0; i < iterations; ++i) {
      CUDA_EXPECT_SUCCESS(cudaEventCreate(&starts_[i]));
      CUDA_EXPECT_SUCCESS(cudaEventCreate(&stops_[i]));
    }
  }

  ~IterationEvents() {
    for (auto event : starts_) {
      if (event != nullptr) {
        CUDA_EXPECT_SUCCESS(cudaEventDestroy(event));
      }
    }
    for (auto event : stops_) {
      if (event != nullptr) {
        CUDA_EXPECT_SUCCESS(cudaEventDestroy(event));
      }
    }
  }

  IterationEvents(const IterationEvents&) = delete;
  IterationEvents& operator=(const IterationEvents&) = delete;

  cudaEvent_t start(int iteration) const {
    return starts_[iteration];
  }

  cudaEvent_t stop(int iteration) const {
    return stops_[iteration];
  }

  int iterations() const {
    return static_cast<int>(starts_.size());
  }

 private:
  std::vector<cudaEvent_t> starts_;
  std::vector<cudaEvent_t> stops_;
};

void poison_destination(void* data, size_t nbytes, cudaStream_t stream) {
  CUDA_EXPECT_SUCCESS(cudaMemsetAsync(data, 0xa5, nbytes, stream));
}

float sum_elapsed_ms(const IterationEvents& events) {
  float elapsed_ms = 0;
  for (int i = 0; i < events.iterations(); ++i) {
    float iteration_ms = 0;
    CUDA_EXPECT_SUCCESS(
        cudaEventElapsedTime(&iteration_ms, events.start(i), events.stop(i)));
    elapsed_ms += iteration_ms;
  }
  return elapsed_ms;
}

std::string format_size(size_t bytes) {
  if (bytes >= GB) {
    return std::to_string(bytes / GB) + "GB";
  }
  if (bytes >= MB) {
    return std::to_string(bytes / MB) + "MB";
  }
  if (bytes >= KB) {
    return std::to_string(bytes / KB) + "KB";
  }
  return std::to_string(bytes) + "B";
}

struct ByteMismatch {
  size_t byte_index;
  std::uint8_t observed;
  std::uint8_t expected;
};

std::optional<ByteMismatch> find_first_rank_pattern_mismatch(
    const void* data,
    size_t nbytes,
    int expected_rank,
    cudaStream_t stream) {
  if (nbytes == 0) {
    return std::nullopt;
  }

  const auto* bytes = static_cast<const std::uint8_t*>(data);
  std::vector<std::uint8_t> host(std::min(kMismatchCopyChunkBytes, nbytes));
  for (size_t offset = 0; offset < nbytes; offset += host.size()) {
    const size_t chunk_bytes = std::min(host.size(), nbytes - offset);
    if (!expect_cuda_success(
            cudaMemcpyAsync(
                host.data(),
                bytes + offset,
                chunk_bytes,
                cudaMemcpyDeviceToHost,
                stream),
            "cudaMemcpyAsync(mismatch chunk)")) {
      return std::nullopt;
    }
    if (!expect_cuda_success(
            cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(mismatch chunk)")) {
      return std::nullopt;
    }
    for (size_t i = 0; i < chunk_bytes; ++i) {
      const size_t byte_index = offset + i;
      const std::uint8_t expected =
          comms::prims::ib::benchmark::expected_rank_pattern_byte(
              expected_rank, byte_index);
      if (host[i] != expected) {
        return ByteMismatch{
            .byte_index = byte_index,
            .observed = host[i],
            .expected = expected,
        };
      }
    }
  }

  return std::nullopt;
}

std::string describe_first_mismatch(
    const std::optional<ByteMismatch>& mismatch) {
  if (!mismatch.has_value()) {
    return " firstMismatchByte=not-found";
  }
  return " firstMismatchByte=" + std::to_string(mismatch->byte_index) +
      " observed=" + std::to_string(static_cast<int>(mismatch->observed)) +
      " expected=" + std::to_string(static_cast<int>(mismatch->expected));
}

std::optional<ByteMismatch> find_first_mismatch_from_records(
    const std::vector<ib_bench::MismatchRecord>& mismatch_partials,
    int iteration) {
  const size_t offset =
      static_cast<size_t>(iteration) * ib_bench::kChecksumPartialBlocks;
  uint64_t index = ib_bench::kNoMismatchIndex;
  uint32_t observed = 0;
  uint32_t expected = 0;
  for (int i = 0; i < ib_bench::kChecksumPartialBlocks; ++i) {
    const auto& record = mismatch_partials[offset + i];
    if (record.index < index) {
      index = record.index;
      observed = record.observed;
      expected = record.expected;
    }
  }

  if (index == ib_bench::kNoMismatchIndex) {
    return std::nullopt;
  }
  return ByteMismatch{
      .byte_index = static_cast<size_t>(index),
      .observed = static_cast<uint8_t>(observed),
      .expected = static_cast<uint8_t>(expected),
  };
}

struct ChecksumVerification {
  int wrong{0};
  int first_wrong_iteration{-1};
  uint64_t observed_checksum{0};
  uint64_t expected_checksum{0};
  std::optional<ByteMismatch> mismatch;
};

ChecksumVerification verify_checksums(
    const std::vector<uint64_t>& observed_checksums,
    uint64_t expected_checksum,
    const std::vector<ib_bench::MismatchRecord>& mismatch_partials) {
  ChecksumVerification result;
  result.expected_checksum = expected_checksum;
  if (!observed_checksums.empty()) {
    result.observed_checksum = observed_checksums.front();
  }

  for (int i = 0; i < static_cast<int>(observed_checksums.size()); ++i) {
    if (observed_checksums[i] != expected_checksum) {
      ++result.wrong;
      if (result.first_wrong_iteration < 0) {
        result.first_wrong_iteration = i;
        result.observed_checksum = observed_checksums[i];
        result.mismatch =
            find_first_mismatch_from_records(mismatch_partials, i);
      }
    }
  }
  return result;
}

struct WindowSetup {
  std::unique_ptr<at::cuda::MemPool> mem_pool;
  at::Tensor win_tensor;
  at::Tensor src_tensor;
  std::shared_ptr<torch::comms::TorchCommWindow> win;
  DeviceWindowNCCL* dev_win{nullptr};
  RegisteredBufferNCCL src_buf{};
};

WindowSetup create_window_setup(
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

  // Allocate src outside pool for proper alignment.
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

void teardown_window(
    WindowSetup& s,
    std::shared_ptr<torch::comms::TorchComm>& torchcomm) {
  s.win->deregister_local_buffer(s.src_buf);
  s.win->tensor_deregister();
  s.win.reset();
  s.mem_pool.reset();
  torchcomm->barrier(false);
}

} // namespace

class IbSendRecvBenchmark : public ::testing::Test {
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

    // Create a standalone NCCL communicator for baseline comparison.
    // Bootstrap ncclUniqueId via TCPStore (same MASTER_ADDR/PORT as TorchComm).
    const char* host = std::getenv("MASTER_ADDR");
    const char* port_str = std::getenv("MASTER_PORT");
    if (host && port_str) {
      c10d::TCPStoreOptions opts;
      opts.port = std::stoi(port_str);
      opts.isServer = (rank_ == 0);
      opts.waitWorkers = false;
      opts.useLibUV = true;
      auto store = c10::make_intrusive<c10d::TCPStore>(std::string{host}, opts);
      auto prefixed =
          c10::make_intrusive<c10d::PrefixStore>("nccl_baseline", store);

      ncclUniqueId nccl_id;
      if (rank_ == 0) {
        ncclGetUniqueId(&nccl_id);
        auto id_vec = std::vector<uint8_t>(
            reinterpret_cast<uint8_t*>(&nccl_id),
            reinterpret_cast<uint8_t*>(&nccl_id) + sizeof(nccl_id));
        prefixed->set("nccl_id", id_vec);
      }
      prefixed->wait({"nccl_id"});
      auto id_vec = prefixed->get("nccl_id");
      memcpy(&nccl_id, id_vec.data(), sizeof(nccl_id));

      ncclCommInitRank(&nccl_comm_, num_ranks_, nccl_id, rank_);
    }
  }

  void TearDown() override {
    if (nccl_comm_) {
      ncclCommDestroy(nccl_comm_);
      nccl_comm_ = nullptr;
    }
    torchcomm_.reset();
    wrapper_.reset();
  }

  struct BenchResult {
    size_t total_bytes;
    int num_blocks;
    int iterations;
    float elapsed_ms;
    double bw_gbps;
    int wrong;
    uint64_t observed_checksum;
    uint64_t expected_checksum;
  };

  BenchResult run_put_benchmark(size_t total_bytes, int num_blocks) {
    int signal_count = 2 * num_blocks;
    auto s = create_window_setup(
        torchcomm_, allocator_, device_index_, total_bytes, signal_count);

    int dst_rank = (rank_ + 1) % num_ranks_;
    int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

    auto stream = at::cuda::getStreamFromPool(false, device_index_);

    // Warmup
    {
      c10::cuda::CUDAStreamGuard guard(stream);
      comms::prims::ib::benchmark::launch_put_bw_kernel(
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
      comms::prims::ib::benchmark::launch_put_bw_kernel(
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
    teardown_window(s, torchcomm_);

    return {total_bytes, num_blocks, kMeasureIters, elapsed_ms, bw, 0, 0, 0};
  }

  BenchResult run_send_recv_benchmark(
      size_t total_bytes,
      size_t section_bytes,
      int pipeline_depth,
      int num_blocks) {
    int signal_count = 2 * num_blocks;
    int counter_count = num_blocks;

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

    {
      c10::cuda::CUDAStreamGuard guard(stream);
      comms::prims::ib::benchmark::launch_init_rank_pattern_kernel(
          src_ptr, total_bytes, rank_, stream.stream());
    }

    // Persistent step state: 2 * num_blocks (senders + receivers).
    DeviceScratch<int64_t> step_state(2 * num_blocks);
    CUDA_EXPECT_SUCCESS(
        cudaMemset(step_state.get(), 0, 2 * num_blocks * sizeof(int64_t)));

    {
      c10::cuda::CUDAStreamGuard guard(stream);
      for (int i = 0; i < kWarmupIters; ++i) {
        comms::prims::ib::launch_send_recv_kernel(
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
            step_state.get(),
            stream.stream());
      }
    }
    stream.synchronize();
    torchcomm_->barrier(false);

    DeviceScratch<uint64_t> expected_checksum_device(1);
    DeviceScratch<uint64_t> expected_partials_device(
        ib_bench::kChecksumPartialBlocks);
    DeviceScratch<uint64_t> observed_checksums_device(kMeasureIters);
    DeviceScratch<uint64_t> observed_partials_device(
        kMeasureIters * ib_bench::kChecksumPartialBlocks);
    DeviceScratch<ib_bench::MismatchRecord> mismatch_partials_device(
        kMeasureIters * ib_bench::kChecksumPartialBlocks);

    {
      c10::cuda::CUDAStreamGuard guard(stream);
      ib_bench::launch_rank_pattern_checksum(
          total_bytes,
          src_rank,
          expected_checksum_device.get(),
          expected_partials_device.get(),
          stream.stream());
    }

    IterationEvents events(kMeasureIters);
    std::vector<uint64_t> observed_checksums(kMeasureIters);
    std::vector<ib_bench::MismatchRecord> mismatch_partials(
        kMeasureIters * ib_bench::kChecksumPartialBlocks);
    uint64_t expectedChecksum = 0;

    {
      c10::cuda::CUDAStreamGuard guard(stream);
      for (int i = 0; i < kMeasureIters; ++i) {
        poison_destination(dst_ptr, total_bytes, stream.stream());
        CUDA_EXPECT_SUCCESS(cudaEventRecord(events.start(i), stream.stream()));
        comms::prims::ib::launch_send_recv_kernel(
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
            step_state.get(),
            stream.stream());
        CUDA_EXPECT_SUCCESS(cudaEventRecord(events.stop(i), stream.stream()));
        ib_bench::launch_buffer_checksum(
            dst_ptr,
            total_bytes,
            src_rank,
            observed_checksums_device.at(i),
            observed_partials_device.at(
                static_cast<size_t>(i) * ib_bench::kChecksumPartialBlocks),
            mismatch_partials_device.at(
                static_cast<size_t>(i) * ib_bench::kChecksumPartialBlocks),
            stream.stream());
        poison_destination(dst_ptr, total_bytes, stream.stream());
      }
      CUDA_EXPECT_SUCCESS(cudaMemcpyAsync(
          observed_checksums.data(),
          observed_checksums_device.get(),
          observed_checksums.size() * sizeof(uint64_t),
          cudaMemcpyDeviceToHost,
          stream.stream()));
      CUDA_EXPECT_SUCCESS(cudaMemcpyAsync(
          &expectedChecksum,
          expected_checksum_device.get(),
          sizeof(uint64_t),
          cudaMemcpyDeviceToHost,
          stream.stream()));
      CUDA_EXPECT_SUCCESS(cudaMemcpyAsync(
          mismatch_partials.data(),
          mismatch_partials_device.get(),
          mismatch_partials.size() * sizeof(ib_bench::MismatchRecord),
          cudaMemcpyDeviceToHost,
          stream.stream()));
    }
    stream.synchronize();

    float elapsed_ms = sum_elapsed_ms(events);
    double total_data = static_cast<double>(total_bytes) * kMeasureIters;
    double bw = total_data / (elapsed_ms / 1000.0) / 1e9;
    const auto verification = verify_checksums(
        observed_checksums, expectedChecksum, mismatch_partials);
    EXPECT_EQ(verification.wrong, 0)
        << "SendRecv checksum mismatch for total=" << format_size(total_bytes)
        << " section=" << format_size(section_bytes) << " pd=" << pipeline_depth
        << " nblk=" << num_blocks << " wrong=" << verification.wrong
        << " firstWrongIteration=" << verification.first_wrong_iteration
        << " observedChecksum=" << verification.observed_checksum
        << " expectedChecksum=" << verification.expected_checksum
        << describe_first_mismatch(verification.mismatch);

    // Teardown
    win->deregister_local_buffer(staging_buf);
    win->tensor_deregister();
    win.reset();
    mem_pool.reset();
    torchcomm_->barrier(false);

    return {
        total_bytes,
        num_blocks,
        kMeasureIters,
        elapsed_ms,
        bw,
        verification.wrong,
        verification.observed_checksum,
        verification.expected_checksum};
  }

  BenchResult run_nccl_baseline(size_t total_bytes) {
    EXPECT_NE(nccl_comm_, nullptr) << "NCCL comm not initialized";
    int peer_rank = (rank_ + 1) % num_ranks_;

    auto options =
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index_);
    size_t total_count = total_bytes / sizeof(float);
    auto send_tensor = at::zeros({static_cast<int64_t>(total_count)}, options);
    auto recv_tensor = at::zeros({static_cast<int64_t>(total_count)}, options);

    auto stream = at::cuda::getStreamFromPool(false, device_index_);
    {
      c10::cuda::CUDAStreamGuard guard(stream);
      comms::prims::ib::benchmark::launch_init_rank_pattern_kernel(
          send_tensor.data_ptr(), total_bytes, rank_, stream.stream());
    }

    {
      c10::cuda::CUDAStreamGuard guard(stream);
      for (int i = 0; i < kWarmupIters; ++i) {
        ncclGroupStart();
        ncclSend(
            send_tensor.data_ptr(),
            total_bytes,
            ncclChar,
            peer_rank,
            nccl_comm_,
            stream.stream());
        ncclRecv(
            recv_tensor.data_ptr(),
            total_bytes,
            ncclChar,
            peer_rank,
            nccl_comm_,
            stream.stream());
        ncclGroupEnd();
      }
    }
    stream.synchronize();
    torchcomm_->barrier(false);

    DeviceScratch<uint64_t> expected_checksum_device(1);
    DeviceScratch<uint64_t> expected_partials_device(
        ib_bench::kChecksumPartialBlocks);
    DeviceScratch<uint64_t> observed_checksums_device(kMeasureIters);
    DeviceScratch<uint64_t> observed_partials_device(
        kMeasureIters * ib_bench::kChecksumPartialBlocks);
    DeviceScratch<ib_bench::MismatchRecord> mismatch_partials_device(
        kMeasureIters * ib_bench::kChecksumPartialBlocks);

    {
      c10::cuda::CUDAStreamGuard guard(stream);
      ib_bench::launch_rank_pattern_checksum(
          total_bytes,
          peer_rank,
          expected_checksum_device.get(),
          expected_partials_device.get(),
          stream.stream());
    }

    IterationEvents events(kMeasureIters);
    std::vector<uint64_t> observed_checksums(kMeasureIters);
    std::vector<ib_bench::MismatchRecord> mismatch_partials(
        kMeasureIters * ib_bench::kChecksumPartialBlocks);
    uint64_t expectedChecksum = 0;

    {
      c10::cuda::CUDAStreamGuard guard(stream);
      for (int i = 0; i < kMeasureIters; ++i) {
        poison_destination(
            recv_tensor.data_ptr(), total_bytes, stream.stream());
        CUDA_EXPECT_SUCCESS(cudaEventRecord(events.start(i), stream.stream()));
        ncclGroupStart();
        ncclSend(
            send_tensor.data_ptr(),
            total_bytes,
            ncclChar,
            peer_rank,
            nccl_comm_,
            stream.stream());
        ncclRecv(
            recv_tensor.data_ptr(),
            total_bytes,
            ncclChar,
            peer_rank,
            nccl_comm_,
            stream.stream());
        ncclGroupEnd();
        CUDA_EXPECT_SUCCESS(cudaEventRecord(events.stop(i), stream.stream()));
        ib_bench::launch_buffer_checksum(
            recv_tensor.data_ptr(),
            total_bytes,
            peer_rank,
            observed_checksums_device.at(i),
            observed_partials_device.at(
                static_cast<size_t>(i) * ib_bench::kChecksumPartialBlocks),
            mismatch_partials_device.at(
                static_cast<size_t>(i) * ib_bench::kChecksumPartialBlocks),
            stream.stream());
        poison_destination(
            recv_tensor.data_ptr(), total_bytes, stream.stream());
      }
      CUDA_EXPECT_SUCCESS(cudaMemcpyAsync(
          observed_checksums.data(),
          observed_checksums_device.get(),
          observed_checksums.size() * sizeof(uint64_t),
          cudaMemcpyDeviceToHost,
          stream.stream()));
      CUDA_EXPECT_SUCCESS(cudaMemcpyAsync(
          &expectedChecksum,
          expected_checksum_device.get(),
          sizeof(uint64_t),
          cudaMemcpyDeviceToHost,
          stream.stream()));
      CUDA_EXPECT_SUCCESS(cudaMemcpyAsync(
          mismatch_partials.data(),
          mismatch_partials_device.get(),
          mismatch_partials.size() * sizeof(ib_bench::MismatchRecord),
          cudaMemcpyDeviceToHost,
          stream.stream()));
    }
    stream.synchronize();

    float elapsed_ms = sum_elapsed_ms(events);
    double total_data = static_cast<double>(total_bytes) * kMeasureIters;
    double bw = total_data / (elapsed_ms / 1000.0) / 1e9;
    const auto verification = verify_checksums(
        observed_checksums, expectedChecksum, mismatch_partials);
    EXPECT_EQ(verification.wrong, 0)
        << "NCCL SendRecv checksum mismatch for total="
        << format_size(total_bytes) << " wrong=" << verification.wrong
        << " firstWrongIteration=" << verification.first_wrong_iteration
        << " observedChecksum=" << verification.observed_checksum
        << " expectedChecksum=" << verification.expected_checksum
        << describe_first_mismatch(verification.mismatch);

    return {
        total_bytes,
        0,
        kMeasureIters,
        elapsed_ms,
        bw,
        verification.wrong,
        verification.observed_checksum,
        verification.expected_checksum};
  }

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  std::shared_ptr<c10::Allocator> allocator_;
  ncclComm_t nccl_comm_{nullptr};
  int rank_{0};
  int num_ranks_{0};
  int device_index_{0};
};

TEST_F(IbSendRecvBenchmark, PutSingleBlock) {
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
    auto r = run_put_benchmark(sz, 1);
    if (rank_ == 0) {
      double lat_us = r.elapsed_ms * 1000.0 / r.iterations;
      printf(
          "%-12s  %5d  %10.1f  %10.2f\n",
          format_size(r.total_bytes).c_str(),
          r.iterations,
          lat_us,
          r.bw_gbps);
    }
  }
}

TEST_F(IbSendRecvBenchmark, PutMultiBlock) {
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
    auto r = run_put_benchmark(total, nblk);
    if (rank_ == 0) {
      double lat_us = r.elapsed_ms * 1000.0 / r.iterations;
      printf(
          "%-8d  %-12s  %5d  %10.1f  %10.2f\n",
          nblk,
          format_size(total / nblk).c_str(),
          r.iterations,
          lat_us,
          r.bw_gbps);
    }
  }
}

TEST_F(IbSendRecvBenchmark, SendRecvMultiBlock) {
  // Uses pipelined kernel with section=total (steps=1, PD=1) — equivalent
  // to the old non-pipelined sendRecvBwKernel.
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
    auto r = run_send_recv_benchmark(total, total, 1, nblk);
    if (rank_ == 0) {
      double lat_us = r.elapsed_ms * 1000.0 / r.iterations;
      printf(
          "%-8d  %-12s  %5d  %10.1f  %10.2f\n",
          nblk,
          format_size(total / nblk).c_str(),
          r.iterations,
          lat_us,
          r.bw_gbps);
    }
  }
}

TEST_F(IbSendRecvBenchmark, SendRecvTileSweep) {
  // 128 blocks, sweep tile sizes by varying total (steps=1, PD=1).
  // Tile = total / 128. Larger tile -> less per-tile overhead.
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

    auto r = run_send_recv_benchmark(total, section, pd, nblk);
    if (rank_ == 0) {
      printf(
          "%-12s  %-10s  %10.2f\n",
          format_size(total).c_str(),
          format_size(tile).c_str(),
          r.bw_gbps);
    }
  }
}

TEST_F(IbSendRecvBenchmark, SendRecvParamSweep) {
  // Full parameter sweep: blocks x section x pd for Pareto analysis.
  size_t total = 512 * MB;
  std::vector<int> block_counts = {16, 32, 64, 128};
  std::vector<size_t> section_sizes = {
      32 * MB, 64 * MB, 128 * MB, 256 * MB, 512 * MB};
  std::vector<int> pd_values = {1, 2, 4};

  if (rank_ == 0) {
    printf(
        "\n=== SendRecv Pipelined: Full Parameter Sweep (512MB total) ===\n");
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
      if (sec > total) {
        continue;
      }
      int total_steps = total / sec;
      for (int pd : pd_values) {
        if (pd > total_steps) {
          continue;
        }
        size_t tile = sec / nblk;
        size_t staging = static_cast<size_t>(pd) * sec * 2; // send + recv

        char name[64];
        snprintf(
            name,
            sizeof(name),
            "b%d_s%s_p%d",
            nblk,
            format_size(sec).c_str(),
            pd);

        auto r = run_send_recv_benchmark(total, sec, pd, nblk);
        if (rank_ == 0) {
          printf(
              "%-22s | %-6d | %-10s | %-4d | %-10s | %-6d | %-10s | %-12.2f\n",
              name,
              nblk,
              format_size(sec).c_str(),
              pd,
              format_size(tile).c_str(),
              total_steps,
              format_size(staging).c_str(),
              r.bw_gbps);
        }
      }
    }
  }
}

TEST_F(IbSendRecvBenchmark, SendRecvSizeSweep) {
  // Full size sweep from 32KB to 4GB. 128 blocks, section=total (1 step),
  // PD=1. Tile = total/128. Shows bandwidth across the full size range.
  int nblk = 128;
  int pd = 1;
  std::vector<size_t> sizes = {
      32 * KB,
      64 * KB,
      128 * KB,
      256 * KB,
      512 * KB,
      1 * MB,
      2 * MB,
      4 * MB,
      8 * MB,
      16 * MB,
      32 * MB,
      64 * MB,
      128 * MB,
      256 * MB,
      512 * MB,
      1 * GB,
      2 * GB,
      4 * GB};

  if (rank_ == 0) {
    printf("\n=== SendRecv Size Sweep (128 blocks, section=total, PD=1) ===\n");
    printf("%-12s  %-10s  %10s\n", "Total", "Tile", "BW(GB/s)");
    printf("%-12s  %-10s  %10s\n", "-----", "----", "--------");
  }

  for (size_t total : sizes) {
    size_t section = total;
    auto r = run_send_recv_benchmark(total, section, pd, nblk);
    if (rank_ == 0) {
      printf(
          "%-12s  %-10s  %10.2f\n",
          format_size(total).c_str(),
          format_size(total / nblk).c_str(),
          r.bw_gbps);
    }
  }
}

TEST_F(IbSendRecvBenchmark, SendRecvLargeTileSweep) {
  // Large-buffer sweep: total={1GB,2GB,4GB}, tile={8MB,16MB}, PD={1,2,4},
  // 128 blocks. OOM guard at 4GB per direction.
  int nblk = 128;
  std::vector<size_t> totals = {1 * GB, 2 * GB, 4 * GB};
  std::vector<size_t> tile_sizes = {8 * MB, 16 * MB};
  std::vector<int> pd_values = {1, 2, 4};

  if (rank_ == 0) {
    printf("\n=== SendRecv Large Tile Sweep (128 blocks) ===\n");
    printf(
        "%-12s | %-10s | %-10s | %-4s | %-6s | %-10s | %-12s\n",
        "Total",
        "Section",
        "Tile",
        "PD",
        "Steps",
        "Staging",
        "BW(GB/s)");
    printf(
        "----------------------------------------------------------------------"
        "----------------------------------\n");
  }

  for (size_t total : totals) {
    for (size_t tile : tile_sizes) {
      size_t section = tile * nblk;
      if (section > total) {
        continue;
      }
      int total_steps = total / section;
      for (int pd : pd_values) {
        if (pd > total_steps) {
          continue;
        }
        // OOM guard: staging + window = 2 * pd * section, src + dst = 2 * total
        size_t ring_bytes = static_cast<size_t>(pd) * section;
        size_t mem_per_dir = total + ring_bytes * 2; // src/dst + staging + win
        if (mem_per_dir > 16 * GB) {
          if (rank_ == 0) {
            printf(
                "%-12s | %-10s | %-10s | %-4d | SKIP (OOM: %s/dir)\n",
                format_size(total).c_str(),
                format_size(section).c_str(),
                format_size(tile).c_str(),
                pd,
                format_size(mem_per_dir).c_str());
          }
          continue;
        }

        auto r = run_send_recv_benchmark(total, section, pd, nblk);
        if (rank_ == 0) {
          size_t staging = static_cast<size_t>(pd) * section * 2;
          printf(
              "%-12s | %-10s | %-10s | %-4d | %-6d | %-10s | %-12.2f\n",
              format_size(total).c_str(),
              format_size(section).c_str(),
              format_size(tile).c_str(),
              pd,
              total_steps,
              format_size(staging).c_str(),
              r.bw_gbps);
        }
      }
    }
  }
}

TEST_F(IbSendRecvBenchmark, SendRecvCorrectness) {
  // Verify data correctness: fill src with rank-specific pattern, run
  // sendrecv, check dst has peer's pattern. Tests multiple configs
  // including PD > 1 to exercise the ring buffer offset logic.
  struct Config {
    size_t total;
    size_t section;
    int pd;
    int nblk;
  };
  std::vector<Config> configs = {
      // PD=1: single slot, no ring offset
      {64 * KB, 64 * KB, 1, 128},
      {1 * MB, 1 * MB, 1, 128},
      {16 * MB, 16 * MB, 1, 128},
      // PD=2: exercises ring buffer slot offsets
      {2 * MB, 1 * MB, 2, 128},
      {16 * MB, 8 * MB, 2, 128},
      // PD=4: deeper pipeline
      {16 * MB, 4 * MB, 4, 128},
      // Fewer blocks
      {4 * MB, 4 * MB, 1, 32},
      {4 * MB, 2 * MB, 2, 64},
  };

  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

  for (const auto& cfg : configs) {
    int signal_count = 2 * cfg.nblk;
    int counter_count = cfg.nblk;
    size_t ring_bytes = cfg.pd * cfg.section;
    size_t ring_count = ring_bytes / sizeof(float);
    size_t total_count = cfg.total / sizeof(float);

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

    auto stream = at::cuda::getStreamFromPool(false, device_index_);
    {
      c10::cuda::CUDAStreamGuard guard(stream);
      comms::prims::ib::benchmark::launch_init_rank_pattern_kernel(
          src_tensor.data_ptr<float>(), cfg.total, rank_, stream.stream());
    }

    // Persistent step state for correctness test (single launch).
    int64_t* step_state = nullptr;
    cudaMalloc(&step_state, 2 * cfg.nblk * sizeof(int64_t));
    cudaMemset(step_state, 0, 2 * cfg.nblk * sizeof(int64_t));

    {
      c10::cuda::CUDAStreamGuard guard(stream);
      comms::prims::ib::launch_send_recv_kernel(
          dev_win,
          staging_buf,
          src_tensor.data_ptr<float>(),
          staging_tensor.data_ptr<float>(),
          win_tensor.data_ptr<float>(),
          dst_tensor.data_ptr<float>(),
          cfg.total,
          cfg.section,
          cfg.pd,
          dst_rank,
          src_rank,
          cfg.nblk,
          step_state,
          stream.stream());
    }
    stream.synchronize();
    torchcomm_->barrier(false);
    cudaFree(step_state);

    const uint64_t observedChecksum =
        comms::prims::ib::benchmark::compute_buffer_checksum(
            dst_tensor.data_ptr<float>(), cfg.total, stream.stream());
    const uint64_t expectedChecksum =
        comms::prims::ib::benchmark::compute_rank_pattern_checksum(
            cfg.total, src_rank, stream.stream());
    const bool match = observedChecksum == expectedChecksum;
    const auto mismatch = match ? std::optional<ByteMismatch>{}
                                : find_first_rank_pattern_mismatch(
                                      dst_tensor.data_ptr<float>(),
                                      cfg.total,
                                      src_rank,
                                      stream.stream());
    EXPECT_TRUE(match) << "Correctness FAILED for total="
                       << format_size(cfg.total)
                       << " section=" << format_size(cfg.section)
                       << " pd=" << cfg.pd << " nblk=" << cfg.nblk
                       << " observedChecksum=" << observedChecksum
                       << " expectedChecksum=" << expectedChecksum
                       << describe_first_mismatch(mismatch);
    if (rank_ == 0) {
      const std::string mismatch_details =
          match ? "" : describe_first_mismatch(mismatch);
      printf(
          "Correctness: total=%-8s section=%-8s pd=%d nblk=%-4d wrong=%d %s%s\n",
          format_size(cfg.total).c_str(),
          format_size(cfg.section).c_str(),
          cfg.pd,
          cfg.nblk,
          match ? 0 : 1,
          match ? "PASS" : "FAIL",
          mismatch_details.c_str());
    }

    win->deregister_local_buffer(staging_buf);
    win->tensor_deregister();
    win.reset();
    mem_pool.reset();
    torchcomm_->barrier(false);
  }
}

TEST_F(IbSendRecvBenchmark, NcclSendRecvMeasuredLoopCorrectnessSmall) {
  constexpr size_t total = 64 * KB;

  auto nccl_r = run_nccl_baseline(total);

  EXPECT_EQ(nccl_r.wrong, 0);
  EXPECT_GT(nccl_r.elapsed_ms, 0);
}

TEST_F(IbSendRecvBenchmark, SendRecvVsNccl) {
  // Side-by-side comparison: our sendrecv kernel vs NCCL baseline.
  // Both use the same sizes, warmup, and measurement iterations.
  int nblk = 128;
  int pd = 1;
  std::vector<size_t> sizes = {
      32 * KB,
      64 * KB,
      128 * KB,
      256 * KB,
      512 * KB,
      1 * MB,
      2 * MB,
      4 * MB,
      8 * MB,
      16 * MB,
      32 * MB,
      64 * MB,
      128 * MB,
      256 * MB,
      512 * MB,
      1 * GB,
      2 * GB,
      4 * GB};

  if (rank_ == 0) {
    printf(
        "\n=== SendRecv vs NCCL Baseline (128 blocks, section=total, PD=1) ===\n");
    printf(
        "%-12s  %12s  %12s  %10s\n",
        "Total",
        "Kernel(GB/s)",
        "NCCL(GB/s)",
        "Speedup");
    printf(
        "%-12s  %12s  %12s  %10s\n",
        "-----",
        "------------",
        "----------",
        "-------");
  }

  for (size_t total : sizes) {
    auto kernel_r = run_send_recv_benchmark(total, total, pd, nblk);
    auto nccl_r = run_nccl_baseline(total);

    if (rank_ == 0) {
      double speedup =
          nccl_r.bw_gbps > 0 ? kernel_r.bw_gbps / nccl_r.bw_gbps : 0;
      printf(
          "%-12s  %12.2f  %12.2f  %9.2fx\n",
          format_size(total).c_str(),
          kernel_r.bw_gbps,
          nccl_r.bw_gbps,
          speedup);
    }
  }
}

TEST_F(IbSendRecvBenchmark, SendRecvFewBlocks) {
  // Compare kernel with 1 and 2 blocks vs NCCL baseline.
  // Uses section=total (1 step), PD=1. Large sizes only.
  int pd = 1;
  std::vector<int> block_counts = {1, 2};
  std::vector<size_t> sizes = {
      1 * MB,
      4 * MB,
      16 * MB,
      64 * MB,
      256 * MB,
      512 * MB,
      1 * GB,
      2 * GB,
      4 * GB};

  for (int nblk : block_counts) {
    if (rank_ == 0) {
      printf(
          "\n=== SendRecv vs NCCL (%d block%s, section=total, PD=1) ===\n",
          nblk,
          nblk > 1 ? "s" : "");
      printf(
          "%-12s  %12s  %12s  %10s\n",
          "Total",
          "Kernel(GB/s)",
          "NCCL(GB/s)",
          "Speedup");
      printf(
          "%-12s  %12s  %12s  %10s\n",
          "-----",
          "------------",
          "----------",
          "-------");
    }

    for (size_t total : sizes) {
      auto kernel_r = run_send_recv_benchmark(total, total, pd, nblk);
      auto nccl_r = run_nccl_baseline(total);

      if (rank_ == 0) {
        double speedup =
            nccl_r.bw_gbps > 0 ? kernel_r.bw_gbps / nccl_r.bw_gbps : 0;
        printf(
            "%-12s  %12.2f  %12.2f  %9.2fx\n",
            format_size(total).c_str(),
            kernel_r.bw_gbps,
            nccl_r.bw_gbps,
            speedup);
      }
    }
  }
}
