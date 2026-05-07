// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include "comms/torchcomms/TorchCommHooks.hpp"
#include "comms/torchcomms/TorchCommTypes.hpp"
#include "comms/torchcomms/device/cuda/HashKernel.h"
#include "comms/torchcomms/hooks/common/ThreadSafeLogFile.hpp"

namespace torch::comms {

class TorchComm;

inline constexpr size_t kDefaultRingSize = 1024 * 1024;
inline constexpr int kDefaultNumBlocks = 8;

struct HashBuffer {
  HashEntry* hash_entries{nullptr};
  uint64_t* next_empty_hash_entry{nullptr};
  uint64_t next_unflushed_hash_entry{0};
  size_t num_hash_entries{0};

  ~HashBuffer();
  HashBuffer() = default;
  HashBuffer(const HashBuffer&) = delete;
  HashBuffer& operator=(const HashBuffer&) = delete;
  HashBuffer(HashBuffer&&) = delete;
  HashBuffer& operator=(HashBuffer&&) = delete;
};

class ChashHook : public std::enable_shared_from_this<ChashHook> {
 public:
  ChashHook(
      const std::string& output,
      size_t ring_size = kDefaultRingSize,
      int num_blocks = kDefaultNumBlocks);

  ~ChashHook();

  ChashHook(const ChashHook&) = delete;
  ChashHook& operator=(const ChashHook&) = delete;
  ChashHook(ChashHook&&) = delete;
  ChashHook& operator=(ChashHook&&) = delete;

  void registerWithComm(std::shared_ptr<TorchComm> comm);

 private:
  void onPreHook(TorchComm* comm, size_t op_id, const PreHookArgs& args);

  void onPostHook(TorchComm* comm, size_t op_id, const PostHookArgs& args);

  void flushAllHashes();
  static constexpr uint64_t PHASE_MASK = 1ULL << 63;
  static constexpr uint64_t PHASE_PRE = 0;
  static constexpr uint64_t PHASE_POST = 1ULL << 63;
  static constexpr uint64_t LABEL_MASK = ~PHASE_MASK;

  void launchHashKernel(
      int device_index,
      const at::Tensor& tensor,
      HashBuffer& buf,
      uint64_t user_context);

  size_t ring_size_;
  int num_blocks_;
  int max_threads_per_block_{0};

  ThreadSafeLogFile log_file_;
  std::atomic<uint64_t> next_label_{1};

  std::unordered_map<std::string, std::unique_ptr<HashBuffer>>
      comm_hash_buffers_;

  // op_id -> label mapping (pre-hook stores, post-hook consumes)
  std::unordered_map<size_t, uint64_t> op_labels_;

  struct TensorInfo {
    std::vector<at::Tensor> tensors;
    uint64_t label;
    std::string comm_name;
    int device_index;
  };
  std::mutex pending_mutex_;
  std::unordered_map<uint64_t, TensorInfo> pending_async_;
};

} // namespace torch::comms
