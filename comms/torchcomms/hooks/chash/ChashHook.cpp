// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/torchcomms/hooks/chash/ChashHook.hpp"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/hooks/common/SignatureBuilder.hpp"

#include <fmt/core.h>

#if __has_include(<cuda_runtime_api.h>) && __has_include(<ATen/cuda/CUDAContext.h>)
#include <ATen/cuda/CUDAContext.h>
#define CHASH_HAS_CUDA 1
#endif

namespace torch::comms {

namespace {
constexpr int kChashVersion = 1;
} // namespace

// -- HashBuffer --

HashBuffer::~HashBuffer() {
#ifdef CHASH_HAS_CUDA
  if (hash_entries) {
    cudaFreeHost(hash_entries);
  }
  if (next_empty_hash_entry) {
    cudaFreeHost(next_empty_hash_entry);
  }
#endif
}

// -- ChashHook public --

ChashHook::ChashHook(
    const std::string& output,
    size_t ring_size,
    int num_blocks)
    : ring_size_(ring_size), num_blocks_(num_blocks) {
  log_file_.open(output);
  log_file_.writeLine(fmt::format("V|{}", kChashVersion));
}

ChashHook::~ChashHook() = default;

void ChashHook::registerWithComm(std::shared_ptr<TorchComm> comm) {
#ifndef CHASH_HAS_CUDA
  log_file_.writeLine("WARN|CUDA not enabled. Skipping registration.");
  return;
#else
  std::string comm_name(comm->getCommName());

  log_file_.writeLine(
      buildNewCommSignature(comm_name, comm->getRank(), comm->getSize()));

  if (max_threads_per_block_ == 0) {
    int device_index = comm->getDevice().index();
    if (device_index >= 0) {
      cudaDeviceProp prop;
      auto err = cudaGetDeviceProperties(&prop, device_index);
      if (err != cudaSuccess) {
        throw std::runtime_error("ChashHook: cudaGetDeviceProperties failed");
      }
      max_threads_per_block_ = prop.maxThreadsPerBlock;
    }
  }

  auto buf = std::make_unique<HashBuffer>();
  buf->num_hash_entries = ring_size_;

  void* entries_alloc = nullptr;
  auto err = cudaHostAlloc(
      &entries_alloc, ring_size_ * sizeof(HashEntry), cudaHostAllocDefault);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "ChashHook: cudaHostAlloc failed for hash entries");
  }
  buf->hash_entries = static_cast<HashEntry*>(entries_alloc);
  memset(buf->hash_entries, 0, ring_size_ * sizeof(HashEntry));

  void* unfilled_alloc = nullptr;
  err = cudaHostAlloc(&unfilled_alloc, sizeof(uint64_t), cudaHostAllocDefault);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "ChashHook: cudaHostAlloc failed for next_empty_hash_entry");
  }
  buf->next_empty_hash_entry = static_cast<uint64_t*>(unfilled_alloc);
  *buf->next_empty_hash_entry = 0;

  comm_hash_buffers_.emplace(comm_name, std::move(buf));

  auto* comm_ptr = comm.get();
  comm->registerPreHook(
      [this, comm_ptr](size_t op_id, const PreHookArgs& args) {
        this->onPreHook(comm_ptr, op_id, args);
      });

  comm->registerPostHook(
      [this, comm_ptr](size_t op_id, const PostHookArgs& args) {
        this->onPostHook(comm_ptr, op_id, args);
      });

  comm->registerGraphReplayHook(
      [this](uint64_t, uint64_t, void*, size_t, std::string_view) {
        this->flushAllHashes();
      });
#endif
}

} // namespace torch::comms
