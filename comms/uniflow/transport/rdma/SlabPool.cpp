// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/rdma/SlabPool.h"

#include <cstring>

#include "comms/uniflow/logging/Logger.h"

namespace uniflow {

SlabPool::SlabPool(
    const SlabPoolConfig& config,
    std::shared_ptr<CudaApi> cudaApi,
    std::shared_ptr<IbvApi> ibvApi,
    const std::vector<NicResources>& nics)
    : config_(config),
      cudaApi_(std::move(cudaApi)),
      ibvApi_(std::move(ibvApi)) {
  if (config_.slabSize == 0 || config_.totalSize < config_.slabSize) {
    throw std::invalid_argument(
        "SlabPool: slabSize must be > 0 and totalSize must be >= slabSize");
  }
  numSlabs_ = config_.totalSize / config_.slabSize;

  auto allocResult =
      cudaApi_->hostAlloc(config_.totalSize, cudaHostAllocMapped);
  if (!allocResult) {
    throw std::runtime_error(
        "SlabPool: cudaHostAlloc failed: " + allocResult.error().toString());
  }
  buffer_ = allocResult.value();

  constexpr int kAccess =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

  mrs_.reserve(nics.size());
  for (const auto& nic : nics) {
    auto mrResult = ibvApi_->regMr(nic.pd, buffer_, config_.totalSize, kAccess);
    if (!mrResult) {
      for (auto* mr : mrs_) {
        ibvApi_->deregMr(mr);
      }
      mrs_.clear();
      cudaApi_->hostFree(buffer_);
      buffer_ = nullptr;
      throw std::runtime_error(
          "SlabPool: ibv_reg_mr failed: " + mrResult.error().toString());
    }
    mrs_.push_back(mrResult.value());
  }

  freeList_.reserve(numSlabs_);
  for (uint16_t i = 0; i < numSlabs_; ++i) {
    freeList_.push_back(i);
  }

  // Initialize atomic bitmaps: 0 = free, 1 = acquired.
  // Pre-mark bits beyond numSlabs_ as acquired so they can never be handed out.
  if (numSlabs_ <= 128) {
    if (numSlabs_ <= 64) {
      low.store(
          numSlabs_ == 64 ? 0ULL : ~((1ULL << numSlabs_) - 1),
          std::memory_order_relaxed);
      high.store(~0ULL, std::memory_order_relaxed);
    } else {
      low.store(0ULL, std::memory_order_relaxed);
      uint64_t highSlabs = numSlabs_ - 64;
      high.store(
          highSlabs == 64 ? 0ULL : ~((1ULL << highSlabs) - 1),
          std::memory_order_relaxed);
    }
  }

  UNIFLOW_LOG_INFO(
      "SlabPool: {} slabs x {} bytes = {} bytes, {} NICs registered",
      numSlabs_,
      config_.slabSize,
      config_.totalSize,
      nics.size());
}

SlabPool::~SlabPool() {
  if (buffer_ == nullptr) {
    return;
  }
  for (auto* mr : mrs_) {
    ibvApi_->deregMr(mr);
  }
  mrs_.clear();
  cudaApi_->hostFree(buffer_);
  buffer_ = nullptr;
}

Result<uint16_t> SlabPool::acquire() {
  if (numSlabs_ <= 128) {
    return acquireByAtomic();
  } else {
    return acquireByMutex();
  }
}

void SlabPool::release(uint16_t slabIdx) {
  if (numSlabs_ <= 128) {
    releaseByAtomic(slabIdx);
  } else {
    releaseByMutex(slabIdx);
  }
}

Result<uint16_t> SlabPool::acquireByMutex() {
  std::lock_guard<std::mutex> lock(mu_);
  if (freeList_.empty()) {
    return Err(ErrCode::ResourceExhausted, "SlabPool: no free slabs");
  }
  uint16_t idx = freeList_.back();
  freeList_.pop_back();
  return idx;
}

void SlabPool::releaseByMutex(uint16_t slabIdx) {
  std::lock_guard<std::mutex> lock(mu_);
  freeList_.push_back(slabIdx);
}

Result<uint16_t> SlabPool::acquireByAtomic() {
  auto try_acquire =
      [](std::atomic<uint64_t>& bitmap) -> std::optional<uint16_t> {
    uint64_t current;
    uint64_t next;
    uint16_t bit_index;

    do {
      current = bitmap.load(std::memory_order_relaxed);
      uint64_t available = ~current;

      if (available == 0) {
        return std::nullopt; // This segment is full
      }

      bit_index = __builtin_ctzll(available);
      next = current | (1ULL << bit_index);

    } while (!bitmap.compare_exchange_weak(
        current, next, std::memory_order_acquire, std::memory_order_relaxed));

    return bit_index;
  };

  if (auto idx = try_acquire(low)) {
    return idx.value();
  } else if (auto idx = try_acquire(high)) {
    return idx.value() + 64;
  } else {
    return Err(ErrCode::ResourceExhausted, "SlabPool: no free slabs");
  }
}

void SlabPool::releaseByAtomic(uint16_t slabIdx) {
  if (slabIdx < 64) {
    low.fetch_and(~(1ULL << slabIdx), std::memory_order_release);
  } else {
    high.fetch_and(~(1ULL << (slabIdx - 64)), std::memory_order_release);
  }
}

void* SlabPool::slabPtr(uint16_t idx) const {
  return static_cast<char*>(buffer_) +
      static_cast<size_t>(idx) * config_.slabSize;
}

uint64_t SlabPool::slabAddr(uint16_t idx) const {
  return reinterpret_cast<uint64_t>(slabPtr(idx));
}

uint32_t SlabPool::lkey(size_t nicIdx) const {
  return mrs_[nicIdx]->lkey;
}

uint32_t SlabPool::rkey(size_t nicIdx) const {
  return mrs_[nicIdx]->rkey;
}

} // namespace uniflow
