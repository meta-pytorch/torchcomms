// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/rdma/RdmaSlabPool.h"

#include "comms/uniflow/logging/Logger.h"
#include "comms/uniflow/transport/rdma/RdmaResources.h"

namespace uniflow {

RdmaSlabPool::RdmaSlabPool(
    const RdmaSlabPoolConfig& config,
    std::shared_ptr<CudaApi> cudaApi,
    std::shared_ptr<IbvApi> ibvApi,
    std::shared_ptr<std::vector<NicResources>> nics)
    : config_(config),
      cudaApi_(std::move(cudaApi)),
      ibvApi_(std::move(ibvApi)),
      nics_(std::move(nics)) {
  if (config_.slabSize == 0 || config_.slabNum == 0) {
    throw std::invalid_argument(
        "RdmaSlabPool: slabSize must be > 0 and totalSize must be >= slabSize");
  }

  // Per-slab state buffer (one uint32_t per slab) for async copy tracking.
  const size_t stateSize = config_.slabNum * sizeof(uint64_t);
  const size_t bufferSize = config_.slabNum * config_.slabSize;
  const size_t totalSize = bufferSize + stateSize;

  auto slabAllocResult = cudaApi_->hostAlloc(totalSize, cudaHostAllocMapped);
  if (!slabAllocResult) {
    throw std::runtime_error(
        "RdmaSlabPool: slab cudaHostAlloc failed: " +
        slabAllocResult.error().toString());
  }
  slabBuffer_ = slabAllocResult.value();
  stateBuffer_ = static_cast<char*>(slabBuffer_) + bufferSize;
  std::memset(stateBuffer_, 0, stateSize);

  auto devPtrResult = cudaApi_->hostGetDevicePointer(slabBuffer_);
  if (!devPtrResult) {
    throw std::runtime_error(
        "RdmaSlabPool: state cudaHostGetDevicePointer failed: " +
        devPtrResult.error().toString());
  }
  slabDeviceBufferPtr_ = reinterpret_cast<uintptr_t>(devPtrResult.value());
  stateDeviceBufferPtr_ = slabDeviceBufferPtr_ + bufferSize;

  constexpr int kAccess =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

  slabMrs_.reserve(nics_->size());
  for (const auto& nic : *nics_) {
    auto mrResult = ibvApi_->regMr(nic.pd, slabBuffer_, totalSize, kAccess);
    if (!mrResult) {
      for (auto* mr : slabMrs_) {
        ibvApi_->deregMr(mr);
      }
      slabMrs_.clear();
      cudaApi_->hostFree(slabBuffer_);
      slabBuffer_ = nullptr;
      throw std::runtime_error(
          "RdmaSlabPool: ibv_reg_mr failed: " + mrResult.error().toString());
    }
    slabMrs_.push_back(mrResult.value());
  }

  if (config_.slabNum <= 128) {
    if (config_.slabNum <= 64) {
      low.store(
          config_.slabNum == 64 ? 0ULL : ~((1ULL << config_.slabNum) - 1),
          std::memory_order_relaxed);
      high.store(~0ULL, std::memory_order_relaxed);
    } else {
      uint64_t highSlabs = config_.slabNum - 64;
      high.store(
          highSlabs == 64 ? 0ULL : ~((1ULL << highSlabs) - 1),
          std::memory_order_relaxed);
    }
  } else {
    freeList_.reserve(config_.slabNum);
    for (uint16_t i = 0; i < config_.slabNum; ++i) {
      freeList_.push_back(i);
    }
  }

  UNIFLOW_LOG_INFO(
      "RdmaSlabPool: {} slabs x {} bytes = {} bytes, {} NICs registered",
      config_.slabNum,
      config_.slabSize,
      totalSize,
      nics_->size());
}

RdmaSlabPool::~RdmaSlabPool() {
  for (auto* mr : slabMrs_) {
    ibvApi_->deregMr(mr);
  }
  slabMrs_.clear();
  if (slabBuffer_) {
    cudaApi_->hostFree(slabBuffer_);
    slabBuffer_ = nullptr;
    stateBuffer_ = nullptr;
  }
}

Result<uint16_t> RdmaSlabPool::acquire() {
  if (config_.slabNum <= 128) {
    return acquireByAtomic();
  } else {
    return acquireByMutex();
  }
}

Result<std::vector<uint16_t>> RdmaSlabPool::acquire(uint32_t count) {
  if (config_.slabNum <= 128) {
    return acquireByAtomic(count);
  } else {
    return acquireByMutex(count);
  }
}

void RdmaSlabPool::release(uint16_t slabIdx) {
  if (config_.slabNum <= 128) {
    releaseByAtomic(slabIdx);
  } else {
    releaseByMutex(slabIdx);
  }
}

void RdmaSlabPool::release(const std::vector<uint16_t>& slabIndices) {
  if (config_.slabNum <= 128) {
    releaseByAtomic(slabIndices);
  } else {
    releaseByMutex(slabIndices);
  }
}

Result<uint16_t> RdmaSlabPool::acquireByMutex() {
  std::lock_guard<std::mutex> lock(mu_);
  if (freeList_.empty()) {
    return Err(ErrCode::ResourceExhausted, "RdmaSlabPool: no free slabs");
  }
  uint16_t idx = freeList_.back();
  freeList_.pop_back();
  return idx;
}

Result<std::vector<uint16_t>> RdmaSlabPool::acquireByMutex(uint32_t count) {
  std::lock_guard<std::mutex> lock(mu_);
  if (freeList_.size() < count) {
    return Err(
        ErrCode::ResourceExhausted,
        "RdmaSlabPool: need " + std::to_string(count) + " slabs, have " +
            std::to_string(freeList_.size()));
  }
  std::vector<uint16_t> acquired(freeList_.end() - count, freeList_.end());
  freeList_.resize(freeList_.size() - count);
  return acquired;
}

void RdmaSlabPool::releaseByMutex(uint16_t slabIdx) {
  std::lock_guard<std::mutex> lock(mu_);
  freeList_.push_back(slabIdx);
}

void RdmaSlabPool::releaseByMutex(const std::vector<uint16_t>& slabIndices) {
  std::lock_guard<std::mutex> lock(mu_);
  freeList_.insert(freeList_.end(), slabIndices.begin(), slabIndices.end());
}

Result<uint16_t> RdmaSlabPool::acquireByAtomic() {
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

  if (auto lIdx = try_acquire(low)) {
    return lIdx.value();
  } else if (auto hIdx = try_acquire(high)) {
    return hIdx.value() + 64;
  } else {
    return Err(ErrCode::ResourceExhausted, "RdmaSlabPool: no free slabs");
  }
}

Result<std::vector<uint16_t>> RdmaSlabPool::acquireByAtomic(uint32_t count) {
  std::vector<uint16_t> acquired;
  acquired.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    auto r = acquireByAtomic();
    if (!r) {
      releaseByAtomic(acquired);
      return Err(
          ErrCode::ResourceExhausted, "RdmaSlabPool: insufficient slabs");
    }
    acquired.push_back(r.value());
  }
  return acquired;
}

void RdmaSlabPool::releaseByAtomic(uint16_t slabIdx) {
  if (slabIdx < 64) {
    low.fetch_and(~(1ULL << slabIdx), std::memory_order_release);
  } else {
    high.fetch_and(~(1ULL << (slabIdx - 64)), std::memory_order_release);
  }
}

void RdmaSlabPool::releaseByAtomic(const std::vector<uint16_t>& slabIndices) {
  uint64_t lowMask = 0;
  uint64_t highMask = 0;
  for (auto idx : slabIndices) {
    if (idx < 64) {
      lowMask |= (1ULL << idx);
    } else {
      highMask |= (1ULL << (idx - 64));
    }
  }
  if (lowMask != 0) {
    low.fetch_and(~lowMask, std::memory_order_release);
  }
  if (highMask != 0) {
    high.fetch_and(~highMask, std::memory_order_release);
  }
}

void* RdmaSlabPool::slabPtr(uint16_t idx) const {
  return static_cast<char*>(slabBuffer_) +
      static_cast<size_t>(idx) * config_.slabSize;
}

uint64_t RdmaSlabPool::slabAddr(uint16_t idx) const {
  return reinterpret_cast<uint64_t>(slabPtr(idx));
}

uint32_t RdmaSlabPool::slabLkey(size_t nicIdx) const {
  return slabMrs_[nicIdx]->lkey;
}

uint32_t RdmaSlabPool::slabRkey(size_t nicIdx) const {
  return slabMrs_[nicIdx]->rkey;
}

void* RdmaSlabPool::statePtr(uint16_t idx) const {
  return static_cast<char*>(stateBuffer_) + idx * sizeof(uint64_t);
}

uint64_t RdmaSlabPool::stateAddr(uint16_t idx) const {
  return reinterpret_cast<uint64_t>(statePtr(idx));
}

uint64_t RdmaSlabPool::stateDeviceAddr(uint16_t idx) const {
  return stateDeviceBufferPtr_ + idx * sizeof(uint64_t);
}

} // namespace uniflow
