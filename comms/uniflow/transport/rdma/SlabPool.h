// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/ibverbs/IbvApi.h"
#include "comms/uniflow/transport/rdma/RdmaTransport.h"

namespace uniflow {

struct SlabPoolConfig {
  size_t totalSize{64 * 1024 * 1024};
  size_t slabSize{512 * 1024};
};

/// Factory-level shared pool of host-pinned staging buffers for copy-based
/// send/recv. All slabs are shared across all connections — no per-connection
/// reservation. Slabs are dynamically acquired per transfer and returned
/// immediately upon completion.
///
/// The pool is a contiguous block of mmap+mlock host-pinned memory, registered
/// with every NIC's PD via ibv_reg_mr. All slabs share the same lkey/rkey per
/// NIC, eliminating per-slab registration overhead.
///
/// Thread safety: acquire() and release() are mutex-protected. The mutex is
/// not on the hot path when all callers are serialized on a single EventBase
/// thread, but is present for correctness if multiple EventBases share a pool.
class SlabPool {
 public:
  SlabPool(
      const SlabPoolConfig& config,
      std::shared_ptr<CudaApi> cudaApi,
      std::shared_ptr<IbvApi> ibvApi,
      const std::vector<NicResources>& nics);

  ~SlabPool();

  SlabPool(const SlabPool&) = delete;
  SlabPool& operator=(const SlabPool&) = delete;

  Result<uint16_t> acquire();
  void release(uint16_t slabIdx);

  void* slabPtr(uint16_t idx) const;
  uint64_t slabAddr(uint16_t idx) const;
  uint32_t lkey(size_t nicIdx) const;
  uint32_t rkey(size_t nicIdx) const;

  size_t slabSize() const {
    return config_.slabSize;
  }
  size_t numSlabs() const {
    return numSlabs_;
  }

 private:
  SlabPoolConfig config_;
  std::shared_ptr<CudaApi> cudaApi_;
  std::shared_ptr<IbvApi> ibvApi_;
  void* buffer_{nullptr};
  size_t numSlabs_{0};
  std::vector<ibv_mr*> mrs_;

  std::mutex mu_;
  std::vector<uint16_t> freeList_;
  Result<uint16_t> acquireByMutex();
  void releaseByMutex(uint16_t slabIdx);

  std::atomic<uint64_t> low{}; // Bits 0-63
  std::atomic<uint64_t> high{}; // Bits 64-127
  Result<uint16_t> acquireByAtomic();
  void releaseByAtomic(uint16_t slabIdx);
};

} // namespace uniflow
