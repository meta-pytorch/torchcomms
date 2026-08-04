// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <optional>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "comms/utils/commSpecs.h"
#include "comms/utils/memtrace/Types.h"

namespace ncclx::memory {

class SlabAllocator {
 public:
  SlabAllocator();
  ~SlabAllocator();

  commResult_t cuCallocAsync(
      void** ptr,
      size_t numBytes,
      cudaStream_t stream,
      const meta::comms::memtrace::MemCallsite& callsite,
      const CommLogData* logMetaData = nullptr);

  commResult_t cuMalloc(
      void** ptr,
      size_t numBytes,
      const meta::comms::memtrace::MemCallsite& callsite,
      const CommLogData* logMetaData = nullptr,
      CUmemGenericAllocationHandle* handlep = nullptr,
      size_t* newSlabSize = nullptr);

  size_t getUsedMem() const {
    return totalMemAllocated_;
  }

 private:
  commResult_t computeCumemGranualirity();
  commResult_t allocateMem(
      void** ptr,
      size_t numBytes,
      const meta::comms::memtrace::MemCallsite& callsite,
      const CommLogData* logMetaData,
      CUmemGenericAllocationHandle* handlep,
      size_t* newSlabSize);
  // pointer to start of each slab, used to free memory during communicator
  // destroy
  std::vector<void*> slabPtrs_;
  // CommLogData retained from the allocation path so the destructor's free can
  // be attributed to the same communicator (commHash) as its allocation.
  std::optional<CommLogData> logMetaData_;
  size_t freeSize_{0};
  void* startPtr_{nullptr};
  CUmemGenericAllocationHandle slabHandle_{};
  size_t granularity_{0};
  CUmemAllocationProp prop_{};
  size_t totalMemAllocated_{0};
};
} // namespace ncclx::memory
