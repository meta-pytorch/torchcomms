// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/uniflow/Segment.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"

namespace uniflow {

class RdmaSlab;

class CopyEngine {
 public:
  CopyEngine(
      MemoryType memType,
      int deviceId,
      std::optional<cudaStream_t> stream,
      CudaApi* cudaApi,
      CudaDriverApi* cudaDriverApi);

  void copyToSlab(RdmaSlab& slab, const void* src, size_t len);
  void copyFromSlab(void* dst, RdmaSlab& slab, size_t len);
  bool isCopyDone(RdmaSlab& slab) const;
  void resetState(RdmaSlab& slab);

 private:
  MemoryType memType_;
  int deviceId_;
  std::optional<cudaStream_t> stream_;
  CudaApi* cudaApi_;
  CudaDriverApi* cudaDriverApi_;
};

} // namespace uniflow
