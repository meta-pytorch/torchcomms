// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

#include "comms/uniflow/Segment.h"
#include "comms/uniflow/drivers/DeviceAdapter.h"

namespace uniflow {

class RdmaSlab;

/// Borrows the caller's stream and buffers. Completion or a successful exact
/// stream drain must precede releasing a slab or the client segment.
class CopyEngine {
 public:
  CopyEngine(
      MemoryType memType,
      int deviceId,
      std::optional<DeviceStream> stream,
      std::shared_ptr<DeviceAdapter> deviceAdapter);
  ~CopyEngine();

  CopyEngine(const CopyEngine&) = delete;
  CopyEngine& operator=(const CopyEngine&) = delete;
  CopyEngine(CopyEngine&&) = delete;
  CopyEngine& operator=(CopyEngine&&) = delete;

  /// Checks capabilities before any copy or receive handshake is enqueued.
  Status validate(const RdmaSlab& slab) const;

  Status copyToSlab(RdmaSlab& slab, const void* src, size_t len);
  Status copyFromSlab(void* dst, RdmaSlab& slab, size_t len);
  Result<bool> isCopyDone(const RdmaSlab& slab);
  Status resetState(RdmaSlab& slab);

  /// Keeps pending completion owners on failure; failure is not quiescence.
  Status drain();

 private:
  struct PendingCopy {
    bool mappedMarker{false};
    std::unique_ptr<DeviceCompletion> completion;
  };

  Status copy(
      RdmaSlab& slab,
      void* dst,
      const void* src,
      size_t len,
      DeviceCopyKind kind);
  Status failAndDrain(Err error);

  const MemoryType memType_;
  const int deviceId_;
  const std::optional<DeviceStream> stream_;
  const std::shared_ptr<DeviceAdapter> deviceAdapter_;
  std::unordered_map<uint64_t, PendingCopy> pendingCopies_;
};

} // namespace uniflow
