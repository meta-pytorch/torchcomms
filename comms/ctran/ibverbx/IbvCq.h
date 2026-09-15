// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/device/structs.h"

namespace ibverbx {

// Ibv CompletionQueue(CQ)
class IbvCq {
 public:
  IbvCq() = default;
  ~IbvCq();

  // disable copy constructor
  IbvCq(const IbvCq&) = delete;
  IbvCq& operator=(const IbvCq&) = delete;

  // move constructor
  IbvCq(IbvCq&& other) noexcept;
  IbvCq& operator=(IbvCq&& other) noexcept;

  ibv_cq* cq() const;
  int32_t getDeviceId() const;

  // create device cq, map cq buffer to GPU
  Expected<struct device_cq> getDeviceCq() const noexcept;

  inline Expected<std::vector<ibv_wc>> pollCq(int numEntries);

  // Request notification when the next completion is added to this CQ
  Status reqNotifyCq(int solicited_only) const;

 private:
  friend class IbvDevice;

  IbvCq(ibv_cq* cq, int32_t deviceId);

  ibv_cq* cq_{nullptr};
  int32_t deviceId_{-1}; // The IbvDevice's DeviceId that corresponds to this
                         // Completion Queue (CQ)
};

// IbvCq inline functions
inline Expected<std::vector<ibv_wc>> IbvCq::pollCq(int numEntries) {
  std::vector<ibv_wc> wcs(numEntries);
  int numPolled = cq_->context->ops.poll_cq(cq_, numEntries, wcs.data());
  if (numPolled < 0) {
    wcs.clear();
    return makeUnexpected(
        Error(EINVAL, fmt::format("Call to pollCq() returned {}", numPolled)));
  } else {
    wcs.resize(numPolled);
  }
  return wcs;
}

} // namespace ibverbx
