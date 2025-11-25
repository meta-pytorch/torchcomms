// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/Expected.h>
#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/Ibvcore.h"

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
  inline folly::Expected<std::vector<ibv_wc>, Error> pollCq(int numEntries);

  // Request notification when the next completion is added to this CQ
  folly::Expected<folly::Unit, Error> reqNotifyCq(int solicited_only) const;

 private:
  friend class IbvDevice;

  explicit IbvCq(ibv_cq* cq);

  ibv_cq* cq_{nullptr};
};

// IbvCq inline functions
inline folly::Expected<std::vector<ibv_wc>, Error> IbvCq::pollCq(
    int numEntries) {
  std::vector<ibv_wc> wcs(numEntries);
  int numPolled = cq_->context->ops.poll_cq(cq_, numEntries, wcs.data());
  if (numPolled < 0) {
    wcs.clear();
    return folly::makeUnexpected(
        Error(EINVAL, fmt::format("Call to pollCq() returned {}", numPolled)));
  } else {
    wcs.resize(numPolled);
  }
  return wcs;
}

} // namespace ibverbx
