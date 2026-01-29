// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

// IbvAh: Address Handle
// Used by DC transport to route messages to remote DCTs
class IbvAh {
 public:
  IbvAh() = default;
  ~IbvAh();

  // disable copy constructor
  IbvAh(const IbvAh&) = delete;
  IbvAh& operator=(const IbvAh&) = delete;

  // move constructor
  IbvAh(IbvAh&& other) noexcept;
  IbvAh& operator=(IbvAh&& other) noexcept;

  ibv_ah* ah() const;

 private:
  friend class IbvPd;

  explicit IbvAh(ibv_ah* ah);

  ibv_ah* ah_{nullptr};
};

} // namespace ibverbx
