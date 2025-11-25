// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

// IbvMr: Memory Region
class IbvMr {
 public:
  ~IbvMr();

  // disable copy constructor
  IbvMr(const IbvMr&) = delete;
  IbvMr& operator=(const IbvMr&) = delete;

  // move constructor
  IbvMr(IbvMr&& other) noexcept;
  IbvMr& operator=(IbvMr&& other) noexcept;

  ibv_mr* mr() const;

 private:
  friend class IbvPd;

  explicit IbvMr(ibv_mr* mr);

  ibv_mr* mr_{nullptr};
};

} // namespace ibverbx
