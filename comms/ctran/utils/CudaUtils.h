// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/Expected.h>
#include <folly/String.h>
#include <string>

namespace ctran::utils {

class BusId {
 public:
  static BusId makeFrom(const int cudaDev);
  static BusId makeFrom(const std::string& busIdStr);
  static BusId makeFrom(const int64_t& busId);

  std::string toStr() noexcept;
  int64_t toInt64();

  bool operator==(const BusId& other) const = default;

 private:
  // reserve enough space for the busId string
  std::string busId_{"00000000:00:00.0\0", 17};

  // busId must be always initialized to a valid value so users must use
  // factories not constructor
  BusId(){};
};

folly::Expected<int, std::string> getCudaArch(int cudaDev);

} // namespace ctran::utils
