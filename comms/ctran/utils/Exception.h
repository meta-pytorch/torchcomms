// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <exception>
#include <optional>

#include <fmt/ranges.h>

#include "comms/utils/Conversion.h"
#include "comms/utils/commSpecs.h"

namespace ctran::utils {
class Exception : public std::exception {
 public:
  explicit Exception() {
    Exception("", commSuccess);
  }

  explicit Exception(
      const std::string context,
      commResult_t result,
      std::optional<int> rank = std::nullopt,
      std::optional<uint64_t> commHash = std::nullopt,
      std::optional<std::string> desc = std::nullopt)
      : result_(result), rank_(rank), commHash_(commHash), desc_(desc) {
    std::vector<std::string> vec;
    if (rank) {
      vec.emplace_back(fmt::format("rank: {}", *rank));
    }
    if (commHash) {
      vec.emplace_back(fmt::format("commHash: {:x}", *commHash));
    }
    if (desc) {
      vec.emplace_back(fmt::format("desc: {}", *desc));
    }
    msg_ = fmt::format(
        "Exception: {}, {}, result: {} ({})",
        context,
        fmt::join(vec, ", "),
        meta::comms::commCodeToName(result_),
        result_);
  };

  const char* what() const noexcept override {
    return msg_.c_str();
  }

  int rank() const {
    return rank_.value_or(-1);
  }

  uint64_t commHash() const {
    return commHash_.value_or(-1);
  }

  std::string desc() const {
    return desc_.value_or("undefined");
  }

  commResult_t result() const {
    return result_;
  }

 private:
  std::string msg_;
  commResult_t result_{commSuccess};
  std::optional<int> rank_;
  std::optional<uint64_t> commHash_;
  std::optional<std::string> desc_;
};
} // namespace ctran::utils
