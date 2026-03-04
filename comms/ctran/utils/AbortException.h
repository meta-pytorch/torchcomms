// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <optional>
#include <string>

#include "comms/ctran/utils/Exception.h"

namespace ctran::utils {

class AbortException : public Exception {
 public:
  explicit AbortException(
      const std::string context,
      bool retriable = false,
      std::optional<int> rank = std::nullopt,
      std::optional<uint64_t> commHash = std::nullopt,
      std::optional<std::string> desc = std::nullopt);

  bool isRetriable() const;

 private:
  bool retriable_;
};

} // namespace ctran::utils
