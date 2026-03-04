// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>

#include "debug.h"

constexpr int kDebugRepeatLogCount = 5;

namespace ncclx::detail {

template <typename Tag>
bool warnFirstNExactImpl(std::size_t n) {
  static std::atomic<std::size_t> counter{0};
  auto const value = counter.load(std::memory_order_relaxed);
  return value < n && counter.fetch_add(1, std::memory_order_relaxed) < n;
}

} // namespace ncclx::detail

#define WARN_IF(cond, ...) \
  if (cond) {              \
    WARN(__VA_ARGS__);     \
  }

#define WARN_FIRST_N(n, ...)                                                  \
  WARN_IF(                                                                    \
      [&] {                                                                   \
        struct ncclx_detail_log_tag {};                                       \
        return ::ncclx::detail::warnFirstNExactImpl<ncclx_detail_log_tag>(n); \
      }(),                                                                    \
      __VA_ARGS__)
