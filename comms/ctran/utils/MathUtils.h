// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <type_traits>

namespace ctran::utils {

template <typename T>
constexpr bool isPowerOfTwo(T value) {
  static_assert(std::is_integral_v<T>);
  return value > T{0} && (value & (value - T{1})) == T{0};
}

} // namespace ctran::utils
