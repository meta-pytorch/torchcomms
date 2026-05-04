// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <fmt/format.h>
#include <cstring>
#include <system_error>
namespace ncclx::tcpstore::detail {

inline std::error_code lastError() noexcept {
  return std::error_code{errno, std::generic_category()};
}

class NetworkError : public std::runtime_error {
 public:
  explicit NetworkError(const std::string& message)
      : std::runtime_error(message) {}
};

} // namespace ncclx::tcpstore::detail

namespace fmt {

template <>
struct formatter<std::error_code> {
  constexpr decltype(auto) parse(format_parse_context& ctx) const {
    return ctx.begin();
  }

  // This function formats the std::error_code
  template <typename FormatContext>
  auto format(const std::error_code& ec, FormatContext& ctx) const
      -> decltype(ctx.out()) {
    return format_to(ctx.out(), "{}: {}", ec.value(), ec.message());
  }
};

} // namespace fmt
