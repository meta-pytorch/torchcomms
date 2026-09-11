// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

/**
 * Small Expected helper for ibverbx, which needs value-or-error returns without
 * depending on folly.
 *
 * This intentionally implements only the API shape ibverbx uses:
 * Expected<T, E>, Unexpected<E>, makeUnexpected(E&&), hasValue(), hasError(),
 * explicit operator bool(), checked value()/error() accessors, and the
 * operator* and operator-> value accessors that mirror value(). T and E must be
 * distinct, non-reference, non-void types.
 *
 * Adapted from an existing internal implementation, kept here so that ibverbx
 * and its OSS/CMake build stay self-contained.
 */

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include <variant>

namespace ibverbx::utils {

// Placeholder value for an operation that succeeds without producing one;
// Expected<T, E> cannot represent a void value type. Not errno-specific, so it
// lives here rather than beside ErrnoStatus.
struct Unit {};

namespace detail {

// Reading the wrong alternative is a caller bug, so it aborts in every build.
// std::get would throw std::bad_variant_access, which an opt build turns into
// an unannounced exception and a no-exceptions build turns into a bare
// terminate.
template <std::size_t kIndex, typename Storage>
auto& requireAlternative(Storage& storage, const char* const what) {
  auto* const held = std::get_if<kIndex>(&storage);
  if (held == nullptr) {
    std::fputs(what, stderr);
    std::fputc('\n', stderr);
    std::abort();
  }
  return *held;
}

} // namespace detail

template <typename E>
class Unexpected {
 public:
  explicit Unexpected(const E& error) : error_(error) {}
  explicit Unexpected(E&& error) : error_(std::move(error)) {}

  E& error() & {
    return error_;
  }

  const E& error() const& {
    return error_;
  }

  E&& error() && {
    return std::move(error_);
  }

  const E&& error() const&& {
    return std::move(error_);
  }

 private:
  E error_;
};

template <typename E>
Unexpected<std::decay_t<E>> makeUnexpected(E&& error) {
  return Unexpected<std::decay_t<E>>(std::forward<E>(error));
}

template <typename T, typename E>
class Expected {
  static_assert(
      !std::is_same_v<T, E>,
      "Expected<T, E> requires T and E to be distinct types");
  static_assert(
      !std::is_reference_v<T> && !std::is_reference_v<E>,
      "Expected<T, E> does not support reference value or error types");
  static_assert(
      !std::is_void_v<T> && !std::is_void_v<E>,
      "Expected<T, E> does not support void value or error types");

 public:
  static constexpr const char* kValueOnError =
      "ibverbx::utils::Expected: value() read on an error";
  static constexpr const char* kErrorOnValue =
      "ibverbx::utils::Expected: error() read on a value";

  /* implicit */ Expected(const T& value)
      : storage_(std::in_place_index<0>, value) {}
  /* implicit */ Expected(T&& value)
      : storage_(std::in_place_index<0>, std::move(value)) {}
  /* implicit */ Expected(const Unexpected<E>& unexpected)
      : storage_(std::in_place_index<1>, unexpected.error()) {}
  /* implicit */ Expected(Unexpected<E>&& unexpected)
      : storage_(std::in_place_index<1>, std::move(unexpected).error()) {}

  bool hasValue() const {
    return storage_.index() == 0;
  }

  bool hasError() const {
    return !hasValue();
  }

  explicit operator bool() const {
    return hasValue();
  }

  T& value() & {
    return detail::requireAlternative<0>(storage_, kValueOnError);
  }

  const T& value() const& {
    return detail::requireAlternative<0>(storage_, kValueOnError);
  }

  T&& value() && {
    return std::move(detail::requireAlternative<0>(storage_, kValueOnError));
  }

  const T&& value() const&& {
    return std::move(detail::requireAlternative<0>(storage_, kValueOnError));
  }

  T& operator*() & {
    return value();
  }

  const T& operator*() const& {
    return value();
  }

  T&& operator*() && {
    return std::move(*this).value();
  }

  const T&& operator*() const&& {
    return std::move(*this).value();
  }

  T* operator->() {
    return &value();
  }

  const T* operator->() const {
    return &value();
  }

  E& error() & {
    return detail::requireAlternative<1>(storage_, kErrorOnValue);
  }

  const E& error() const& {
    return detail::requireAlternative<1>(storage_, kErrorOnValue);
  }

  E&& error() && {
    return std::move(detail::requireAlternative<1>(storage_, kErrorOnValue));
  }

  const E&& error() const&& {
    return std::move(detail::requireAlternative<1>(storage_, kErrorOnValue));
  }

 private:
  std::variant<T, E> storage_;
};

} // namespace ibverbx::utils
