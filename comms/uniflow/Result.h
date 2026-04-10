// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <future>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

namespace uniflow {

#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)

#define CHECK_THROW_EXCEPTION(cond, exception)                              \
  do {                                                                      \
    if (!(cond)) {                                                          \
      throw exception(                                                      \
          "Check failed: " #cond ", in " __FILE__ ":" STRINGIFY(__LINE__)); \
    }                                                                       \
  } while (0)

#define CHECK_THROW_ERROR(expr)                         \
  do {                                                  \
    auto _res = (expr);                                 \
    if (!_res) {                                        \
      throw std::runtime_error(_res.error().message()); \
    }                                                   \
  } while (0)

#define CHECK_RETURN(res)            \
  do {                               \
    if (!(res)) {                    \
      return std::move(res).error(); \
    }                                \
  } while (0)

#define CHECK_EXPR(expr)              \
  do {                                \
    auto _res = (expr);               \
    if (!(_res)) {                    \
      return std::move(_res).error(); \
    }                                 \
  } while (0)

#define CHECK_SET_PROMISE(promise, expr, ...)             \
  do {                                                    \
    auto _check_res = (expr);                             \
    if (!_check_res) {                                    \
      __VA_ARGS__;                                        \
      (promise).set_value(std::move(_check_res).error()); \
      return;                                             \
    }                                                     \
  } while (0)

// X macro for error codes - add new codes here
#define UNIFLOW_ERROR_CODES(X) \
  X(NotImplemented)            \
  X(DriverError)               \
  X(TopologyDisconnect)        \
  X(InvalidArgument)           \
  X(NotConnected)              \
  X(TransportError)            \
  X(ConnectionFailed)          \
  X(MemoryRegistrationFailed)  \
  X(Timeout)                   \
  X(ResourceExhausted)

/// Error codes for UniFlow operations
enum class ErrCode : uint32_t {
#define X(name) name,
  UNIFLOW_ERROR_CODES(X)
#undef X
};

/// Returns a human-readable string for the given error code
inline const char* errorCodeToString(ErrCode code) noexcept {
  switch (code) {
#define X(name)       \
  case ErrCode::name: \
    return #name;
    UNIFLOW_ERROR_CODES(X)
#undef X
    default:
      return "Unknown";
  }
}

/// Represents an error with a code and optional message
class Err {
 public:
  explicit Err(ErrCode code) noexcept : code_(code) {}

  Err(ErrCode code, std::string message) noexcept
      : code_(code), message_(std::move(message)) {}

  /// Returns the error code
  ErrCode code() const noexcept {
    return code_;
  }

  /// Returns the error message (may be empty)
  const std::string& message() const noexcept {
    return message_;
  }

  /// Returns a formatted string representation of this error
  std::string toString() const {
    if (message_.empty()) {
      return errorCodeToString(code_);
    }
    return std::string(errorCodeToString(code_)) + ": " + message_;
  }

 private:
  ErrCode code_;
  std::string message_;
};

/// A Result type that holds either a value of type T or an Error.
/// This provides explicit error handling without exceptions.
template <typename T>
class Result {
 public:
  /// Construct a successful Result with a value
  /* implicit */ Result(T value) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : data_(std::move(value)) {}

  /// Construct a successful Result from a convertible type U.
  /// This enables implicit conversion from unique_ptr<Derived> to
  /// Result<unique_ptr<Base>> when Derived* is convertible to Base*.
  template <
      typename U,
      std::enable_if_t<
          std::is_constructible_v<T, U&&> &&
              !std::is_same_v<std::decay_t<U>, T> &&
              !std::is_same_v<std::decay_t<U>, Result<T>> &&
              !std::is_same_v<std::decay_t<U>, Err> &&
              !std::is_same_v<std::decay_t<U>, ErrCode>,
          int> = 0>
  /* implicit */ Result(U&& value) noexcept(
      std::is_nothrow_constructible_v<T, U&&>)
      : data_(T(std::forward<U>(value))) {}

  /// Construct a failed Result with an error
  /* implicit */ Result(Err error) noexcept : data_(std::move(error)) {}

  /// Construct a failed Result with an error code
  /* implicit */ Result(ErrCode code) noexcept : data_(Err(code)) {}

  /// Returns true if the Result contains a value
  bool hasValue() const noexcept {
    return std::holds_alternative<T>(data_);
  }

  /// Returns true if the Result contains an error
  bool hasError() const noexcept {
    return std::holds_alternative<Err>(data_);
  }

  /// Alias for hasValue()
  explicit operator bool() const noexcept {
    return hasValue();
  }

  /// Returns a reference to the value. Undefined behavior if hasValue() is
  /// false.
  T& value() & {
    return std::get<T>(data_);
  }

  /// Returns a const reference to the value
  const T& value() const& {
    return std::get<T>(data_);
  }

  /// Returns an rvalue reference to the value
  T&& value() && {
    return std::get<T>(std::move(data_));
  }

  /// Returns a reference to the error. Undefined behavior if hasError() is
  /// false.
  Err& error() & {
    return std::get<Err>(data_);
  }

  /// Returns a const reference to the error
  const Err& error() const& {
    return std::get<Err>(data_);
  }

  /// Arrow operator for convenient member access on the held value.
  /// Undefined behavior if hasValue() is false.
  T* operator->() {
    return &std::get<T>(data_);
  }

  const T* operator->() const {
    return &std::get<T>(data_);
  }

  /// Returns the value or a default if this is an error
  template <typename U>
  T valueOr(U&& defaultValue) const& {
    if (hasValue()) {
      return value();
    }
    return static_cast<T>(std::forward<U>(defaultValue));
  }

  /// Returns the value or a default if this is an error (move version)
  template <typename U>
  T valueOr(U&& defaultValue) && {
    if (hasValue()) {
      return std::move(*this).value();
    }
    return static_cast<T>(std::forward<U>(defaultValue));
  }

 private:
  std::variant<T, Err> data_;
};

/// Specialization of Result for void
template <>
class Result<void> {
 public:
  /// Construct a successful Result<void>
  Result() noexcept : error_(std::nullopt) {}

  /// Construct a failed Result<void> with an error
  /* implicit */ Result(Err error) noexcept : error_(std::move(error)) {}

  /// Construct a failed Status with an error code
  /* implicit */ Result(ErrCode code) noexcept : error_(Err(code)) {}

  /// Returns true if the Result represents success
  bool hasValue() const noexcept {
    return !error_.has_value();
  }

  /// Returns true if the Result represents an error
  bool hasError() const noexcept {
    return error_.has_value();
  }

  /// Alias for hasValue()
  explicit operator bool() const noexcept {
    return hasValue();
  }

  /// Returns a reference to the error. Undefined behavior if hasError() is
  /// false.
  Err& error() & {
    return *error_;
  }

  /// Returns a const reference to the error
  const Err& error() const& {
    return *error_;
  }

 private:
  std::optional<Err> error_;
};

/// Alias for Result<void> - represents success or an error
using Status = Result<void>;

/// Helper function to create a successful Status
inline Status Ok() noexcept {
  return Status();
}

template <typename T>
std::future<std::decay_t<T>> make_ready_future(T&& value) {
  std::promise<std::decay_t<T>> p;
  p.set_value(std::forward<T>(value));
  return p.get_future();
}

inline std::future<void> make_ready_future() {
  std::promise<void> p;
  p.set_value();
  return p.get_future();
}

} // namespace uniflow
