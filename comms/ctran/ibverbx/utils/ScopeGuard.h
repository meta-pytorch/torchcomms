// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <type_traits>
#include <utility>

namespace ibverbx::utils {

// Minimal header-only RAII scope guard (STL-only). Adapted from an existing
// internal implementation so that ibverbx stays self-contained.
template <typename F>
class ScopeGuard {
 public:
  explicit ScopeGuard(F&& fn) : fn_(std::move(fn)) {}
  explicit ScopeGuard(const F& fn) : fn_(fn) {}

  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;

  ScopeGuard(ScopeGuard&& other) noexcept(
      std::is_nothrow_move_constructible_v<F>)
      : fn_(std::move(other.fn_)), active_(other.active_) {
    other.dismiss();
  }

  ScopeGuard& operator=(ScopeGuard&&) = delete;

  ~ScopeGuard() noexcept(noexcept(std::declval<F&>()())) {
    if (active_) {
      fn_();
    }
  }

  void dismiss() noexcept {
    active_ = false;
  }

 private:
  F fn_;
  bool active_{true};
};

template <typename F>
ScopeGuard<std::decay_t<F>> makeScopeGuard(F&& fn) {
  return ScopeGuard<std::decay_t<F>>(std::forward<F>(fn));
}

} // namespace ibverbx::utils
