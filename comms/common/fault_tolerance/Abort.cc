// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/common/fault_tolerance/Abort.h"

#ifdef COMMS_FAULT_TOLERANCE_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace comms::fault_tolerance {

std::string_view abortReasonToString(AbortReason reason) {
  switch (reason) {
    case AbortReason::NONE:
      return "none";
    case AbortReason::ABORTED:
      return "aborted";
    case AbortReason::TIMED_OUT:
      return "timed_out";
  }
  return "unknown";
}

Abort::Abort(bool enabled, AbortBehavior behavior) : behavior_(behavior) {
  if (!enabled) {
    return;
  }

#ifdef COMMS_FAULT_TOLERANCE_WITH_CUDA
  const auto status = cudaHostAlloc(
      reinterpret_cast<void**>(&state_),
      sizeof(AbortState),
      cudaHostAllocMapped);
  if (status != cudaSuccess) {
    state_ = nullptr;
    if (status == cudaErrorInsufficientDriver || status == cudaErrorNoDevice ||
        status == cudaErrorInitializationError ||
        status == cudaErrorMemoryAllocation) {
      state_ = new AbortState;
    } else {
      throw std::runtime_error(
          "cudaHostAlloc failed for Abort state: " +
          std::string(cudaGetErrorString(status)));
    }
  } else {
    stateMapped_ = true;
  }
#else
  state_ = new AbortState;
#endif
  state_->abort = encode(AbortReason::NONE);
  state_->timeoutMs = -1;
}

Abort::~Abort() {
  if (state_ == nullptr) {
    return;
  }
#ifdef COMMS_FAULT_TOLERANCE_WITH_CUDA
  if (stateMapped_) {
    (void)cudaFreeHost(state_);
  } else {
    delete state_;
  }
#else
  delete state_;
#endif
}

void Abort::setAbort(AbortReason newReason) {
  if (state_ == nullptr) {
    return;
  }

  markAbort(newReason);
}

bool Abort::isAborted() {
  if (state_ == nullptr) {
    return false;
  }

  if (loadAbortReason() != encode(AbortReason::NONE)) {
    return true;
  }

  if (isTimedOut()) {
    return true;
  }

  return loadAbortReason() != encode(AbortReason::NONE);
}

AbortReason Abort::reason() const {
  if (state_ == nullptr) {
    return AbortReason::NONE;
  }
  return static_cast<AbortReason>(loadAbortReason());
}

bool Abort::isTimeoutActive() const {
  return state_ != nullptr && hasTimeout_.load(std::memory_order_acquire);
}

bool Abort::isTimedOut() {
  if (state_ == nullptr) {
    return false;
  }

  if (loadAbortReason() == encode(AbortReason::TIMED_OUT)) {
    return true;
  }

  if (!hasTimeout_.load(std::memory_order_acquire)) {
    return false;
  }

  auto now = std::chrono::steady_clock::now();
  if (now >= deadline_.load(std::memory_order_acquire)) {
    markAbort(AbortReason::TIMED_OUT);
    return loadAbortReason() == encode(AbortReason::TIMED_OUT);
  }

  return false;
}

std::chrono::milliseconds Abort::getTimeRemaining() {
  if (state_ == nullptr) {
    return std::chrono::milliseconds{-1};
  }

  if (!hasTimeout_.load(std::memory_order_acquire)) {
    return std::chrono::milliseconds{-1};
  }

  auto now = std::chrono::steady_clock::now();
  auto deadline = deadline_.load(std::memory_order_acquire);
  if (now >= deadline) {
    return std::chrono::milliseconds{0};
  }

  return std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
}

void Abort::startTimeout(std::chrono::milliseconds duration) {
  if (state_ == nullptr) {
    return;
  }

  auto deadline = std::chrono::steady_clock::now() + duration;
  deadline_.store(deadline, std::memory_order_release);
  hasTimeout_.store(true, std::memory_order_release);
}

void Abort::cancelTimeout() {
  if (state_ == nullptr) {
    return;
  }

  hasTimeout_.store(false, std::memory_order_release);
}

void Abort::setDefaultTimeout(std::chrono::milliseconds duration) {
  if (state_ == nullptr) {
    return;
  }

  std::atomic_ref<int64_t>{state_->timeoutMs}.store(
      duration.count(), std::memory_order_release);
}

std::optional<std::chrono::milliseconds> Abort::getDefaultTimeout() const {
  if (state_ == nullptr) {
    return std::nullopt;
  }

  const auto v = std::atomic_ref<int64_t>{state_->timeoutMs}.load(
      std::memory_order_acquire);
  if (v < 0) {
    return std::nullopt;
  }
  return std::chrono::milliseconds{v};
}

int Abort::loadAbortReason() const {
  return std::atomic_ref<int>{state_->abort}.load(std::memory_order_acquire);
}

void Abort::markAbort(AbortReason newReason) {
  if (!isValidTerminalReason(newReason)) {
    throw std::invalid_argument("Abort reason must be ABORTED or TIMED_OUT");
  }
  int expected = encode(AbortReason::NONE);
  std::atomic_ref<int>{state_->abort}.compare_exchange_strong(
      expected,
      encode(newReason),
      std::memory_order_acq_rel,
      std::memory_order_acquire);
}

std::shared_ptr<Abort> createAbort(bool enabled, AbortBehavior behavior) {
  if (enabled) {
    return std::make_shared<Abort>(/*enabled=*/true, behavior);
  } else {
    static const std::shared_ptr<Abort> disabled =
        std::make_shared<Abort>(/*enabled=*/false);
    return disabled;
  }
}

} // namespace comms::fault_tolerance
