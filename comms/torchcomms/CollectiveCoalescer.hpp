// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>
#include <vector>

#include "comms/torchcomms/TorchWork.hpp"

namespace torch {
namespace comms {

class CollectiveCoalescer {
 public:
  virtual ~CollectiveCoalescer() = default;

  void startCoalescing() {
    TORCH_CHECK(
        !is_coalescing_active(),
        "start called while coalescing is already active");

    coalescing_active_ = true;
    coalesced_tensors_.clear();

    onCoalescingStart();
  }

  c10::intrusive_ptr<TorchWork> endCoalescing() {
    TORCH_CHECK(
        is_coalescing_active(), "end called without active coalescing block");

    onCoalescingEnd();

    c10::intrusive_ptr<TorchWork> work{nullptr};
    if (has_coalesced_tensors()) {
      work = createCoalescedWork(coalesced_tensors_);
    }

    coalescing_active_ = false;
    coalesced_tensors_.clear();

    return work;
  }

  bool is_coalescing_active() const {
    return coalescing_active_;
  }

  bool has_coalesced_tensors() const {
    return !coalesced_tensors_.empty();
  }

  void register_coalesced_tensors(
      std::initializer_list<std::reference_wrapper<const at::Tensor>> tensors) {
    TORCH_CHECK(
        is_coalescing_active(),
        "register_coalesced_tensors called without active coalescing block");
    coalesced_tensors_.insert(
        coalesced_tensors_.end(), tensors.begin(), tensors.end());
  }

 protected:
  /**
   * Backend-specific hook called when coalescing starts.
   * Override to implement backend-specific logic (e.g., ncclGroupStart).
   */
  virtual void onCoalescingStart() {}

  /**
   * Backend-specific hook called when coalescing ends.
   * Override to implement backend-specific logic (e.g., ncclGroupEnd).
   */
  virtual void onCoalescingEnd() {}

  /**
   * Create a work handle for the coalesced operations.
   * Override to create backend-specific work objects.
   */
  virtual c10::intrusive_ptr<TorchWork> createCoalescedWork(
      const std::vector<at::Tensor>& /* tensors */) {
    throw std::runtime_error("Backend does not support coalesced operations");
  }

  bool coalescing_active_{false};
  std::vector<at::Tensor> coalesced_tensors_;
};

} // namespace comms
} // namespace torch
