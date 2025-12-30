// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>

#include "comms/torchcomms/TorchCommBackend.hpp"
#include "comms/torchcomms/TorchWork.hpp"

struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

namespace torch {
namespace comms {

/**
 * Base class for NCCL coalescing context management.
 *
 * This base class provides common members and accessors for RAII-based
 * coalescing context management in CUDA-based communication backends
 * (NCCL, NCCLX). Derived classes implement backend-specific
 * constructor/destructor logic.
 */
template <typename TComm, typename TWork>
class NCCLCoalescingContextBase {
 public:
  NCCLCoalescingContextBase(const NCCLCoalescingContextBase&) = delete;
  NCCLCoalescingContextBase& operator=(const NCCLCoalescingContextBase&) =
      delete;
  NCCLCoalescingContextBase(NCCLCoalescingContextBase&&) = delete;
  NCCLCoalescingContextBase& operator=(NCCLCoalescingContextBase&&) = delete;

  c10::intrusive_ptr<TorchWork> get_work() {
    return work_;
  }

  cudaStream_t get_stream() const {
    return stream_;
  }

 protected:
  NCCLCoalescingContextBase() {
    static_assert(
        std::is_base_of<TorchCommBackend, TComm>::value,
        "TComm must inherit from TorchCommBackend");
    static_assert(
        std::is_base_of<TorchWork, TWork>::value,
        "TWork must inherit from TorchWork");
  }
  ~NCCLCoalescingContextBase() = default;

  cudaStream_t stream_{nullptr};
  bool is_coalescing_{false};
  c10::intrusive_ptr<TWork> work_{nullptr};
  TComm* comm_{nullptr};
};

} // namespace comms
} // namespace torch
