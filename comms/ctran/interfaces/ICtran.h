// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"

/* IB verbs cannot register buffers <= page size because of a bug in
 * the driver. */
#define CTRAN_MIN_REGISTRATION_SIZE (getpagesize() + 1)

class CtranComm;
class CtranMapper;
class CtranGpe;
class CtranAlgo;

namespace ctran {
class Profiler;
} // namespace ctran

/*
 * Abstract class for defining the components and APIs of CTRAN.
 *
 * IMPORTANT: Always include this header (ICtran.h) instead of Ctran.h when
 * working with any CTRAN modules. This follows the mediator design pattern to:
 * 1. Break circular dependencies among CTRAN components
 * 2. Reduce coupling by providing a clean interface for component access
 * 3. Avoid importing the heavyweight Ctran object with all its dependencies
 */
class ICtran {
 public:
  virtual ~ICtran() = default;

  virtual bool isInitialized() const = 0;

  virtual commResult_t commRegister(void* buff, size_t size, void** handle) = 0;
  virtual commResult_t commDeregister(void* handle) = 0;

  virtual void updateOpCount() = 0;
  // OpCount shared with both Ctran and NCCL/RCCL when co-running.
  virtual uint64_t getOpCount() const = 0;
  // OpCount to track calls only to Ctran collectives
  virtual uint64_t getCtranOpCount() const = 0;

  std::unique_ptr<CtranMapper> mapper{nullptr};
  // IMPORTANT: Member destruction order matters! C++ destroys members in
  // reverse order of declaration. algo must be declared BEFORE gpe so that
  // gpe is destroyed first. This ensures the GPE thread is terminated
  // (via ~CtranGpe -> terminate()) before ~CtranAlgo runs and frees CUDA
  // resources that the GPE thread might still be using.
  std::unique_ptr<CtranAlgo> algo{nullptr};
  std::unique_ptr<CtranGpe> gpe{nullptr};
  std::unique_ptr<ctran::Profiler> profiler{nullptr};

  uint64_t numGroupedDefaultOps{0};
};

inline bool ctranIsUsed() {
  return (NCCL_SENDRECV_ALGO == NCCL_SENDRECV_ALGO::ctran);
}

commResult_t ctranInit(CtranComm* comm);
// Check whether the default CTran associated with the comm is initialized.
// If to check a dedicated CTran instance, use ctran->isInitialized() instead.
bool ctranInitialized(CtranComm* comm);

// Finalize any outstanding communication associated with the CtranComm
// instance. Any resource release would be handled when destroying the CtranComm
// instance.
// It should be called before the CtranComm is destroyed and only used in the
// destroy path.
commResult_t ctranFinalize(CtranComm* comm);
