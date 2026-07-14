// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <mutex>
#include <utility>

// A one-shot cleanup token for a persistent request. The cleanup closure
// releases the request's pooled GpeKernelSync (pipeSync) + scoped registration
// (via destroyPersistentRequest); it does NOT free/delete the request object
// itself. The request is deleted by its external owner (ncclx::pFree, tests,
// bench, or the graph-destroy callback). The token must run EXACTLY ONCE, no
// matter which teardown path fires first (eager preq free, CUDA graph-destroy
// callback, or comm cleanup before CtranGpe::terminate()).
//
// The token is heap-managed via shared_ptr and OUTLIVES both the request object
// and the comm: a late graph-destroy callback (after the comm already drained
// and freed the request) must still be able to call run() safely -- it just
// no-ops. std::call_once gives cross-thread "exactly once" since the
// graph-destroy callback may run on a different thread than the comm/preq free.
struct PersistentCleanup {
  std::once_flag flag_;
  std::function<void()> fn_;

  explicit PersistentCleanup(std::function<void()> fn) : fn_(std::move(fn)) {}

  void run() {
    std::call_once(flag_, [this] { fn_(); });
  }
};
