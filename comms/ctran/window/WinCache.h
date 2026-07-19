// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <memory>

#include "comms/ctran/utils/CtranAvlTree.h"
#include "comms/utils/commSpecs.h"

namespace ctran {

struct CtranWin;

// Per-comm cache mapping a registered window's data-buffer range
// [base,base+len) to the owning CtranWin, so a buffer address resolves to its
// containing window. Backed by CtranAvlTree (as RegCache): internally
// synchronized, native interval containment; never dereferences CtranWin, so it
// is standalone unit-testable.
class WinCache {
 public:
  // Caches win's range [base, base+len) and returns the AVL handle for the new
  // entry in *outHdl, so the caller can erase exactly this entry later. A null
  // win returns commInvalidArgument; a zero-length range is skipped
  // (commSuccess, not lookupable). *outHdl is set to nullptr on the zero-length
  // skip and on error. Overlapping ranges are ALLOWED and all cached -- the
  // same buffer may be registered as multiple overlapping windows.
  commResult_t
  insert(const void* base, std::size_t len, CtranWin* win, void** outHdl);

  // Removes the entry identified by hdl (the handle returned by insert).
  // A null handle is a safe no-op, so a never-cached window is safe to erase.
  void erase(void* hdl);

  // Returns a cached window whose range contains [addr, addr+bytes), or
  // nullptr. Among overlapping matches the choice is deterministic (by
  // insertion order and geometry). Non-owning: the cache does not keep the
  // window alive.
  CtranWin* find(const void* addr, std::size_t bytes) const {
    return static_cast<CtranWin*>(tree_->searchVal(addr, bytes));
  }

 private:
  // Address-range -> owning window; internally synchronized, O(log n) interval
  // containment. Held by unique_ptr because CtranAvlTree has a std::mutex (not
  // movable) and WinCache must stay movable (CtranComm holds it by value).
  std::unique_ptr<CtranAvlTree> tree_{std::make_unique<CtranAvlTree>()};
};

} // namespace ctran
