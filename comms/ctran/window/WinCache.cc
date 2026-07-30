// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/window/WinCache.h"

#include "comms/utils/logger/LogUtils.h"

namespace ctran {

commResult_t WinCache::insert(
    const void* base,
    std::size_t len,
    CtranWin* win,
    void** outHdl) {
  *outHdl = nullptr;
  if (win == nullptr) {
    return commInvalidArgument;
  }
  // A zero-length range is not lookupable; nothing to cache.
  if (len == 0) {
    return commSuccess;
  }
  // Overlapping ranges are allowed (a buffer may be registered as multiple
  // overlapping windows). CtranAvlTree keeps overlaps in a fallback list;
  // find() returns any containing window -- fine here since ctwin requires
  // symmetric.
  *outHdl = tree_->insert(base, len, win);
  return commSuccess;
}

void WinCache::erase(void* hdl) {
  if (hdl != nullptr) {
    (void)tree_->remove(hdl);
  }
}

} // namespace ctran
