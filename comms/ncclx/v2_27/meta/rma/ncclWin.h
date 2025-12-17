// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/Synchronized.h>
#include <unordered_map>

#include "comms/ctran/window/CtranWin.h"

struct ncclWindow;

struct ncclWin {
  // communicator associated with this window
  ncclComm_t comm;

  // implementation of ncclWin on top of Ctran
  ctran::CtranWin* ctranWindow;
};

// Thread-safe wrapper for ncclWindow -> ncclWin mapping
//
// Design Note: We use a global map because ncclWindow_vidmem (in v2_28) is a
// device-side structure that cannot store host pointers or be accessed by the
// host. The map is per-process, which is correct for NCCL's
// one-process-per-rank model.
//
// Thread Safety: All methods are thread-safe using folly::Synchronized with
// read-write locks. Multiple threads can safely call find() concurrently
// (shared read lock), while insert() and erase() acquire exclusive write locks.
//
// Lifecycle:
// - insert(): Called during window allocation (ncclWinAllocate/ncclWinRegister)
//             to register the handle -> ncclWin mapping
// - find():   Called during RMA operations to retrieve the ncclWin pointer
//             Returns nullptr if the handle is invalid or has been freed
// - erase():  Called during window deallocation (ncclWinFree) to remove the
//             mapping and allow cleanup
//
// Usage Pattern:
//   ncclWindow_t handle = allocate_window(...);
//   ncclWinMap().insert(handle, ncclWin_ptr);  // On allocation
//   ...
//   ncclWin* ptr = ncclWinMap().find(handle);  // During RMA ops (check for
//   nullptr!)
//   ...
//   ncclWinMap().erase(handle);                // On deallocation
//   delete ncclWin_ptr;
//   delete handle;
class NcclWinMap {
 public:
  static NcclWinMap& instance() {
    static NcclWinMap map;
    return map;
  }

  void insert(ncclWindow* handle, ncclWin* win) {
    auto locked = map_.wlock();
    (*locked)[handle] = win;
  }

  ncclWin* find(ncclWindow* handle) const {
    auto locked = map_.rlock();
    auto it = locked->find(handle);
    return (it != locked->end()) ? it->second : nullptr;
  }

  void erase(ncclWindow* handle) {
    auto locked = map_.wlock();
    locked->erase(handle);
  }

 private:
  NcclWinMap() = default;
  ~NcclWinMap() = default;
  NcclWinMap(const NcclWinMap&) = delete;
  NcclWinMap& operator=(const NcclWinMap&) = delete;

  folly::Synchronized<std::unordered_map<ncclWindow*, ncclWin*>> map_;
};

// Convenience function to access the map
inline NcclWinMap& ncclWinMap() {
  return NcclWinMap::instance();
}
