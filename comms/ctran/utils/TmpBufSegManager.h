// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include <unistd.h>
#include <unordered_set>
#include <vector>
#include "comms/ctran/utils/Utils.h"

namespace ctran::utils {
/* Help manage temporary buffer segments in algorithms
 */

template <typename T, T MaxNumSegs>
class TmpBufSegManager {
  struct SegInfo {
    size_t len{0};
    size_t offset{0}; // offset since allocated base
  };

 public:
  TmpBufSegManager() {
    // Default alginment is page size unless overridden by callsite
    alignment_ = getpagesize();
  }

  // Insert an id and its length. Manager will track it and compute its offset
  // from base.
  // If the id exceeds the predefined MaxNumSegs range, or already inserted
  // before, insert is a no-op and return false. A successful insert returns
  // true.
  bool insert(const T id, const size_t len_) {
    // Do not insert if already exist or exceed the range; likely a bug
    if (insertedIds_.contains(id) || id >= MaxNumSegs) {
      return false;
    }
    // Make sure all buffer segments are multiples of alignment
    size_t len = ctran::utils::align(len_, alignment_);
    segments.at(static_cast<size_t>(id)) = {len, totalLen};
    totalLen += len;
    insertedIds_.insert(id);
    return true;
  }

  // Check if a given id has been inserted in manager.
  // It is suggested to be used before getSegInfo(). It often indicates a
  // callsite bug if it queries an id that has not yet been inserted.
  // NOTE: We define the check and get APIs separately so flexible for callsite
  // to optionally skip the check.
  bool contains(const T id) const {
    return insertedIds_.contains(id);
  }

  // Get the segment info for a given id.
  // This API doesn't validate the id. See contains API for optional check.
  SegInfo getSegInfo(const T id) const {
    return segments.at(static_cast<size_t>(id));
  }

  std::array<SegInfo, static_cast<size_t>(MaxNumSegs)> segments;
  size_t totalLen{0};

 private:
  std::unordered_set<T> insertedIds_;
  size_t alignment_{0};
};
} // namespace ctran::utils
