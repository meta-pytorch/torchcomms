/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sharded_relay_graph_scratch.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace rcclx {
namespace relay {

namespace {

// (cacheTag, device, stream, key, graphId) -> current buffer. The graph id is
// part of the key so a captured buffer is never shared with, or freed by,
// anything outside its own graph.
using ScratchKey =
    std::tuple<const void*, int, const void*, int, unsigned long long>;

struct ScratchEntry {
  void* buffer{nullptr};
  size_t size{0};
};

// Per-graph bookkeeping. destructorRegistered is a CLAIM rather than a record
// of success: it is set under tableMutex() by whichever call will go on to call
// ncclCudaGraphAddDestructor outside the lock, so two threads capturing into
// the same graph cannot both register, and is cleared again if that call fails.
struct GraphAllocs {
  bool destructorRegistered{false};
  // Every allocation ever handed to this graph, so a grown-over predecessor is
  // still reclaimable. It cannot be freed at growth time: the graph has already
  // captured nodes that reference it.
  std::vector<void*> buffers;
};

// Function-local statics rather than file-scope objects: this is linked into a
// library whose teardown order relative to other translation units is not
// something we want to depend on.
std::mutex& tableMutex() {
  static std::mutex m;
  return m;
}

std::map<ScratchKey, ScratchEntry>& table() {
  static std::map<ScratchKey, ScratchEntry> t;
  return t;
}

// Every allocation ever handed to a graph, keyed by graph id.
std::map<unsigned long long, GraphAllocs>& graphAllocs() {
  static std::map<unsigned long long, GraphAllocs> m;
  return m;
}

// Graph ids whose destructor has fired, behind their own mutex. The destructor
// callback touches nothing else and makes no HIP call: it can run on a
// HIP-internal thread, and RCCL's own graph destructor (persistentDestructor in
// enqueue.cc) is equally careful to only enqueue there and reclaim later.
std::mutex& deadMutex() {
  static std::mutex m;
  return m;
}

std::vector<unsigned long long>& deadGraphs() {
  static std::vector<unsigned long long> v;
  return v;
}

void graphDiedCallback(void* arg) {
  auto* graphId = static_cast<unsigned long long*>(arg);
  {
    std::lock_guard<std::mutex> lock(deadMutex());
    deadGraphs().push_back(*graphId);
  }
  delete graphId;
}

// Free the buffers of every graph whose destructor has fired. Deliberately
// called from graphScratchGet, on a user thread, rather than from the
// destructor callback itself, which can run on a HIP-internal thread and must
// make no HIP call.
//
// The consequence is that the dead list is only drained by a LATER capture. If
// capture-path usage stops for good, the last dead graphs' buffers are held
// until process exit. That is accepted rather than fixed: the alternative is
// freeing from the callback, which is exactly what is not safe there. The
// residue is bounded by the buffers of the graphs destroyed after the final
// graphScratchGet call, and every one of them was already held for that graph's
// whole lifetime, so this costs no more than keeping the last generation of
// graphs alive slightly longer.
void reclaimDeadGraphs() {
  std::vector<unsigned long long> dead;
  {
    std::lock_guard<std::mutex> lock(deadMutex());
    if (deadGraphs().empty()) {
      return;
    }
    dead.swap(deadGraphs());
  }

  std::vector<void*> stale;
  {
    std::lock_guard<std::mutex> lock(tableMutex());
    for (unsigned long long id : dead) {
      auto allocIt = graphAllocs().find(id);
      if (allocIt != graphAllocs().end()) {
        stale.insert(
            stale.end(),
            allocIt->second.buffers.begin(),
            allocIt->second.buffers.end());
        graphAllocs().erase(allocIt);
      }
    }
    for (auto it = table().begin(); it != table().end();) {
      const unsigned long long id = std::get<4>(it->first);
      it = (std::find(dead.begin(), dead.end(), id) != dead.end())
          ? table().erase(it)
          : std::next(it);
    }
  }

  // Outside the lock: hipFree can block, and nothing here needs the table.
  for (void* buffer : stale) {
    (void)cudaFree(buffer);
  }
}

// Matches the rounding the per-collective caches apply, so a graph does not end
// up with a differently sized buffer than the uncaptured path would have used.
size_t roundedAllocSize(size_t requiredBytes) {
  constexpr size_t kGranularity = 64ull * 1024 * 1024;
  if (requiredBytes < 1024 * 1024) {
    return requiredBytes;
  }
  return ((requiredBytes + kGranularity - 1) / kGranularity) * kGranularity;
}

} // namespace

void* graphScratchGet(
    const void* cacheTag,
    int key,
    size_t requiredBytes,
    cudaStream_t stream,
    struct ncclCudaGraph graph) {
  if (requiredBytes == 0) {
    return nullptr;
  }

  reclaimDeadGraphs();

  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return nullptr;
  }

  const ScratchKey mapKey{
      cacheTag, device, static_cast<const void*>(stream), key, graph.graphId};

  {
    std::lock_guard<std::mutex> lock(tableMutex());
    auto it = table().find(mapKey);
    if (it != table().end() && it->second.size >= requiredBytes) {
      return it->second.buffer;
    }
  }

  // Allocate outside the capture. Relaxed capture mode is what makes a blocking
  // allocation legal on a thread with a capture in progress; enqueue.cc does
  // the same exchange before allocating a graph's persistent work buffer.
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  if (cudaThreadExchangeStreamCaptureMode(&mode) != cudaSuccess) {
    return nullptr;
  }
  void* buffer = nullptr;
  const size_t allocSize = roundedAllocSize(requiredBytes);
  const cudaError_t err = cudaMalloc(&buffer, allocSize);
  (void)cudaThreadExchangeStreamCaptureMode(&mode);
  if (err != cudaSuccess) {
    return nullptr;
  }

  // tableMutex() is process-wide and shared by every collective's cache, so no
  // HIP call is made while holding it: doing so serializes concurrent captures
  // on unrelated streams. The lock is taken twice instead -- once to CLAIM the
  // destructor registration for this graph, once to publish the buffer -- with
  // ncclCudaGraphAddDestructor and the error-path cudaFree in between, outside
  // it. reclaimDeadGraphs() frees outside the lock for the same reason.
  bool mustRegister = false;
  {
    std::lock_guard<std::mutex> lock(tableMutex());
    GraphAllocs& allocs = graphAllocs()[graph.graphId];
    if (!allocs.destructorRegistered) {
      allocs.destructorRegistered = true;
      mustRegister = true;
    }
  }

  if (mustRegister) {
    // One destructor per graph is enough, and the callback needs an id that
    // outlives this frame.
    auto* tag = new unsigned long long(graph.graphId);
    if (ncclCudaGraphAddDestructor(graph, graphDiedCallback, tag) !=
        ncclSuccess) {
      delete tag;
      {
        std::lock_guard<std::mutex> lock(tableMutex());
        // Release the claim. Nothing has been published for this buffer yet, so
        // dropping an entry that holds no buffers leaves no trace; if another
        // call has since published one, leave its buffers alone and only let
        // the registration be retried.
        auto it = graphAllocs().find(graph.graphId);
        if (it != graphAllocs().end()) {
          if (it->second.buffers.empty()) {
            graphAllocs().erase(it);
          } else {
            it->second.destructorRegistered = false;
          }
        }
      }
      (void)cudaFree(buffer);
      return nullptr;
    }
  }

  {
    std::lock_guard<std::mutex> lock(tableMutex());
    graphAllocs()[graph.graphId].buffers.push_back(buffer);
    ScratchEntry& entry = table()[mapKey];
    entry.buffer = buffer;
    entry.size = allocSize;
  }
  return buffer;
}

} // namespace relay
} // namespace rcclx
