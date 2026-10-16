// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Colltrace v2: one ring buffer per communicator, keyed by commDesc.
//
// Collective kernels publish {collId, start|end} into the ring from inside the
// kernel (ColltraceDeviceEventScope, the device ABI in colltrace/common). The
// host side has two parts: the controller (CollTraceV2, this file) services
// every colltrace call -- registration, id allocation, kernel handles, kernel
// counts -- and the CollEvents thread pulls each ring into a bounded per-comm
// host buffer on a timer. getCollEvents() pulls immediately and returns
// everything buffered, so a caller never waits on the timer. Nothing is
// interpreted here; hang detection belongs to the consumer of the events.
//
// Hook contract, all on the host: newCollId(commDesc) is called at the top of
// every collective API call and returns its collId, so a call that launches no
// kernel still gets an id; getKernelHandle(collId) is called for each kernel
// the collective launches and returns the handle for that kernel's launch args;
// getNumKernels(collId) reports how many were launched for an id.
// Callers that launch at the end of the API call fetch the id back with
// getLatestCollId(commDesc). Under an explicit group, kernels launch at group
// end and so belong to the last collective enqueued on the comm. Under CUDA
// graph capture the hooks fire once at capture and every replay re-emits the
// ids.
//
// A collId is 16 bits of communicator index followed by a 48-bit
// per-communicator counter, so it identifies the communicator on its own;
// COLLTRACE_INVALID_COLLID (all ones) is never handed out. The ring stores only
// the low 32 bits of the counter and the drain restores the rest from the host
// counter.
//
// Self-contained: it depends on no communication library, so the one instance
// (in libcommsutils outside Buck) serves every backend in the process (NCCLX,
// MCCL, ...) and the Python bindings. It is switched on by the same knobs its
// clients use, NCCL_COLLTRACE (non-empty) with NCCL_COLLTRACE_VERSION=2, and
// sized by its own environment variables, read once when the instance is
// constructed:
//   COLLTRACE_RING_ENTRIES        entries per communicator ring (16 bytes
//                                 each, rounded up to a power of 2); default
//                                 65536
//   COLLTRACE_MAX_BUFFERED_EVENTS drained events kept per communicator until
//                                 read, oldest dropped past it; default 1M
//   COLLTRACE_COLL_HISTORY        most recent collectives per communicator
//                                 whose kernel count is still known; default
//                                 64M
//   COLLTRACE_DRAIN_INTERVAL_MS   drain thread period; default 100
//
// registerComm allocates the ring and warms the GPU clock calibration, so it
// must run at communicator init, never inside graph capture. An unregistered
// commDesc is inert: newCollId returns COLLTRACE_INVALID_COLLID,
// getKernelHandle of that returns an unarmed handle, getCollEvents() returns
// nothing.
//
// The ring is Overwrite: if the drain falls behind, the oldest entries are lost
// and the loss is logged; the kernel never waits on the host.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "comms/utils/colltrace/common/ColltraceDeviceHandle.h"

// The API its consumers (libnccl, libmccl, the Python bindings) resolve,
// exported whatever visibility the enclosing library is built with.
#define COLLTRACEV2_API __attribute__((visibility("default")))

namespace meta::comms::colltracev2 {

// Returned by newCollId / getLatestCollId when there is no collective; never a
// real id.
constexpr uint64_t COLLTRACE_INVALID_COLLID = ~uint64_t{0};

enum class EventKind : uint8_t {
  kCollStartEvent,
  kCollEndEvent,
};

struct Event {
  uint64_t collId;
  EventKind kind;
  std::chrono::system_clock::time_point ts;
};

COLLTRACEV2_API bool enabled();

struct CommState;
class CollEvents;

class COLLTRACEV2_API CollTraceV2 {
 public:
  CollTraceV2();
  ~CollTraceV2();

  // Shared by every backend that arms kernels.
  static CollTraceV2& instance();

  CollTraceV2(const CollTraceV2&) = delete;
  CollTraceV2& operator=(const CollTraceV2&) = delete;
  CollTraceV2(CollTraceV2&&) = delete;
  CollTraceV2& operator=(CollTraceV2&&) = delete;

  bool registerComm(std::string_view commDesc);
  void deregisterComm(std::string_view commDesc);

  uint64_t newCollId(std::string_view commDesc);
  colltrace::ColltraceDeviceHandle getKernelHandle(uint64_t collId);

  // COLLTRACE_INVALID_COLLID if none
  uint64_t getLatestCollId(std::string_view commDesc);
  // Number of kernels launched so far for collId. Counts are kept for a comm's
  // most recent COLLTRACE_COLL_HISTORY collectives; older or unknown ids
  // return -1.
  int getNumKernels(uint64_t collId);
  std::vector<Event> getCollEvents(std::string_view commDesc);

 private:
  std::shared_ptr<CommState> find(std::string_view commDesc);
  std::shared_ptr<CommState> find(uint32_t commIndex);

  const uint32_t ringEntries_;
  const std::size_t collHistory_;
  std::unique_ptr<CollEvents> collEvents_;

  std::mutex commsMu_;
  uint32_t nextCommIndex_{1};
  std::unordered_map<std::string, std::shared_ptr<CommState>> commsByDesc_;
  std::unordered_map<uint32_t, std::shared_ptr<CommState>> commsByIndex_;
};

} // namespace meta::comms::colltracev2
