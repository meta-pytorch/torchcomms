// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/utils/TypeUtils.h"
#include "comms/utils/colltrace/CollRecord.h"
#include "comms/utils/cvars/nccl_cvars.h"

// Allow test class to access private members
class MapperTraceTest;
class CtranMapperRequest;

namespace ncclx::colltrace {

extern thread_local bool shouldMapperTraceCurrentThread;

// Fixme: Give a proper type for the mapper handle.
using mapperHandle_t = void*;

// It is the virtual address of the remote buffer. Meaning although
// it is a void* it cannot be dereferenced. Giving it a type alias
// so its meaning won't be confused.
using remoteBuffer_t = void*;

struct CollStart {
  std::shared_ptr<meta::comms::colltrace::ICollRecord> coll;

  std::string serialize(bool quoted = false) const;
};

struct CollEnd {
  // We don't need any additional information, because only one collective
  // Should be running at a time.
  std::string serialize(bool quoted = false) const;
};

struct CopyStart {
  void* sourceBuffer;
  void* destBuffer;
  size_t length;
  cudaStream_t stream;
  CtranMapperRequest* req;

  std::string serialize(bool quoted = false) const;
};

struct SendCtrlStart {
  void* buffer;
  mapperHandle_t mapperHandle;
  int peerRank;
  CtranMapperRequest* req;

  std::string serialize(bool quoted = false) const;
};

struct RecvCtrlStart {
  // This point will only hold valid address after we received MapperRequestEnd
  // for this recvCtrl.
  remoteBuffer_t* recvBufferPtr;
  CtranMapperRemoteAccessKey* accessKeyPtr;
  int peerRank;
  CtranMapperRequest* req;

  std::string serialize(bool quoted = false) const;
};

struct SendSyncCtrlStart {
  int peerRank;
  CtranMapperRequest* req;

  std::string serialize(bool quoted = false) const;
};

struct RecvSyncCtrlStart {
  int peerRank;
  CtranMapperRequest* req;

  std::string serialize(bool quoted = false) const;
};

struct PutStart {
  void* sendBuffer;
  remoteBuffer_t remoteBuffer;
  size_t length;
  int peerRank;
  mapperHandle_t sourceHandle;
  CtranMapperRemoteAccessKey remoteAccessKey;
  CtranMapperRequest* req;

  std::string serialize(bool quoted = false) const;
};

// This could signal the end of either SendCtrl, RecvCtrl or Put depending
// on what CtranMapperRequest is trying to send
struct MapperRequestEnd {
  CtranMapperRequest* req;

  std::string serialize(bool quoted = false) const;
};

struct RecvNotified {
  int peerRank;

  std::string serialize(bool quoted = false) const;
};

using MapperRequestStartEvent = std::variant<
    CopyStart,
    SendCtrlStart,
    RecvCtrlStart,
    SendSyncCtrlStart,
    RecvSyncCtrlStart,
    PutStart>;

template <typename T>
concept MapperRequestStartEventType =
    isVariantMember<T, MapperRequestStartEvent>::value;

using MapperEvent = std::variant<
    CollStart,
    CollEnd,
    CopyStart,
    SendCtrlStart,
    RecvCtrlStart,
    SendSyncCtrlStart,
    RecvSyncCtrlStart,
    PutStart,
    RecvNotified,
    MapperRequestEnd>;

template <typename T>
concept MapperEventType = isVariantMember<T, MapperEvent>::value;

struct MapperRequestEventInfo {
  MapperEvent event;
  uint64_t seqNum;

  std::string serialize(bool quoted = false) const;
};

class MapperTrace {
  friend class ::MapperTraceTest;

 public:
  struct Dump {
    std::shared_ptr<meta::comms::colltrace::ICollRecord> currentColl;
    std::unordered_map<int, int> recvNotifiedByPeer;
    std::unordered_map<int, int> putFinishedByPeer;
    std::vector<MapperRequestEventInfo> unfinishedRequests;
  };
  Dump dump();

  /*
   * Function for recording mapper event. Only one thread shall be able to
   * call this function! Otherwise the eventHistory_ will be corrupted.
   * Currently this function will be called in gpeThreadFn thread.
   */
  template <MapperEventType T>
  void recordMapperEvent(T event) {
    if constexpr (std::is_same_v<T, CollEnd>) {
      if (callback_.has_value()) {
        CLOGF_SUBSYS(INFO, COLL, "MapperTrace: Callback triggered");
        (*callback_)();
      }
      shouldMapperTraceCurrentThread = false;
    } else if constexpr (std::is_same_v<T, CollStart>) {
      shouldMapperTraceCurrentThread = true;
    }
    // Only calls recordMapperEventImpl at the start and end of a collective
    // As they are less sensitive to the performance and we want to keep track
    // of the collective history.
    if constexpr (std::is_same_v<T, CollStart> || std::is_same_v<T, CollEnd>) {
      curCollInfoLocked_.withWLock([this, event](CurCollInfo& curCollInfo) {
        recordMapperEventImpl(event, curCollInfo);
      });
      return;
    }
    // Do not record the event on non-GPE threads or a collective is not
    // running
    if (!shouldMapperTraceCurrentThread) {
      return;
    }
    auto nextIndex = eventHistorySizeAtomic_.load(std::memory_order_relaxed);
    if (nextIndex == NCCL_MAPPERTRACE_EVENT_RECORD_MAX) {
      return;
    }
    eventHistory_.emplace_back(std::move(event));
    eventHistorySizeAtomic_.store(nextIndex + 1, std::memory_order_release);
  }

  void registerBeforeCollEndCallback(std::function<void()> callback);

  MapperTrace();

 private:
  // Valid for only the current collective.
  struct CurCollInfo {
    int64_t readIndex{0};
    std::shared_ptr<meta::comms::colltrace::ICollRecord> currentColl;
    std::unordered_map<int, int> recvNotifiedByPeer;
    std::unordered_map<int, int> putFinishedByPeer;
    // Requests that have been created in Mapper and not yet finished
    // The request might be unfinished due to the request is still ongoing or
    // the request's complete has never been checked.
    std::unordered_map<const CtranMapperRequest*, uint64_t> unfinishedRequests;

    void clear();
  };

  // Start of recordMapperEvent exclusive resources.
  // Always align these variables to the next cache line to avoid false sharing.
  // Any variables in this block should not be modified by other threads.
  // And don't forget to align the next variable to the next cache line.
  alignas(folly::hardware_destructive_interference_size)
      std::vector<MapperEvent> eventHistory_;
  std::atomic<int64_t> eventHistorySizeAtomic_{0};
  const uint64_t maxEventCount_{NCCL_MAPPERTRACE_EVENT_RECORD_MAX};

  // End of recordMapperEvent exclusive resources.
  // Align the next variable to the next cache line.

  alignas(folly::hardware_destructive_interference_size)
      folly::Synchronized<CurCollInfo> curCollInfoLocked_;
  // For testing purpose, trigger a callback before CollEnd is recorded
  // So that we can have a chance to dump the trace before CollEnd erases
  // the current collective Mapper state.
  std::optional<std::function<void()>> callback_;

  // All recordMapperEventImpl should be called with mapperMutex_ held.
  template <MapperRequestStartEventType T>
  void recordMapperEventImpl(T startEvent, CurCollInfo& curCollInfo) {
    // -1 since we already put the event in the eventHistory_ in
    // recordMapperEvent
    if (startEvent.req) {
      curCollInfo.unfinishedRequests.emplace(
          startEvent.req, curCollInfo.readIndex);
    }
  }

  void recordMapperEventImpl(CollStart collStart, CurCollInfo& curCollInfo);
  void recordMapperEventImpl(CollEnd collEnd, CurCollInfo& curCollInfo);
  void recordMapperEventImpl(
      RecvNotified recvNotified,
      CurCollInfo& curCollInfo);
  void recordMapperEventImpl(
      MapperRequestEnd mapperRequestEnd,
      CurCollInfo& curCollInfo);
};

MapperTrace* getMapperTrace(CtranComm* comm);

template <typename T>
bool recordMapperEvent(MapperTrace* trace, T event) {
  if (!trace) {
    return false;
  }
  trace->recordMapperEvent(event);
  return true;
}

template <typename T>
bool recordMapperEvent(CtranComm* comm, T event) {
  auto trace = getMapperTrace(comm);
  if (trace == nullptr) {
    return false;
  }
  trace->recordMapperEvent(event);
  return true;
}

} // namespace ncclx::colltrace
