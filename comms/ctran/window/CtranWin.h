// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <folly/Synchronized.h>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/hints/Hints.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/ctran/utils/DevMemType.h"
#include "comms/ctran/window/Types.h"

namespace ctran {
struct CtranWin {
  // TODO: remove the communicator from the window allocation.
  // We will need Ctran instead of CtranComm to allocate the window. Current
  // implementation still uses CtranComm for:
  // 1. the communicator's logMetaData for memory logging purposes.
  // 2. the communicator's ctran mapper for network registration.
  // 3. the communicator's bootstrap for intra node bootstrap all gather.
  CtranComm* comm;

  // Remote window info (addr, rkey, dataBytes) for all peers in this window
  std::vector<window::RemWinInfo> remWinInfo;

  // This rank's local data buffer size in bytes
  size_t dataBytes{0};
  // Signal buffer size in number of uint64_t elements per rank
  size_t signalSize{0};
  // The ctran mapper handles for caching the allocated buffer segment
  void* baseSegHdl{nullptr};
  // The ctran mapper handles for caching the allocated buffer registration
  void* baseRegHdl{nullptr};
  // The ctran mapper handles for caching the data segment
  void* dataSegHdl{nullptr};
  // The ctran mapper handles for caching the data registration
  void* dataRegHdl{nullptr};
  // The base pointer of allocated buffer by the window
  void* winBasePtr{nullptr};
  // The pointer of the data buffer of this window
  void* winDataPtr{nullptr};
  // The pointer of the signal buffer of this window
  uint64_t* winSignalPtr{nullptr};
  // Stores signal values for waiting, used to track progress
  std::deque<std::atomic<uint64_t>> waitSignalVal{};

  // Stores signal values for sent signals, used to track progress
  std::deque<std::atomic<uint64_t>> signalVal{};

  CtranWin(
      CtranComm* comm,
      size_t dataSize,
      DevMemType bufType = DevMemType::kCumem);

  inline uint64_t updateOpCount(
      const int rank,
      const window::OpCountType type = window::OpCountType::kWinScope) {
    const auto key = std::make_pair(rank, type);
    auto locked = opCountMap_.wlock();
    auto opCount = 0;

    auto it = locked->find(key);
    if (it == locked->end()) {
      // tracked after first update, starting from value 1
      locked->insert(std::make_pair(key, 1));
    } else {
      opCount = it->second;
      it->second++;
    }
    return opCount;
  }

  inline uint64_t ctranNextWaitSignalVal(int peer) {
    FB_CHECKTHROW_EX_NOCOMM(
        peer < signalSize,
        "peer rank {} exceed window signal buffer size {}",
        peer,
        signalSize);
    return waitSignalVal[peer].fetch_add(1, std::memory_order_relaxed);
  }

  inline uint64_t ctranNextSignalVal(int peer) {
    FB_CHECKTHROW_EX_NOCOMM(
        peer < signalSize,
        "peer rank {} exceed window signal buffer size {}",
        peer,
        signalSize);
    return signalVal[peer].fetch_add(1, std::memory_order_relaxed);
  }

  commResult_t allocate(void* userBufPtr = nullptr);
  commResult_t exchange();

  commResult_t free();

  bool nvlEnabled(int rank) const;

  // Get data size for specific rank
  inline size_t getDataSize(int rank) const {
    if (rank >= 0 && rank < static_cast<int>(remWinInfo.size())) {
      return remWinInfo[rank].dataBytes;
    }
    return 0; // invalid rank
  }

  inline bool isGpuMem() const {
    return bufType_ == DevMemType::kCudaMalloc ||
        bufType_ == DevMemType::kCumem;
  }

 private:
  DevMemType bufType_{DevMemType::kCumem};
  // whether allocate window data buffer or provided by users
  bool allocDataBuf_{true};
  // rank: window::OpCountType as key
  folly::Synchronized<
      std::unordered_map<std::pair<int, window::OpCountType>, uint64_t>>
      opCountMap_;
  // Actual size allocated for the total buffer per rank in this window
  size_t range_{0};
};

commResult_t ctranWinAllocate(
    size_t size,
    CtranComm* comm,
    void** baseptr,
    CtranWin** win,
    const meta::comms::Hints& hints = meta::comms::Hints());

commResult_t ctranWinRegister(
    const void* baseptr,
    size_t size,
    CtranComm* comm,
    CtranWin** win,
    const meta::comms::Hints& hints = meta::comms::Hints());

commResult_t ctranWinSharedQuery(int rank, CtranWin* win, void** addr);

commResult_t ctranWinFree(CtranWin* win);

} // namespace ctran
