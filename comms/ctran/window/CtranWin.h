// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <folly/Synchronized.h>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/hints/Hints.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
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

  // remote window info (addr, rkey) for peers participated in this window
  std::vector<window::RemWinInfo> remWinInfo;

  // User specified size in bytes of the data buffer per rank in this window
  size_t dataBytes{0};
  // User specified signal buffer size(in uint64_t) per rank in this window
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

  CtranWin(
      CtranComm* comm,
      size_t dataSize,
      size_t signalSize,
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

  commResult_t allocate(void* userBufPtr = nullptr);
  commResult_t exchange();

  commResult_t free();

  bool nvlEnabled(int rank) const;

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
