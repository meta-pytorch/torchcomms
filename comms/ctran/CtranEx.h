// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <fmt/format.h>
#include <cstdint>
#include <string>
#include <vector>

#include "comms/utils/commSpecs.h"

// Fordward declaration to avoid dependency on CtranComm.h
class CtranComm;

namespace ctran {
enum __attribute__((visibility("default"))) CtranExBackend {
  kCtranIbBackend,
};

struct __attribute__((visibility("default"))) CtranExHostInfo {
  int port{-1};
  std::string ipv6{""};
  std::string hostname{""};
  std::string ifName{""}; // The interface name that server binds to.
                          // The client should also binds to the same ifname as
                          // peer QP server when initiating connection

  std::string toString() const;
};

class __attribute__((visibility("default"))) CtranExRequest {
 public:
  CtranExRequest();
  ~CtranExRequest();

  bool isComplete() const;
  commResult_t test(bool& complete);
  commResult_t wait();

  friend class CtranEx;
  friend class CtranExComm;

 protected:
  void* impl_{nullptr};
  void initialize();
};

// // Forward declaration to avoid dependency on CtranComm.h
// class CtranComm;

class __attribute__((visibility("default"))) CtranEx {
 public:
  CtranEx(
      const int rank,
      const int cudaDevice,
      const CtranExHostInfo& hostInfo,
      const std::vector<CtranExBackend> backends,
      const std::string& desc);

  ~CtranEx();

  bool isInitialized();

  commResult_t regMem(const void* ptr, const size_t size, void** regHdl);
  commResult_t deregMem(void* regHdl);

  commResult_t isendCtrl(
      const void* buf,
      const size_t size,
      const void* bufRegHdl,
      const int peerRank,
      const CtranExHostInfo& peerHostInfo,
      CtranExRequest** req);

  commResult_t irecvCtrl(
      const int peerRank,
      const CtranExHostInfo& peerHostInfo,
      void** peerRemoteBuf,
      uint32_t* peerRemoteKey,
      CtranExRequest** req);

  commResult_t isendCtrl(
      const int peerRank,
      const CtranExHostInfo& peerHostInfo,
      CtranExRequest** req);

  commResult_t irecvCtrl(
      const int peerRank,
      const CtranExHostInfo& peerHostInfo,
      CtranExRequest** req);

  commResult_t isendCtrl(const int peerRank, CtranExRequest** req);

  commResult_t irecvCtrl(const int peerRank, CtranExRequest** req);

  commResult_t iput(
      const void* localBuf,
      const std::size_t len,
      void* localRegHdl,
      const int peerRank,
      void* peerRemoteBuf,
      uint32_t peerRemoteKey,
      bool notify,
      CtranExRequest** req);

  commResult_t
  iflush(const void* localBuf, void* localRegHdl, CtranExRequest** req);

  commResult_t checkNotify(int peerRank, bool& done);

  commResult_t waitNotify(int peerRank);

  commResult_t releaseRemoteTransStates();

  commResult_t initRemoteTransStates();

 private:
  void* impl_{nullptr};
};
} // namespace ctran

template <>
struct fmt::formatter<ctran::CtranExBackend> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(ctran::CtranExBackend status, FormatContext& ctx) const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};
