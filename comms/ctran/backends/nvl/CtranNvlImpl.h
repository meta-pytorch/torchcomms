// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_NVL_IMPL_H_
#define CTRAN_NVL_IMPL_H_

#include <mutex>
#include <string>
#include <unordered_map>

#include "comms/ctran/backends/nvl/CtranNvl.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/utils/commSpecs.h"

struct CtranNvlRegElem {
  // User passed addr, size at ncclCommRegister
  const void* buf{nullptr};
  const size_t len{0};
  ctran::utils::CtranIpcMem ipcMem;

 public:
  CtranNvlRegElem(const void* buf, const size_t len, int cudaDev)
      : buf(buf), len(len), ipcMem(cudaDev, "NVL RegElem") {};
  ~CtranNvlRegElem() {};

  commResult_t tryLoad(bool& supported) {
    return ipcMem.tryLoad(buf, len, supported);
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "buf: " << buf << ", len: " << len
       << ", ipcMem: " << ipcMem.toString();
    return ss.str();
  }
};

struct CtranNvlRemRegElem {
  ctran::utils::CtranIpcRemMem ipcRemMem;

 public:
  CtranNvlRemRegElem(
      const ctran::utils::CtranIpcDesc& ipcDesc,
      int cudaDev,
      const struct CommLogData* logMetaData)
      : ipcRemMem(ipcDesc, cudaDev, logMetaData, "NVL RemRegElem") {};

  std::string toString() const {
    return ipcRemMem.toString();
  }
};

class CtranNvl::Impl {
 public:
  Impl() = default;
  ~Impl() = default;

  // two GPUs could be connected through the NVLink fabric or traditional
  // intra-host NVLink.
  struct NvlSupportMode {
    bool nvlIntraHost{false};
    bool nvlFabric{false};
  };

  // Import a remote registration
  commResult_t importRemReg(
      int peer,
      const ctran::utils::CtranIpcDesc& ipcDesc,
      void** mappedBase);

  // Release a remote registration
  commResult_t releaseRemReg(int peer, void* basePtr);

  // Release all remote registrations
  void clearAllRemReg();

  // Get the number of existing remote registrations for a given peer
  size_t getNumRemMem(int peerRank) const;

  CtranComm* comm{nullptr};
  // index of the vector is the peer rank, and the value is the support mode
  // the size of the vector is comm->stateX->nRanks().
  std::vector<NvlSupportMode> nvlRankSupportMode;

  // Cache of imported remote registration
  std::unordered_map<
      int, // rank
      std::unordered_map<
          uint64_t, // ptr
          std::unique_ptr<CtranNvlRemRegElem>>>
      remRegMap;

  // Lock to protect remRegMap.
  // Both import/release should be called only by the GPE thread during run,
  // except release at destruction is handled by the main thread. Thus, using
  // lock to avoid double-release.
  mutable std::mutex remRegMapMutex;
};
#endif
