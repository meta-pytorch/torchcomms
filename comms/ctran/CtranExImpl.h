// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/ScopeGuard.h>
#include <cstdint>
#include <memory>
#include <string>

#include "comms/ctran/CtranEx.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/utils/AsyncError.h"

namespace ctran {
class CtranExImpl {
 public:
  CtranExImpl(int rank, int cudaDev, const std::string& desc);
  ~CtranExImpl() = default;

  void initialize(
      std::optional<const CtranExHostInfo*> hostInfo,
      const std::vector<CtranExBackend>& backends);

  int rank{-1};
  int cudaDev{-1};
  std::string desc{"undefined"};
  std::unique_ptr<CtranCtrlManager> ctrlMgr{nullptr};
  std::unique_ptr<CtranIb> ctranIb{nullptr};

  std::string describe() const {
    std::string instanceStr{""};
    switch (instanceType_) {
      case InstanceType::kCtranEx:
        instanceStr = "CtranEx";
        break;
    };
    return fmt::format(
        "{}: Rank {} cudaDev {} desc {} this {}",
        instanceStr,
        rank,
        cudaDev,
        desc,
        (void*)this);
  }

 protected:
  enum class InstanceType {
    kCtranEx,
  };
  InstanceType instanceType_{InstanceType::kCtranEx};
};

class CtranExRequestImpl {
 public:
  enum Type {
    SEND_CTRL,
    RECV_CTRL,
    SEND_SYNC_CTRL,
    RECV_SYNC_CTRL,
    PUT,
    FLUSH,
    BCAST,
  };
  Type type{SEND_CTRL};

  // Internal request tracking completion from IB backend.
  CtranIbRequest ibReq;

  // CtranEx layer fields
  union {
    struct {
      ControlMsg msg;
    } sendCtrl;
    struct {
      ControlMsg msg;
      // Pointer to user provided input parameters.
      void** rBuf;
      uint32_t* rKey;
    } recvCtrl;
    struct {
      ControlMsg msg;
    } sendSyncCtrl;
    struct {
      ControlMsg msg;
    } recvSyncCtrl;
    struct {
      // completion is set by GPE thread and checked by calling thread.
      std::atomic_flag complete;
    } bcast;
  };

  // Pointer to ctranIb object in CtranExImpl. We should never see user checks a
  // request if ctranIb has already been released.
  CtranIb* ctranIb{nullptr};
  std::shared_ptr<AsyncError> asyncErr{nullptr};

 public:
  CtranExRequestImpl() {};
  ~CtranExRequestImpl() {};

  // Internal initialization to avoid exposing dependencies in CtranEx.h.
  // Initialized for CtranEx transport APIs.
  void initialize(Type type, CtranIb* ctranIb = nullptr);

  // Initialized broadcast associated with a communicator.
  void initialize(Type type, CtranComm* ctranComm = nullptr);

  // Post processing after IB request is completed. Triggered in
  // CtranExRequest::test() | wait() upon completion
  void atComplete(CtranExRequest* req);

  // Mark the request as complete.
  void complete();
};

inline void backendsToStr(
    const std::vector<CtranExBackend>& backends,
    std::vector<std::string>& backendStrs) {
  for (auto backend : backends) {
    if (backend == CtranExBackend::kCtranIbBackend) {
      backendStrs.push_back("IB");
    }
  }
}

void initEnvCtranEx();
} // namespace ctran

template <>
struct fmt::formatter<ctran::CtranExRequestImpl::Type> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(ctran::CtranExRequestImpl::Type status, FormatContext& ctx)
      const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};
