// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAllvDedup/ResourceImpl.h"
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/utils/CtranTraceLogger.h"

namespace ctran::alltoallvdedup {

class AlgoImpl {
 public:
  PersistArgs pArgs;

  AlgoImpl(
      CtranComm* comm,
      ncclx::CommStateX* statex,
      ICtran* ctran,
      cudaStream_t stream);
  ~AlgoImpl() {};

  enum Phase {
    kExec,
  };

  commResult_t initialize();

  static inline const std::string algoName(Phase phase) {
    if (phase == kExec) {
      return "ctranAllToAllvDedupExec";
    } else {
      return "unknown";
    }
  }

  // For testing purpose
  ResourceImpl* getResource() const {
    return resource_.get();
  }

  commResult_t exec(const ExecArgs& args, const uint64_t opCount);

  // todo setup in initialize?
  std::unique_ptr<ctran::utils::TraceLogger> ctran_trace_logger{nullptr};

 private:
  // initialize constant configuration
  commResult_t initConfig();

  std::unique_ptr<ResourceImpl> resource_{nullptr};

  // FIXME: need comm only for passing OpElem to GPE thread; we need remove it
  // when moving Ctran out from ncclx
  CtranComm* comm_{nullptr};
  ncclx::CommStateX* statex_{nullptr};
  ICtran* ctran_{nullptr};
  cudaStream_t stream_{nullptr};
  PersistConfig config_;
};
} // namespace ctran::alltoallvdedup

template <>
struct fmt::formatter<ctran::alltoallvdedup::AlgoImpl::Phase>
    : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(ctran::alltoallvdedup::AlgoImpl::Phase status, FormatContext& ctx)
      const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};
