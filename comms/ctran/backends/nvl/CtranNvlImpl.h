// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_NVL_IMPL_H_
#define CTRAN_NVL_IMPL_H_

#include <folly/Synchronized.h>
#include <mutex>
#include <string>
#include <unordered_map>

#include "comms/ctran/backends/nvl/CtranNvl.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/utils/commSpecs.h"

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

  CtranComm* comm{nullptr};
  // index of the vector is the peer rank, and the value is the support mode
  // the size of the vector is comm->stateX->nRanks().
  std::vector<NvlSupportMode> nvlRankSupportMode;
};
#endif
