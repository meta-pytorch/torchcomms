// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_IB_CONFIG_H
#define CTRAN_IB_CONFIG_H

#include <comms/utils/cvars/nccl_cvars.h>
#include <cstddef>
#include <cstdint>
#include <optional>

struct CtranIbConfig {
  // Maximum data QPs per peer, divided evenly across its VCs.
  std::optional<int> numQps;
  // Maximum WQE payload in bytes; zero divides an operation evenly over QPs.
  std::optional<size_t> qpScalingTh;
  // Data-QP scheduling mode: spray or dynamic QP load balancing.
  std::optional<enum NCCL_CTRAN_IB_VC_MODE> vcMode;
  // Maximum number of outstanding WQEs on each data QP.
  std::optional<int> qpMsgs;
  // Multi-NIC interleaving is transport policy, not caller configuration: one
  // WQE's size does not describe the aggregate workload. Latency-sensitive
  // callers should select a single-NIC transport instead.

  // Whether to support local flush.
  std::optional<bool> enableLocalFlush;
  // Per-CQ entry cap; a non-positive value uses the device limit.
  std::optional<int> maxNumCqe;
  // Maximum NICs to initialize.
  std::optional<int> maxNumNic;
  // IB traffic class in [0, 255].
  std::optional<int64_t> trafficClass;
};

#endif
