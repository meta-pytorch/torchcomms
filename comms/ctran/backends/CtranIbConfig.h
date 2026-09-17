// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_IB_CONFIG_H
#define CTRAN_IB_CONFIG_H

#include <comms/utils/cvars/nccl_cvars.h>
#include <cstddef>
#include <cstdint>
#include <optional>

struct CtranIbConfig {
  // Maximum data QPs per peer, divided evenly across its VCs.
  int numQps{NCCL_CTRAN_IB_MAX_QPS};
  // Maximum WQE payload in bytes; zero divides an operation evenly over QPs.
  size_t qpScalingTh{NCCL_CTRAN_IB_QP_SCALING_THRESHOLD};
  // Data-QP scheduling mode: spray or dynamic QP load balancing.
  enum NCCL_CTRAN_IB_VC_MODE vcMode { NCCL_CTRAN_IB_VC_MODE };
  // Maximum number of outstanding WQEs on each data QP.
  int qpMsgs{static_cast<int>(NCCL_CTRAN_IB_QP_MAX_MSGS)};
  // Multi-NIC interleaving is transport policy, not caller configuration: one
  // WQE's size does not describe the aggregate workload. Latency-sensitive
  // callers should select a single-NIC transport instead.

  // Whether to support local flush; unset selects the context-specific default.
  std::optional<bool> enableLocalFlush;
  // Per-CQ entry cap; a non-positive value uses the device limit.
  int maxNumCqe{NCCL_CTRAN_IB_MAX_NUM_CQE};
  // Maximum NICs to initialize, capped by NCCL_CTRAN_IB_DEVICES_PER_RANK.
  int maxNumNic{NCCL_CTRAN_IB_DEVICES_PER_RANK};
  // Fallback IB traffic class in [0, 255].
  int64_t trafficClass{NCCL_IB_TC};
};

#endif
