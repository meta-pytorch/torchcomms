// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <linux/types.h> // for __be32
#include <cstdint>

namespace ibverbx {

/// Work Queue Element stride (fixed size for WQE alignment)
constexpr int WQE_STRIDE = 64;

/**
 * GPU-accessible IBGDA data structures for kernel-level RDMA operations.
 *
 * These structures contain GPU-accessible pointers to InfiniBand resources,
 * enabling CUDA kernels to directly post RDMA operations without host
 * involvement. All pointers in these structures point to GPU-mapped memory
 * regions.
 */

struct device_cq {
  // CQ
  void* cq_buf;
  uint32_t ncqes;

  // CQ doorbell record (if needed)
  __be32* cq_dbrec;
} __attribute__((__aligned__(8)));

struct device_qp {
  // QP num & device idx are used to identify the QP
  uint32_t qp_num{};
  uint32_t device_idx{};

  void* wq_buf{};
  uint32_t nwqes{};

  // TODO: add rq support if needed
  // __be32* rq_dbrec;
  __be32* sq_dbrec{};
  uint64_t* bf_reg{};

  int post_send_lock{};

  // Track producer/consumer indices
  volatile uint64_t producer_idx{0}; // WQE producer index
  volatile uint64_t consumer_idx{0}; // CQE consumer index

  // pointer to CQ
  device_cq* cq{};
} __attribute__((__aligned__(8)));
} // namespace ibverbx
