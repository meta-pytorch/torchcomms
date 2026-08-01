// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Host-only regression test for `Config::getNvlBufferSizeHint` /
// `getRdmaBufferSizeHint` (cpp/shared/Config.h). These back the Python
// `Config.get_{nvl,rdma}_buffer_size_hint(hidden_bytes, num_ranks)` bindings
// that a consuming MoE dispatcher's `get_buffer()` calls. The original port
// took zero args and ignored `num_channels`; this guards the faithful
// buffer-sizing formula. Pure host arithmetic — no GPU/HIP/ncclx required.

#include <gtest/gtest.h>

#include "comms/prims/collectives/link_ep/cpp/shared/Config.h"

namespace comms::prims::link_ep {
namespace {

// bf16 hidden=8192 -> 16384 bytes/token (a typical MoE hidden size).
constexpr std::size_t kHiddenBytes = 8192UL * 2UL;

TEST(ConfigBufferSizeHint, NvlHintPositiveAndAligned) {
  Config cfg(/*num_sms=*/20,
             /*num_max_nvl_chunked_send_tokens=*/6,
             /*num_max_nvl_chunked_recv_tokens=*/256);
  const std::size_t nvl =
      cfg.getNvlBufferSizeHint(kHiddenBytes, /*num_ranks=*/8);
  EXPECT_GT(nvl, 0U);
  EXPECT_EQ(nvl % NUM_BUFFER_ALIGNMENT_BYTES, 0U);
}

TEST(ConfigBufferSizeHint, NvlHintScalesWithHidden) {
  Config cfg(20, 6, 256);
  EXPECT_GT(
      cfg.getNvlBufferSizeHint(kHiddenBytes * 2, 8),
      cfg.getNvlBufferSizeHint(kHiddenBytes, 8));
}

// num_channels = num_sms / 2, so a larger num_sms must yield a larger hint.
// This is the term the original placeholder dropped (causing under-allocation).
TEST(ConfigBufferSizeHint, NvlHintScalesWithChannels) {
  Config few_sms(8, 6, 256);
  Config many_sms(40, 6, 256);
  EXPECT_GT(
      many_sms.getNvlBufferSizeHint(kHiddenBytes, 8),
      few_sms.getNvlBufferSizeHint(kHiddenBytes, 8));
}

// Pure-intranode groups (num_ranks <= NUM_MAX_NVL_PEERS) need no RDMA staging.
TEST(ConfigBufferSizeHint, RdmaHintZeroForIntranode) {
  Config cfg(20, 6, 256);
  EXPECT_EQ(cfg.getRdmaBufferSizeHint(kHiddenBytes, 8), 0U);
  EXPECT_EQ(cfg.getRdmaBufferSizeHint(kHiddenBytes, 2), 0U);
}

// Multi-node groups (num_ranks > NUM_MAX_NVL_PEERS) need RDMA staging.
TEST(ConfigBufferSizeHint, RdmaHintPositiveForMultinode) {
  Config cfg(20, 6, 256);
  const std::size_t rdma = cfg.getRdmaBufferSizeHint(kHiddenBytes, 16);
  EXPECT_GT(rdma, 0U);
  EXPECT_EQ(rdma % NUM_BUFFER_ALIGNMENT_BYTES, 0U);
  EXPECT_GT(
      cfg.getRdmaBufferSizeHint(kHiddenBytes * 2, 16),
      cfg.getRdmaBufferSizeHint(kHiddenBytes, 16));
}

} // namespace
} // namespace comms::prims::link_ep
