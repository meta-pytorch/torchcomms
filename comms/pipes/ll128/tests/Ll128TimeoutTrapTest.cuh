// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::pipes::test {

/// Launch LL128 send kernel with a short timeout and no receiver.
/// The sender will poll for ACKs that never arrive, triggering __trap().
void launch_ll128_send_no_recv_timeout(int device, uint32_t timeout_ms);

/// Launch LL128 send/recv with undersized buffer (buffer_num_packets = 2,
/// below kLl128PacketsPerWarp = 4). Should trigger PIPES_DEVICE_CHECK
/// and __trap().
void launch_ll128_send_recv_undersized_buffer(int device);

} // namespace comms::pipes::test
