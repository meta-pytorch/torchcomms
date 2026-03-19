// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::pipes::test {

/// Launch LL128 send kernel with a short timeout and no receiver.
/// The sender will poll for ACKs that never arrive, triggering __trap().
void launch_ll128_send_no_recv_timeout(int device, uint32_t timeout_ms);

} // namespace comms::pipes::test
