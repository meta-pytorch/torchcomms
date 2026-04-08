// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

namespace comms::pipes {

// Maximum number of NIC rails for multi-NIC support.
// Each rail has its own PD and MR, so local buffers need per-rail lkeys.
constexpr int kMaxIbgdaRails = 2;

} // namespace comms::pipes
