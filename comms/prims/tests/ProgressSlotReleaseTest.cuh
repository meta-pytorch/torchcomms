// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

namespace comms::prims::test {

/// Host-visible copy of the fields of one `IbChannelProgress` slot, so the .cc
/// can assert on the slot without including any transport header.
struct SlotProbe {
  int stage;
  unsigned long long nextByte;
  unsigned long long tailPadding;
  long long baseStep;
  long long nextStep;
};

/// `detail::IbSendRecvProgressStage` mirrored as plain ints, same reason.
/// Static-asserted against the enum in the .cu.
constexpr int kStageDone = 0;
constexpr int kStageWaitLocalCompletion = 1;
constexpr int kStageWaitSlotFree = 2;
constexpr int kStageWaitDataReady = 3;

/// Writes a mid-transfer progress slot into device memory and reads it back
/// without touching it, so a test can show the staged state is one that
/// `assert_progress_slot_idle()` would reject.
SlotProbe stage_slot(int startStage, std::size_t startNextByte);

/// Stages the same mid-transfer slot, runs `abandon_progress_state()` over it,
/// then -- from a SECOND kernel launch -- runs `assert_progress_slot_idle()`
/// and reads the slot back.
///
/// The second launch is the whole point. The trap this containment change
/// exists to remove does not fire in the kernel that aborted; it fires in the
/// next kernel queued on the same channel, when its `init_send_progress()`
/// finds the slot still mid-transfer. Running the assert in the same kernel
/// would not exercise that.
SlotProbe abandon_then_reinit(int startStage, std::size_t startNextByte);

} // namespace comms::prims::test
