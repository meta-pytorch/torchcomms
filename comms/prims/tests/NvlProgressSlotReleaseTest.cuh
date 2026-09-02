// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

namespace comms::prims::test {

/// Host-visible copy of one `NvlChannelProgress` slot plus the channel cursor
/// it reserved from, so the .cc can assert without including any transport
/// header.
struct NvlSlotProbe {
  int stage;
  unsigned long long baseByte;
  unsigned long long nextByte;
  unsigned long long payloadBytes;
  unsigned long long tailPadding;
  unsigned long long userBytes;
  unsigned long long maxSignalBytes;
  long long sendCursor;
};

/// `NvlProgressStage` mirrored as plain ints, same reason. Static-asserted
/// against the enum in the .cu.
constexpr int kNvlStageIdle = 0;
constexpr int kNvlStageActive = 1;

/// Writes a mid-transfer progress slot into device memory and reads it back
/// untouched, so a test can show the staged state is one that
/// `assert_progress_slot_idle()` would reject.
NvlSlotProbe stageNvlSlot();

/// Stages the same mid-transfer slot, runs `abandon_progress_state()` over it,
/// then -- from a SECOND kernel launch -- runs the real
/// `assert_progress_slot_idle()` and reads the slot back.
///
/// The second launch is the point. The trap this containment property exists
/// to remove does not fire in the kernel that aborted; it fires in the next
/// kernel queued on the same channel, when its `init_send_progress()` finds the
/// slot still mid-transfer. Running the check in the same kernel would not
/// exercise that.
NvlSlotProbe abandonNvlSlotThenReinit();

} // namespace comms::prims::test
