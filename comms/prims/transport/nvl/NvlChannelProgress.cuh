// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <cstddef>
#include <cstdint>

namespace comms::prims {

enum class NvlProgressStage : uint8_t {
  Idle,
  Active,
};

/*
 * Resumable state for one in-flight NVL send or recv, held per
 * (peer, channel, direction). Device-local: never IPC-exchanged, because a
 * rank's progress through its own operation is not read by the peer.
 *
 * Concurrency: one operation per (peer, channel, direction) at a time, the same
 * contract the blocking send()/recv() rely on. A second progress init traps,
 * but the blocking paths do not inspect `stage`, so interleaving a blocking op
 * with an in-flight progress op on one channel is caller error and goes
 * undetected.
 */
struct NvlChannelProgress {
  // Stream offset reserved at init.
  uint64_t activeBaseByte{0};
  // How far this operation has advanced. Protocol bytes, so a resumed step
  // lands on the chunk boundary the blocking loop would have used.
  std::size_t activeNextByte{0};
  std::size_t activePayloadBytes{0};
  std::size_t activeTailPadding{0};
  std::size_t activeUserBytes{0};
  std::size_t activeMaxSignalBytes{0};
  NvlProgressStage stage{NvlProgressStage::Idle};
};

/*
 * Mirrors IbgdaSendRecvProgressStatus so one loop can drive both transports.
 *
 * Done also covers abort: a wait terminates itself once abort is visible, so
 * the caller's scheduler drains and exits rather than spinning on a channel
 * that will never progress.
 */
enum class NvlSendRecvProgressStatus : uint8_t {
  Waiting,
  Progressed,
  Done,
};

/*
 * Leader verdict for a readiness poll, broadcast as one value. Folding abort
 * into the same verdict keeps an unsuccessful poll to a single broadcast; two
 * broadcasts would cost four group barriers on the path a resumable caller
 * re-enters most often.
 */
inline constexpr uint32_t kNvlProgressWaiting = 0U;
inline constexpr uint32_t kNvlProgressReady = 1U;
inline constexpr uint32_t kNvlProgressAborted = 2U;

} // namespace comms::prims
