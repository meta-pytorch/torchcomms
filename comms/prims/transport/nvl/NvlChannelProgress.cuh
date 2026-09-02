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
  // The buffer activeUserBytes describes: src on a send slot, dst on a recv
  // slot, since the two directions have separate slot arrays. void* because
  // the send side is const and only ever reads through it.
  void* activeUserBuf{nullptr};
  NvlProgressStage stage{NvlProgressStage::Idle};
};

/*
 * Mirrors IbgdaSendRecvProgressStatus so one loop can drive both transports.
 *
 * `Aborted` is terminal and distinct from `Done`, matching the IB enum: the
 * operation stopped because the communicator aborted, so the remaining bytes
 * never moved. Reporting `Done` would let a driver claim success on data that
 * was never transferred, and would leave a transport-agnostic loop -- which
 * already has an abort branch for IB -- needing a transport-specific one here.
 * A driver that only needs to stop polling treats both as terminal.
 */
enum class NvlSendRecvProgressStatus : uint8_t {
  Waiting,
  Progressed,
  Done,
  Aborted,
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
