// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstddef>
#include <functional>

#include "comms/ctran/algos/common/GpeKernelSync.h"

// ─────────────────────────────────────────────────────────────────────────────
// Chunk hook framework — customization points the IB host transport's
// copy-based slot state machine drives. Lives here in
// ctran::transport (not ctran::transport::ib) because IP2pHostTransport.h
// consumes the SendChunkHooks / RecvChunkHooks structs in its
// backend-agnostic SendChunkArgs / RecvChunkArgs.
//
// The pure-zero-copy transport (ctran::transport::ib::HostZcTransport)
// does not call into hooks at all — it issues iput directly from
// iSendChunk and bumps a per-VC counter from progress(). Hooks are
// consumed by HostCbTransport.
// ─────────────────────────────────────────────────────────────────────────────

namespace ctran::transport {
using ctran::algos::GpeKernelSync;

// Context passed to every chunk hook call.
// Contains all per-chunk addressing info so hooks can compute
// source/destination pointers without external state.
struct ChunkContext {
  int slotIdx; // staging buffer slot (0..D-1)
  int round; // how many times this slot has been used
  size_t offset; // byte offset into user buffer
  size_t len; // min(chunkSize, totalSize - offset)
  void* stagingSlot; // &staging[slotIdx * chunkSize]
  const void* userBuf; // user buffer base pointer
  GpeKernelSync* sync; // this chunk's per-chunk host↔device sync
  // Flow-control (populated by transport, used by hooks)
  const uint64_t* remoteReady{nullptr}; // &remoteReady_[slotIdx]
  uint64_t* slotGeneration{nullptr}; // &slotGeneration_[slotIdx]
  void (*signalSlotReady)(void*, int){nullptr}; // transport signal trampoline
  void* signalCtx{nullptr}; // opaque transport pointer
};

// Send-side hooks — customizable per-workflow operations.
//
// The copy-based transport calls these at well-defined points in the
// per-chunk send state machine:
//   PREPARE_DATA:  hooks.prepareData(ctx)
//   WAIT_PREPARE:  hooks.isDataReady(ctx)
//   IPUT:          hooks.getLocalSrc(ctx) → source address
//   WAIT_IPUT:     hooks.onSendDone(ctx) → on completion
struct SendChunkHooks {
  // Prepare local data for IPUT.
  // CB: ctx.sync->post(round) — kicks kernel D2D userBuf[offset] → staging
  std::function<void(ChunkContext&)> prepareData;

  // Non-blocking: is data preparation done?
  // CB: ctx.sync->isComplete(round)
  std::function<bool(ChunkContext&)> isDataReady;

  // Compute IPUT source address.
  // CB: staging slot pointer
  std::function<const void*(ChunkContext&)> getLocalSrc;

  // Called when IPUT completes (optional cleanup/cascading).
  std::function<void(ChunkContext&)> onSendDone;

  // Non-blocking: can we IPUT into this remote slot?
  // CB: *ctx.remoteReady >= *ctx.slotGeneration
  std::function<bool(ChunkContext&)> isRemoteReady;
};

// Recv-side hooks — customizable per-workflow operations.
//
// The copy-based transport calls these at well-defined points in the
// per-chunk recv state machine:
//   PROCESS_DATA:  hooks.processData(ctx)
//   WAIT_PROCESS:  hooks.isProcessDone(ctx)
//   SIGNAL_READY:  hooks.onRecvDone(ctx) → on completion
struct RecvChunkHooks {
  // Process received data.
  // CB: ctx.sync->post(round) — kicks kernel D2D staging → userBuf[offset]
  std::function<void(ChunkContext&)> processData;

  // Non-blocking: is processing done?
  // CB: ctx.sync->isComplete(round)
  std::function<bool(ChunkContext&)> isProcessDone;

  // Called when recv chunk completes.
  std::function<void(ChunkContext&)> onRecvDone;

  // Signal sender that this recv slot is free for the next IPUT.
  // CB: ctx.signalSlotReady(ctx.signalCtx, ctx.slotIdx)
  std::function<void(ChunkContext&)> signalReady;
};

// ═══════════════════════════════════════════════════════════
// Built-in CB hook factories
// ═══════════════════════════════════════════════════════════

// Copy-based: uses per-chunk GpeKernelSync for host↔device D2D coordination.
// The kernel does the actual memcpy; host posts/polls via GpeKernelSync.
inline SendChunkHooks makeCopyBasedSendHooks() {
  return {
      .prepareData = [](ChunkContext& ctx) { ctx.sync->post(ctx.round); },
      .isDataReady =
          [](ChunkContext& ctx) {
            bool done = ctx.sync->isComplete(ctx.round);
            if (done) {
              ctx.sync->resetStatus();
            }
            return done;
          },
      .getLocalSrc =
          [](ChunkContext& ctx) {
            return static_cast<const void*>(ctx.stagingSlot);
          },
      .onSendDone = [](ChunkContext& ctx) { ++*ctx.slotGeneration; },
      .isRemoteReady =
          [](ChunkContext& ctx) {
            return *ctx.remoteReady >= *ctx.slotGeneration;
          },
  };
}

inline RecvChunkHooks makeCopyBasedRecvHooks() {
  return {
      .processData = [](ChunkContext& ctx) { ctx.sync->post(ctx.round); },
      .isProcessDone =
          [](ChunkContext& ctx) {
            bool done = ctx.sync->isComplete(ctx.round);
            if (done) {
              ctx.sync->resetStatus();
            }
            return done;
          },
      .onRecvDone = [](ChunkContext&) {},
      .signalReady =
          [](ChunkContext& ctx) {
            ctx.signalSlotReady(ctx.signalCtx, ctx.slotIdx);
          },
  };
}

} // namespace ctran::transport
