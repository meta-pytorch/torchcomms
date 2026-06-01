// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/transport/ib/ChunkHooks.h"

using namespace ctran::transport;

class ChunkHooksTest : public ::testing::Test {};

// Mixed-mode and pure-ZC hook factories were removed when the host-IB
// transport was split into pure-ZC + pure-CB classes — pure-ZC no
// longer goes through the hook framework at all. Only the CB-side
// factories remain, so this UT exercises the surviving CB hooks plus
// the ChunkContext shape.

TEST_F(ChunkHooksTest, CopyBasedSendHooks_GetLocalSrcReturnsStagingSlot) {
  auto hooks = makeCopyBasedSendHooks();
  char fakeStaging[4096];
  ChunkContext ctx{};
  ctx.stagingSlot = fakeStaging;
  const void* src = hooks.getLocalSrc(ctx);
  EXPECT_EQ(src, fakeStaging);
}

TEST_F(ChunkHooksTest, ChunkContext_SlotAndRound) {
  constexpr int D = 2;
  for (int round = 0; round < 4; ++round) {
    for (int slot = 0; slot < D; ++slot) {
      ChunkContext ctx{};
      ctx.slotIdx = slot;
      ctx.round = round;
      EXPECT_EQ(ctx.slotIdx, slot);
      EXPECT_EQ(ctx.round, round);
    }
  }
}

TEST_F(ChunkHooksTest, CustomHooks_CountInvocations) {
  int prepareCount = 0;
  int readyCount = 0;
  int doneCount = 0;

  SendChunkHooks customHooks{
      .prepareData = [&](ChunkContext&) { prepareCount++; },
      .isDataReady =
          [&](ChunkContext&) {
            readyCount++;
            return true;
          },
      .getLocalSrc =
          [](ChunkContext& ctx) {
            return static_cast<const void*>(ctx.stagingSlot);
          },
      .onSendDone = [&](ChunkContext&) { doneCount++; },
  };

  ChunkContext ctx{};
  customHooks.prepareData(ctx);
  customHooks.prepareData(ctx);
  customHooks.isDataReady(ctx);
  customHooks.onSendDone(ctx);

  EXPECT_EQ(prepareCount, 2);
  EXPECT_EQ(readyCount, 1);
  EXPECT_EQ(doneCount, 1);
}
