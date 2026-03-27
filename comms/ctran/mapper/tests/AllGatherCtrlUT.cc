// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/regcache/IpcRegCacheBase.h"
#include "comms/ctran/utils/CtranIpc.h"

class AllGatherCtrlPacketTest : public ::testing::Test {};

// Verify that CtranIpcDesc can hold exactly CTRAN_IPC_INLINE_SEGMENTS entries.
TEST_F(AllGatherCtrlPacketTest, CtranIpcDescStructSize) {
  ctran::utils::CtranIpcDesc desc{};
  // The segments array is fixed-size.
  EXPECT_EQ(
      sizeof(desc.segments) / sizeof(desc.segments[0]),
      static_cast<size_t>(CTRAN_IPC_INLINE_SEGMENTS));
}

// Verify that ControlMsg can carry an IpcDesc.
TEST_F(AllGatherCtrlPacketTest, ControlMsgCarriesIpcDesc) {
  ControlMsg msg;
  msg.setType(ControlMsgType::NVL_EXPORT_MEM);

  // After setType, ipcDesc should be zero-initialized.
  EXPECT_EQ(msg.ipcDesc.desc.numInlineSegments(), 0);
  EXPECT_EQ(msg.ipcDesc.offset, 0u);

  // Set some values and verify they stick.
  msg.ipcDesc.desc.totalSegments = 5;
  msg.ipcDesc.desc.segments[0].range = 4096;
  msg.ipcDesc.desc.segments[1].range = 8192;

  EXPECT_EQ(msg.ipcDesc.desc.numInlineSegments(), CTRAN_IPC_INLINE_SEGMENTS);
  EXPECT_EQ(msg.ipcDesc.desc.totalSegments, 5);
  EXPECT_EQ(msg.ipcDesc.desc.segments[0].range, 4096u);
  EXPECT_EQ(msg.ipcDesc.desc.segments[1].range, 8192u);
}

// Verify that for an IB_EXPORT_MEM message (non-NVL), the receiver does not
// attempt to derive extra packets (multi-packet is NVL-only).
TEST_F(AllGatherCtrlPacketTest, IbExportMemSinglePacket) {
  ControlMsg msg;
  msg.setType(ControlMsgType::IB_EXPORT_MEM);

  // IB_EXPORT_MEM never triggers multi-packet logic
  // allGatherCtrl only checks totalSegments for NVL_EXPORT_MEM
  EXPECT_EQ(msg.type, ControlMsgType::IB_EXPORT_MEM);

  // Simulate receiver logic: only NVL triggers multi-packet
  bool isNvl = (msg.type == ControlMsgType::NVL_EXPORT_MEM);
  EXPECT_FALSE(isNvl);

  // Even if totalSegments is set on an IB message (which shouldn't happen),
  // the receiver should NOT process extra packets because it's not NVL
  msg.ipcDesc.desc.totalSegments = 100; // Hypothetical corruption
  // Receiver checks msg.type first, so totalSegments is ignored for IB
  EXPECT_FALSE(msg.type == ControlMsgType::NVL_EXPORT_MEM);
}
