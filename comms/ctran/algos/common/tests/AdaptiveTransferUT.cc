// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/algos/common/AdaptiveTransfer.h"
#include "comms/ctran/algos/common/AdaptiveTransferTypes.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/regcache/RegCache.h"

using namespace ctran::algos;

class AdaptiveTransferTypesTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test TransferDirection enum values
TEST_F(AdaptiveTransferTypesTest, TransferDirectionValues) {
  EXPECT_NE(
      static_cast<int>(TransferDirection::SEND),
      static_cast<int>(TransferDirection::RECV));
}

// Test DirectionalTransferConfig default construction
TEST_F(AdaptiveTransferTypesTest, DirectionalTransferConfigDefaults) {
  DirectionalTransferConfig config;

  // Default should be ZERO_COPY for backward compatibility
  EXPECT_EQ(config.mode, TransferMode::ZERO_COPY);
  EXPECT_EQ(config.memHdl, nullptr);
  EXPECT_TRUE(config.isZeroCopy());
}

// Test DirectionalTransferConfig with COPY_BASED mode
TEST_F(AdaptiveTransferTypesTest, DirectionalTransferConfigCopyBased) {
  DirectionalTransferConfig config;
  config.mode = TransferMode::COPY_BASED;

  EXPECT_EQ(config.mode, TransferMode::COPY_BASED);
  EXPECT_FALSE(config.isZeroCopy());
}

// Test DirectionalTransferConfig with memHdl set
TEST_F(AdaptiveTransferTypesTest, DirectionalTransferConfigWithMemHdl) {
  DirectionalTransferConfig config;
  void* fakeHandle = reinterpret_cast<void*>(0x12345678);
  config.memHdl = fakeHandle;

  EXPECT_EQ(config.memHdl, fakeHandle);
  EXPECT_TRUE(config.isZeroCopy());
}

// Test TransferConfig default construction
TEST_F(AdaptiveTransferTypesTest, TransferConfigDefaults) {
  TransferConfig config;

  // Both send and recv should default to ZERO_COPY
  EXPECT_TRUE(config.send.isZeroCopy());
  EXPECT_TRUE(config.recv.isZeroCopy());
  EXPECT_TRUE(config.isSendZeroCopy());
  EXPECT_TRUE(config.isRecvZeroCopy());
  EXPECT_EQ(config.send.memHdl, nullptr);
  EXPECT_EQ(config.recv.memHdl, nullptr);
}

// Test TransferConfig with different send/recv modes
TEST_F(AdaptiveTransferTypesTest, TransferConfigDifferentModes) {
  TransferConfig config;

  // Set send to COPY_BASED, recv stays ZERO_COPY
  config.send.mode = TransferMode::COPY_BASED;

  EXPECT_FALSE(config.isSendZeroCopy());
  EXPECT_TRUE(config.isRecvZeroCopy());

  // Set recv to COPY_BASED as well
  config.recv.mode = TransferMode::COPY_BASED;

  EXPECT_FALSE(config.isSendZeroCopy());
  EXPECT_FALSE(config.isRecvZeroCopy());
}

// Test TransferConfig with different memHdls
TEST_F(AdaptiveTransferTypesTest, TransferConfigDifferentMemHdls) {
  TransferConfig config;
  void* sendHandle = reinterpret_cast<void*>(0x11111111);
  void* recvHandle = reinterpret_cast<void*>(0x22222222);

  config.send.memHdl = sendHandle;
  config.recv.memHdl = recvHandle;

  EXPECT_EQ(config.send.memHdl, sendHandle);
  EXPECT_EQ(config.recv.memHdl, recvHandle);
  EXPECT_NE(config.send.memHdl, config.recv.memHdl);
}

// Test TransferMode enum values
TEST_F(AdaptiveTransferTypesTest, TransferModeValues) {
  EXPECT_NE(
      static_cast<int>(TransferMode::ZERO_COPY),
      static_cast<int>(TransferMode::COPY_BASED));
}

// Test isZeroCopy helper for both modes
TEST_F(AdaptiveTransferTypesTest, IsZeroCopyHelper) {
  DirectionalTransferConfig zeroCopyConfig;
  zeroCopyConfig.mode = TransferMode::ZERO_COPY;
  EXPECT_TRUE(zeroCopyConfig.isZeroCopy());

  DirectionalTransferConfig copyBasedConfig;
  copyBasedConfig.mode = TransferMode::COPY_BASED;
  EXPECT_FALSE(copyBasedConfig.isZeroCopy());
}

// Test TransferConfig convenience methods match underlying config
TEST_F(AdaptiveTransferTypesTest, TransferConfigConvenienceMethods) {
  TransferConfig config;

  // Test that convenience methods match direct access
  EXPECT_EQ(config.isSendZeroCopy(), config.send.isZeroCopy());
  EXPECT_EQ(config.isRecvZeroCopy(), config.recv.isZeroCopy());

  config.send.mode = TransferMode::COPY_BASED;
  EXPECT_EQ(config.isSendZeroCopy(), config.send.isZeroCopy());
  EXPECT_EQ(config.isRecvZeroCopy(), config.recv.isZeroCopy());

  config.recv.mode = TransferMode::COPY_BASED;
  EXPECT_EQ(config.isSendZeroCopy(), config.send.isZeroCopy());
  EXPECT_EQ(config.isRecvZeroCopy(), config.recv.isZeroCopy());
}
