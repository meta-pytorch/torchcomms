// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/backends/ib/CtranIbBase.h"

class CtranIbRequestTest : public ::testing::Test {
 public:
  CtranIbRequestTest() = default;

 protected:
  void SetUp() override {
    int cudaDev = 0;
    ncclCvarInit();
    EXPECT_EQ(cudaSetDevice(cudaDev), cudaSuccess);
  }
};

TEST_F(CtranIbRequestTest, Complete) {
  CtranIbRequest req;
  auto res = req.complete();
  EXPECT_EQ(res, commSuccess);
  EXPECT_TRUE(req.isComplete());
}

TEST_F(CtranIbRequestTest, RePost) {
  CtranIbRequest req;
  auto res = req.complete();
  EXPECT_EQ(res, commSuccess);
  EXPECT_TRUE(req.isComplete());

  req.repost(2);
  EXPECT_FALSE(req.isComplete());

  res = req.complete();
  EXPECT_EQ(res, commSuccess);
  EXPECT_FALSE(req.isComplete());

  res = req.complete();
  EXPECT_EQ(res, commSuccess);
  EXPECT_TRUE(req.isComplete());
}

TEST_F(CtranIbRequestTest, SetRefCount) {
  CtranIbRequest req;

  req.setRefCount(3);
  auto res = req.complete(); // refCount reduced to 2
  EXPECT_EQ(res, commSuccess);
  EXPECT_FALSE(req.isComplete());

  res = req.complete(); // refCount reduced to 1
  EXPECT_EQ(res, commSuccess);

  res = req.complete(); // refCount reduced to 0, complete
  EXPECT_EQ(res, commSuccess);
  EXPECT_TRUE(req.isComplete());

  res = req.complete(); // refCount is less than 0, indicating a bug
  EXPECT_EQ(res, commInternalError);

  // expect the request is still completed even refcount is over-decreased.
  EXPECT_TRUE(req.isComplete());
}

TEST_F(CtranIbRequestTest, RemoteAccessKeyToString) {
  // Test empty key
  CtranIbRemoteAccessKey emptyKey;
  EXPECT_EQ(emptyKey.toString(), "");

  // Test single key
  CtranIbRemoteAccessKey singleKey;
  singleKey.rkeys[0] = 12345;
  singleKey.nKeys = 1;
  EXPECT_EQ(singleKey.toString(), "12345");

  // Test multiple keys
  CtranIbRemoteAccessKey multiKey;
  multiKey.rkeys[0] = 111;
  multiKey.rkeys[1] = 222;
  multiKey.nKeys = 2;
  EXPECT_EQ(multiKey.toString(), "111, 222");

  // Test max keys (CTRAN_MAX_IB_DEVICES_PER_RANK = 2)
  CtranIbRemoteAccessKey maxKey;
  maxKey.rkeys[0] = 0xFFFFFFFF;
  maxKey.rkeys[1] = 0x12345678;
  maxKey.nKeys = CTRAN_MAX_IB_DEVICES_PER_RANK;
  EXPECT_EQ(maxKey.toString(), "4294967295, 305419896");
}

TEST_F(CtranIbRequestTest, RemoteAccessKeyFromString) {
  // Test empty string
  auto emptyKey = CtranIbRemoteAccessKey::fromString("");
  EXPECT_EQ(emptyKey.nKeys, 0);

  // Test single key
  auto singleKey = CtranIbRemoteAccessKey::fromString("12345");
  EXPECT_EQ(singleKey.nKeys, 1);
  EXPECT_EQ(singleKey.rkeys[0], 12345);

  // Test multiple keys without spaces
  auto multiKey1 = CtranIbRemoteAccessKey::fromString("111,222");
  EXPECT_EQ(multiKey1.nKeys, 2);
  EXPECT_EQ(multiKey1.rkeys[0], 111);
  EXPECT_EQ(multiKey1.rkeys[1], 222);

  // Test multiple keys with spaces
  auto multiKey2 = CtranIbRemoteAccessKey::fromString("333, 444");
  EXPECT_EQ(multiKey2.nKeys, 2);
  EXPECT_EQ(multiKey2.rkeys[0], 333);
  EXPECT_EQ(multiKey2.rkeys[1], 444);

  // Test multiple keys with extra spaces
  auto multiKey3 = CtranIbRemoteAccessKey::fromString(" 555 , 666 ");
  EXPECT_EQ(multiKey3.nKeys, 2);
  EXPECT_EQ(multiKey3.rkeys[0], 555);
  EXPECT_EQ(multiKey3.rkeys[1], 666);

  // Test max value keys
  auto maxKey = CtranIbRemoteAccessKey::fromString("4294967295,305419896");
  EXPECT_EQ(maxKey.nKeys, 2);
  EXPECT_EQ(maxKey.rkeys[0], 0xFFFFFFFF);
  EXPECT_EQ(maxKey.rkeys[1], 0x12345678);
}

TEST_F(CtranIbRequestTest, RemoteAccessKeyFromStringErrors) {
  // Test too many keys (more than CTRAN_MAX_IB_DEVICES_PER_RANK = 2)
  EXPECT_THROW(
      CtranIbRemoteAccessKey::fromString("111,222,333"), std::invalid_argument);

  // Test invalid number format
  EXPECT_THROW(
      CtranIbRemoteAccessKey::fromString("invalid"), folly::ConversionError);

  // Test mixed valid and invalid
  EXPECT_THROW(
      CtranIbRemoteAccessKey::fromString("123,invalid"),
      folly::ConversionError);

  // Test negative number (should throw since we're parsing uint32_t)
  EXPECT_THROW(
      CtranIbRemoteAccessKey::fromString("-123"), folly::ConversionError);

  // Test number too large for uint32_t
  EXPECT_THROW(
      CtranIbRemoteAccessKey::fromString("4294967296"), folly::ConversionError);
}

TEST_F(CtranIbRequestTest, RemoteAccessKeyRoundTrip) {
  // Test round-trip consistency: toString -> fromString should be identity

  // Test single key
  CtranIbRemoteAccessKey original1;
  original1.rkeys[0] = 98765;
  original1.nKeys = 1;
  auto roundTrip1 = CtranIbRemoteAccessKey::fromString(original1.toString());
  EXPECT_EQ(roundTrip1.nKeys, original1.nKeys);
  EXPECT_EQ(roundTrip1.rkeys[0], original1.rkeys[0]);

  // Test multiple keys
  CtranIbRemoteAccessKey original2;
  original2.rkeys[0] = 111111;
  original2.rkeys[1] = 222222;
  original2.nKeys = 2;
  auto roundTrip2 = CtranIbRemoteAccessKey::fromString(original2.toString());
  EXPECT_EQ(roundTrip2.nKeys, original2.nKeys);
  EXPECT_EQ(roundTrip2.rkeys[0], original2.rkeys[0]);
  EXPECT_EQ(roundTrip2.rkeys[1], original2.rkeys[1]);

  // Test edge case values
  CtranIbRemoteAccessKey original3;
  original3.rkeys[0] = 0;
  original3.rkeys[1] = 0xFFFFFFFF;
  original3.nKeys = 2;
  auto roundTrip3 = CtranIbRemoteAccessKey::fromString(original3.toString());
  EXPECT_EQ(roundTrip3.nKeys, original3.nKeys);
  EXPECT_EQ(roundTrip3.rkeys[0], original3.rkeys[0]);
  EXPECT_EQ(roundTrip3.rkeys[1], original3.rkeys[1]);
}
