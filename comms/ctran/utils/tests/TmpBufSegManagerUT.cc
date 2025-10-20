// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include "comms/ctran/utils/TmpBufSegManager.h"

using ctran::utils::TmpBufSegManager;

TEST(TmpBufSegManagerTest, Basic) {
  enum class TestBufTypes { kBuf1, kBuf2, kBuf3, kNumBufs };

  TmpBufSegManager<TestBufTypes, TestBufTypes::kNumBufs> segMgr;
  ASSERT_TRUE(segMgr.insert(TestBufTypes::kBuf1, 2));
  ASSERT_TRUE(segMgr.insert(TestBufTypes::kBuf2, 8199));
  ASSERT_TRUE(segMgr.insert(TestBufTypes::kBuf3, 518));
  ASSERT_EQ(segMgr.totalLen, 4096 * 5);

  auto segInfo1 = segMgr.getSegInfo(TestBufTypes::kBuf1);
  ASSERT_EQ(segInfo1.len, 4096);
  ASSERT_EQ(segInfo1.offset, 0);

  auto segInfo2 = segMgr.getSegInfo(TestBufTypes::kBuf2);
  ASSERT_EQ(segInfo2.len, 4096 * 3);
  ASSERT_EQ(segInfo2.offset, 4096);

  auto segInfo3 = segMgr.getSegInfo(TestBufTypes::kBuf3);
  ASSERT_EQ(segInfo3.len, 4096);
  ASSERT_EQ(segInfo3.offset, 4096 * 4);
}

TEST(TmpBufSegManagerTest, InvalidExceed) {
  enum class TestBufTypes { kBuf1, kBuf2, kBuf3, kNumBufs };

  TmpBufSegManager<TestBufTypes, TestBufTypes::kNumBufs> segMgr;
  ASSERT_TRUE(segMgr.insert(TestBufTypes::kBuf1, 2));
  ASSERT_TRUE(segMgr.insert(TestBufTypes::kBuf3, 518));

  ASSERT_FALSE(segMgr.insert(TestBufTypes::kNumBufs, 16));
  ASSERT_FALSE(
      segMgr.insert((TestBufTypes)((int)TestBufTypes::kNumBufs + 1), 16));
}

TEST(TmpBufSegManagerTest, Contains) {
  enum class TestBufTypes { kBuf1, kBuf2, kBuf3, kNumBufs };

  TmpBufSegManager<TestBufTypes, TestBufTypes::kNumBufs> segMgr;
  ASSERT_TRUE(segMgr.insert(TestBufTypes::kBuf1, 2));
  ASSERT_TRUE(segMgr.insert(TestBufTypes::kBuf3, 518));

  ASSERT_TRUE(segMgr.contains(TestBufTypes::kBuf1));
  ASSERT_TRUE(segMgr.contains(TestBufTypes::kBuf3));
  ASSERT_FALSE(segMgr.contains(TestBufTypes::kBuf2));
}
