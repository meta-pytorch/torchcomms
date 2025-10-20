// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/commSpecs.h"
#include "comms/utils/Conversion.h"

#include <collectives.h>
#include <gtest/gtest.h>
#include <nccl.h>

// This test verifies that the sizes of ncclDataType_t types match the
// corresponding commDataType_t types. This is important for ensuring
// compatibility between the two type systems as we migrate from ncclDataType_t
// to commDataType_t.

TEST(DataTypeSizeTest, NcclAndMetaCommDataTypeSizesMatch) {
  // Test each data type to ensure the sizes match
  EXPECT_EQ(ncclTypeSize(ncclChar), commTypeSize(commChar));
  EXPECT_EQ(ncclTypeSize(ncclInt8), commTypeSize(commInt8));
  EXPECT_EQ(ncclTypeSize(ncclUint8), commTypeSize(commUint8));
  EXPECT_EQ(ncclTypeSize(ncclInt), commTypeSize(commInt));
  EXPECT_EQ(ncclTypeSize(ncclInt32), commTypeSize(commInt32));
  EXPECT_EQ(ncclTypeSize(ncclUint32), commTypeSize(commUint32));
  EXPECT_EQ(ncclTypeSize(ncclInt64), commTypeSize(commInt64));
  EXPECT_EQ(ncclTypeSize(ncclUint64), commTypeSize(commUint64));
  EXPECT_EQ(ncclTypeSize(ncclHalf), commTypeSize(commHalf));
  EXPECT_EQ(ncclTypeSize(ncclFloat16), commTypeSize(commFloat16));
  EXPECT_EQ(ncclTypeSize(ncclFloat), commTypeSize(commFloat));
  EXPECT_EQ(ncclTypeSize(ncclFloat32), commTypeSize(commFloat32));
  EXPECT_EQ(ncclTypeSize(ncclDouble), commTypeSize(commDouble));
  EXPECT_EQ(ncclTypeSize(ncclFloat64), commTypeSize(commFloat64));
  EXPECT_EQ(ncclTypeSize(ncclBfloat16), commTypeSize(commBfloat16));

  EXPECT_EQ(ncclTypeSize(ncclFloat8e4m3), commTypeSize(commFloat8e4m3));
  EXPECT_EQ(ncclTypeSize(ncclFloat8e5m2), commTypeSize(commFloat8e5m2));
}
