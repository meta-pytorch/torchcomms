// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/HostCollectiveEngine.h"

#include <gtest/gtest.h>

namespace comms::prims {
namespace {

/*
 * Covers the stream retargeting that every enqueue keys off. The copy and
 * stream-memop paths themselves are exercised by the collectives that drive
 * real hardware.
 *
 * No device and no driver needed: the engine only stores the handle, and
 * nothing here enqueues work, so opaque values exercise the retarget on their
 * own. Resolving the driver entry points is the caller's job -- see the class
 * comment -- and is irrelevant until something actually enqueues.
 */
TEST(HostCollectiveEngineTest, SetStreamRetargetsTheEngine) {
  auto* first = reinterpret_cast<cudaStream_t>(0x1);
  auto* second = reinterpret_cast<cudaStream_t>(0x2);

  HostCollectiveEngine engine(first);
  EXPECT_EQ(engine.stream(), first);

  engine.setStream(second);
  EXPECT_EQ(engine.stream(), second);
}

} // namespace
} // namespace comms::prims
