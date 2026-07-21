// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Unit test for ColltraceEventScope: the RAII scope must emit exactly the armed
// start/end colltrace events into the HRDW ring, elected to a single writer
// regardless of block/thread count, and be a no-op when unarmed.

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <vector>

#include <gtest/gtest.h>

#include "comms/utils/colltrace/ColltraceDeviceHandle.h"
#include "comms/utils/colltrace/GraphCollTraceEvent.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBufferReader.h"

using meta::comms::colltrace::ColltraceDeviceHandle;
using meta::comms::colltrace::GraphCollTraceEvent;
using meta::comms::colltrace::GraphCollTracePhase;

// Defined in ColltraceEventScopeTest.cu.
void launchColltraceScopeKernel(
    const ColltraceDeviceHandle& handle,
    int blocks,
    int threads,
    cudaStream_t stream);

namespace {

// Runs one ColltraceEventScope with the given emit flags across blocks*threads
// threads, then returns the events written to the ring in write order.
std::vector<GraphCollTraceEvent> runScope(
    bool emitStart,
    bool emitEnd,
    uint32_t collId,
    int blocks,
    int threads) {
  hrdw_ring_buffer::HRDWRingBuffer<GraphCollTraceEvent> buf(/*capacity=*/64);
  EXPECT_TRUE(buf.valid());

  ColltraceDeviceHandle handle{};
  handle.collId = collId;
  handle.emitStart = emitStart;
  handle.emitEnd = emitEnd;
  handle.ring = buf.deviceHandle();

  cudaStream_t stream = nullptr;
  EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  launchColltraceScopeKernel(handle, blocks, threads, stream);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  std::vector<GraphCollTraceEvent> events;
  hrdw_ring_buffer::HRDWRingBufferReader<GraphCollTraceEvent> reader(buf);
  reader.poll([&](const auto& entry, uint64_t /*slot*/) {
    events.push_back(entry.data);
  });
  return events;
}

TEST(ColltraceEventScopeTest, EmitsStartThenEndSingleWriter) {
  // 64 threads all construct the scope, but the emit is elected to one writer,
  // so exactly one start and one end land in the ring.
  auto events =
      runScope(/*emitStart=*/true, /*emitEnd=*/true, /*collId=*/42, 2, 32);
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].phase, GraphCollTracePhase::kStart);
  EXPECT_EQ(events[0].collId, 42u);
  EXPECT_EQ(events[1].phase, GraphCollTracePhase::kEnd);
  EXPECT_EQ(events[1].collId, 42u);
}

TEST(ColltraceEventScopeTest, StartOnly) {
  auto events =
      runScope(/*emitStart=*/true, /*emitEnd=*/false, /*collId=*/7, 1, 1);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].phase, GraphCollTracePhase::kStart);
  EXPECT_EQ(events[0].collId, 7u);
}

TEST(ColltraceEventScopeTest, EndOnly) {
  auto events =
      runScope(/*emitStart=*/false, /*emitEnd=*/true, /*collId=*/8, 1, 1);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].phase, GraphCollTracePhase::kEnd);
  EXPECT_EQ(events[0].collId, 8u);
}

TEST(ColltraceEventScopeTest, UnarmedIsNoOp) {
  auto events =
      runScope(/*emitStart=*/false, /*emitEnd=*/false, /*collId=*/9, 4, 64);
  EXPECT_TRUE(events.empty());
}

} // namespace
