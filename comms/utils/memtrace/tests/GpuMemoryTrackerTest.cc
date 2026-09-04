// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <thread>

#include "comms/utils/memtrace/GpuMemoryTracker.h"

namespace meta::comms::memtrace {
namespace {

constexpr GpuMemoryContext kContext{
    .commHash = 0x1234,
    .rank = 2,
    .device = 1,
};

std::size_t index(GpuMemoryComponent component) {
  return static_cast<std::size_t>(component);
}

std::size_t index(GpuMemoryResourceType resource) {
  return static_cast<std::size_t>(resource);
}

TEST(GpuMemoryTrackerTest, AttributesUsageAndTracksPeak) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  ScopedGpuMemoryContext scope{gpuMemoryTracker.get()};
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kNvlP2pDataStaging, 0x1000, 100, 128);
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kIbSendRecvPeerBulk, 0x2000, 200, 256);

  auto snapshot = gpuMemoryTracker->snapshot();
  EXPECT_EQ(snapshot.context, kContext);
  const GpuMemoryUsage expectedUsageAfterAllocation{
      .currentLogicalBytes = 300,
      .currentBytes = 384,
      .peakBytes = 384,
      .totalAllocatedBytes = 384,
  };
  EXPECT_EQ(snapshot.total, expectedUsageAfterAllocation);
  EXPECT_EQ(snapshot.prims, expectedUsageAfterAllocation);
  EXPECT_EQ(snapshot.activeAllocations, 2);
  EXPECT_EQ(
      snapshot.components[index(GpuMemoryComponent::kNvl)].currentBytes, 128);
  EXPECT_EQ(
      snapshot.components[index(GpuMemoryComponent::kIb)].currentBytes, 256);
  EXPECT_EQ(
      snapshot.resources[index(GpuMemoryResourceType::kNvlP2pDataStaging)]
          .currentLogicalBytes,
      100);

  recordGpuMemoryFree(0x1000);
  snapshot = gpuMemoryTracker->snapshot();
  const GpuMemoryUsage expectedUsageAfterFree{
      .currentLogicalBytes = 200,
      .currentBytes = 256,
      .peakBytes = 384,
      .totalAllocatedBytes = 384,
      .totalFreedBytes = 128,
  };
  EXPECT_EQ(snapshot.total, expectedUsageAfterFree);
  EXPECT_EQ(snapshot.prims, expectedUsageAfterFree);
  EXPECT_EQ(snapshot.activeAllocations, 1);
  recordGpuMemoryFree(0x2000);
}

TEST(GpuMemoryTrackerTest, AccountsUnclassifiedMemoryOutsidePrims) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  ScopedGpuMemoryContext scope{gpuMemoryTracker.get()};
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kUnclassified, 0x1050, 128, 128);

  const auto snapshot = gpuMemoryTracker->snapshot();
  const GpuMemoryUsage expectedUsage{
      .currentLogicalBytes = 128,
      .currentBytes = 128,
      .peakBytes = 128,
      .totalAllocatedBytes = 128,
  };
  EXPECT_EQ(snapshot.total, expectedUsage);
  EXPECT_EQ(snapshot.prims, GpuMemoryUsage{});
  EXPECT_EQ(
      snapshot.components[index(GpuMemoryComponent::kUnclassified)],
      expectedUsage);
  EXPECT_EQ(
      snapshot.resources[index(GpuMemoryResourceType::kUnclassified)],
      expectedUsage);
  const auto report = formatGpuMemorySnapshot("test", snapshot);
  EXPECT_NE(
      report.find("unclassified GPU memory: current=0.00 MiB (128 bytes)"),
      std::string::npos);
  EXPECT_NE(
      report.find("resource unclassified: current=0.00 MiB (128 bytes)"),
      std::string::npos);

  recordGpuMemoryFree(0x1050);
}

TEST(
    GpuMemoryTrackerTest,
    PrimsPeakTracksConcurrentUsageRatherThanComponentPeaks) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  ScopedGpuMemoryContext scope{gpuMemoryTracker.get()};
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kNvlP2pDataStaging, 0x1100, 128, 128);
  recordGpuMemoryFree(0x1100);
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kIbSendRecvPeerBulk, 0x1200, 256, 256);
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kMcclAllGatherOverlapCounters, 0x1300, 64, 64);

  const auto snapshot = gpuMemoryTracker->snapshot();
  const GpuMemoryUsage expectedPrimsUsage{
      .currentLogicalBytes = 256,
      .currentBytes = 256,
      .peakBytes = 256,
      .totalAllocatedBytes = 384,
      .totalFreedBytes = 128,
  };
  const GpuMemoryUsage expectedTotalUsage{
      .currentLogicalBytes = 320,
      .currentBytes = 320,
      .peakBytes = 320,
      .totalAllocatedBytes = 448,
      .totalFreedBytes = 128,
  };
  EXPECT_EQ(snapshot.prims, expectedPrimsUsage);
  EXPECT_EQ(snapshot.total, expectedTotalUsage);
  EXPECT_EQ(
      snapshot.components[index(GpuMemoryComponent::kNvl)].peakBytes, 128);
  EXPECT_EQ(snapshot.components[index(GpuMemoryComponent::kIb)].peakBytes, 256);
  recordGpuMemoryFree(0x1200);
  recordGpuMemoryFree(0x1300);
}

TEST(GpuMemoryTrackerTest, RejectsDuplicateBackingWithoutDoubleCounting) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  ScopedGpuMemoryContext scope{gpuMemoryTracker.get()};
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kCommonTransportDispatchTable, 0x3000, 64, 64);
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kIbgdaAtomicReturnSink, 0x3000, 64, 64);

  auto snapshot = gpuMemoryTracker->snapshot();
  EXPECT_EQ(snapshot.total.currentBytes, 64);
  EXPECT_EQ(snapshot.activeAllocations, 1);
  EXPECT_EQ(snapshot.duplicateAllocations, 1);

  recordGpuMemoryFree(0x3000);
  EXPECT_EQ(gpuMemoryTracker->snapshot().total.currentBytes, 0);
}

TEST(GpuMemoryTrackerTest, SeparatesRuntimePointersFromVmmHandles) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  ScopedGpuMemoryContext scope{gpuMemoryTracker.get()};
  constexpr uintptr_t kSharedNumericId = 0x3800;

  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kCommonTransportDispatchTable,
      kSharedNumericId,
      64,
      64);
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kNvlMultimemCombinedBacking,
      kSharedNumericId,
      128,
      128,
      GpuMemoryBackingKind::kVirtualMemoryHandle);

  auto snapshot = gpuMemoryTracker->snapshot();
  EXPECT_EQ(snapshot.total.currentBytes, 192);
  EXPECT_EQ(snapshot.activeAllocations, 2);
  EXPECT_EQ(snapshot.duplicateAllocations, 0);

  recordGpuMemoryFree(kSharedNumericId);
  snapshot = gpuMemoryTracker->snapshot();
  EXPECT_EQ(snapshot.total.currentBytes, 128);

  recordGpuMemoryFree(
      kSharedNumericId, GpuMemoryBackingKind::kVirtualMemoryHandle);
  EXPECT_EQ(gpuMemoryTracker->snapshot().total.currentBytes, 0);
}

TEST(GpuMemoryTrackerTest, FreeLooksUpBackingAndOnlyRecordsOnce) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  ScopedGpuMemoryContext scope{gpuMemoryTracker.get()};
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kIbgdaPeerTransportArray, 0x4000, 96, 128);

  recordGpuMemoryFree(0x4001);
  EXPECT_EQ(gpuMemoryTracker->snapshot().total.currentBytes, 128);

  recordGpuMemoryFree(0x4000);
  recordGpuMemoryFree(0x4000);

  const auto snapshot = gpuMemoryTracker->snapshot();
  const GpuMemoryUsage expectedTotalUsage{
      .peakBytes = 128,
      .totalAllocatedBytes = 128,
      .totalFreedBytes = 128,
  };
  EXPECT_EQ(snapshot.total, expectedTotalUsage);
  EXPECT_EQ(snapshot.activeAllocations, 0);
}

TEST(GpuMemoryTrackerTest, MissingFreeKeepsAllocationAccounted) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  ScopedGpuMemoryContext scope{gpuMemoryTracker.get()};
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kCommonTransportDispatchTable, 0x4500, 64, 128);

  const auto snapshot = gpuMemoryTracker->snapshot();
  const GpuMemoryUsage expectedTotalUsage{
      .currentLogicalBytes = 64,
      .currentBytes = 128,
      .peakBytes = 128,
      .totalAllocatedBytes = 128,
  };
  EXPECT_EQ(snapshot.total, expectedTotalUsage);
  EXPECT_EQ(snapshot.activeAllocations, 1);
  recordGpuMemoryFree(0x4500);
}

TEST(GpuMemoryTrackerTest, NullTrackerIsNoOp) {
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kCommonTransportDispatchTable, 0x5000, 64, 64);
  recordGpuMemoryFree(0x5000);
}

TEST(GpuMemoryTrackerTest, UnspecifiedResourceIsNotTracked) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  ScopedGpuMemoryContext scope{gpuMemoryTracker.get()};

  recordGpuMemoryAllocation(GpuMemoryResourceType::kCount, 0x5050, 64, 64);
  recordGpuMemoryFree(0x5050);

  EXPECT_EQ(gpuMemoryTracker->snapshot().activeAllocations, 0);
  EXPECT_EQ(gpuMemoryTracker->snapshot().total.currentBytes, 0);
}

TEST(GpuMemoryTrackerTest, ScopedTrackerRestoresNestedBinding) {
  auto outerTracker = std::make_shared<GpuMemoryTracker>(kContext);
  auto innerTracker = std::make_shared<GpuMemoryTracker>(kContext);

  {
    ScopedGpuMemoryContext outerScope{outerTracker.get()};
    recordGpuMemoryAllocation(
        GpuMemoryResourceType::kCommonTransportDispatchTable, 0x5100, 64, 64);
    {
      ScopedGpuMemoryContext innerScope{innerTracker.get()};
      recordGpuMemoryAllocation(
          GpuMemoryResourceType::kIbEagerControlBuffer, 0x5200, 128, 128);
    }
    recordGpuMemoryAllocation(
        GpuMemoryResourceType::kNvlP2pSignal, 0x5300, 32, 32);
  }
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kIbgdaAtomicReturnSink, 0x5400, 256, 256);

  EXPECT_EQ(outerTracker->snapshot().total.currentBytes, 96);
  EXPECT_EQ(outerTracker->snapshot().activeAllocations, 2);
  EXPECT_EQ(innerTracker->snapshot().total.currentBytes, 128);
  EXPECT_EQ(innerTracker->snapshot().activeAllocations, 1);

  recordGpuMemoryFree(0x5100);
  recordGpuMemoryFree(0x5200);
  recordGpuMemoryFree(0x5300);
}

TEST(GpuMemoryTrackerTest, NullScopeShadowsAndRestoresOuterBinding) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  {
    ScopedGpuMemoryContext outerScope{gpuMemoryTracker.get()};
    {
      ScopedGpuMemoryContext disabledScope{nullptr};
      recordGpuMemoryAllocation(
          GpuMemoryResourceType::kCommonTransportDispatchTable, 0x5450, 64, 64);
    }
    recordGpuMemoryAllocation(
        GpuMemoryResourceType::kCommonTransportDispatchTable, 0x5460, 32, 32);
  }

  EXPECT_EQ(gpuMemoryTracker->snapshot().total.currentBytes, 32);
  recordGpuMemoryFree(0x5460);
}

TEST(GpuMemoryTrackerTest, FreeUsesRegisteredOwnerAcrossThreads) {
  auto ownerTracker = std::make_shared<GpuMemoryTracker>(kContext);
  auto unrelatedTracker = std::make_shared<GpuMemoryTracker>(kContext);
  {
    ScopedGpuMemoryContext scope{ownerTracker.get()};
    recordGpuMemoryAllocation(
        GpuMemoryResourceType::kIbgdaPeerTransportArray, 0x5500, 256, 256);
  }

  std::thread freeThread{[unrelatedTracker] {
    ScopedGpuMemoryContext scope{unrelatedTracker.get()};
    recordGpuMemoryFree(0x5500);
  }};
  freeThread.join();

  const GpuMemoryUsage expectedOwnerUsage{
      .peakBytes = 256,
      .totalAllocatedBytes = 256,
      .totalFreedBytes = 256,
  };
  EXPECT_EQ(ownerTracker->snapshot().total, expectedOwnerUsage);
  EXPECT_EQ(unrelatedTracker->snapshot().total, GpuMemoryUsage{});
}

TEST(GpuMemoryTrackerTest, FailedFreeKeepsAllocationRegistered) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  {
    ScopedGpuMemoryContext scope{gpuMemoryTracker.get()};
    recordGpuMemoryAllocation(
        GpuMemoryResourceType::kIbSlotCounter, 0x5550, 64, 64);
  }

  const auto generation = beginGpuMemoryFree(0x5550);
  ASSERT_NE(generation, 0);
  finishGpuMemoryFree(
      0x5550, GpuMemoryBackingKind::kRuntimeAllocation, generation, false);

  EXPECT_EQ(gpuMemoryTracker->snapshot().total.currentBytes, 64);
  recordGpuMemoryFree(0x5550);
  EXPECT_EQ(gpuMemoryTracker->snapshot().total.currentBytes, 0);
}

TEST(GpuMemoryTrackerTest, OverlappingFreeSuccessIsAccountedOnce) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  {
    ScopedGpuMemoryContext scope{gpuMemoryTracker.get()};
    recordGpuMemoryAllocation(
        GpuMemoryResourceType::kIbSlotCounter, 0x5560, 64, 64);
  }

  const auto firstGeneration = beginGpuMemoryFree(0x5560);
  const auto secondGeneration = beginGpuMemoryFree(0x5560);
  ASSERT_NE(firstGeneration, 0);
  ASSERT_EQ(secondGeneration, firstGeneration);

  finishGpuMemoryFree(
      0x5560, GpuMemoryBackingKind::kRuntimeAllocation, secondGeneration, true);
  finishGpuMemoryFree(
      0x5560, GpuMemoryBackingKind::kRuntimeAllocation, firstGeneration, false);

  const auto snapshot = gpuMemoryTracker->snapshot();
  const GpuMemoryUsage expectedTotalUsage{
      .peakBytes = 64,
      .totalAllocatedBytes = 64,
      .totalFreedBytes = 64,
  };
  EXPECT_EQ(snapshot.total, expectedTotalUsage);
  EXPECT_EQ(snapshot.activeAllocations, 0);
}

TEST(GpuMemoryTrackerTest, FailedOverlappingFreeKeepsOtherOperationInProgress) {
  auto oldTracker = std::make_shared<GpuMemoryTracker>(kContext);
  auto newTracker = std::make_shared<GpuMemoryTracker>(kContext);
  {
    ScopedGpuMemoryContext scope{oldTracker.get()};
    recordGpuMemoryAllocation(
        GpuMemoryResourceType::kIbSlotCounter, 0x5570, 64, 64);
  }

  const auto firstGeneration = beginGpuMemoryFree(0x5570);
  const auto secondGeneration = beginGpuMemoryFree(0x5570);
  ASSERT_NE(firstGeneration, 0);
  ASSERT_EQ(secondGeneration, firstGeneration);
  finishGpuMemoryFree(
      0x5570,
      GpuMemoryBackingKind::kRuntimeAllocation,
      secondGeneration,
      false);

  {
    ScopedGpuMemoryContext scope{newTracker.get()};
    recordGpuMemoryAllocation(
        GpuMemoryResourceType::kIbgdaPeerTransportArray, 0x5570, 128, 128);
  }
  finishGpuMemoryFree(
      0x5570, GpuMemoryBackingKind::kRuntimeAllocation, firstGeneration, true);

  const GpuMemoryUsage expectedOldUsage{
      .peakBytes = 64,
      .totalAllocatedBytes = 64,
      .totalFreedBytes = 64,
  };
  const GpuMemoryUsage expectedNewUsage{
      .currentLogicalBytes = 128,
      .currentBytes = 128,
      .peakBytes = 128,
      .totalAllocatedBytes = 128,
  };
  EXPECT_EQ(oldTracker->snapshot().total, expectedOldUsage);
  EXPECT_EQ(oldTracker->snapshot().activeAllocations, 0);
  EXPECT_EQ(newTracker->snapshot().total, expectedNewUsage);
  EXPECT_EQ(newTracker->snapshot().activeAllocations, 1);
  EXPECT_EQ(newTracker->snapshot().duplicateAllocations, 0);
  recordGpuMemoryFree(0x5570);
}

TEST(GpuMemoryTrackerTest, ReusedBackingKeepsNewAllocationRegistered) {
  auto oldTracker = std::make_shared<GpuMemoryTracker>(kContext);
  auto newTracker = std::make_shared<GpuMemoryTracker>(kContext);
  {
    ScopedGpuMemoryContext scope{oldTracker.get()};
    recordGpuMemoryAllocation(
        GpuMemoryResourceType::kIbSlotCounter, 0x5580, 64, 64);
  }

  const auto oldGeneration = beginGpuMemoryFree(0x5580);
  ASSERT_NE(oldGeneration, 0);
  {
    ScopedGpuMemoryContext scope{newTracker.get()};
    recordGpuMemoryAllocation(
        GpuMemoryResourceType::kIbgdaPeerTransportArray, 0x5580, 128, 128);
  }
  finishGpuMemoryFree(
      0x5580, GpuMemoryBackingKind::kRuntimeAllocation, oldGeneration, true);

  const GpuMemoryUsage expectedOldUsage{
      .peakBytes = 64,
      .totalAllocatedBytes = 64,
      .totalFreedBytes = 64,
  };
  const GpuMemoryUsage expectedNewUsage{
      .currentLogicalBytes = 128,
      .currentBytes = 128,
      .peakBytes = 128,
      .totalAllocatedBytes = 128,
  };
  EXPECT_EQ(oldTracker->snapshot().total, expectedOldUsage);
  EXPECT_EQ(newTracker->snapshot().total, expectedNewUsage);
  EXPECT_EQ(newTracker->snapshot().duplicateAllocations, 0);
  recordGpuMemoryFree(0x5580);
}

TEST(GpuMemoryTrackerTest, ReusedBackingDoesNotInflatePeak) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  ScopedGpuMemoryContext scope{gpuMemoryTracker.get()};
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kIbSlotCounter, 0x5590, 64, 64);

  const auto oldGeneration = beginGpuMemoryFree(0x5590);
  ASSERT_NE(oldGeneration, 0);
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kIbgdaPeerTransportArray, 0x5590, 128, 128);
  finishGpuMemoryFree(
      0x5590, GpuMemoryBackingKind::kRuntimeAllocation, oldGeneration, true);

  const auto snapshot = gpuMemoryTracker->snapshot();
  const GpuMemoryUsage expectedTotalUsage{
      .currentLogicalBytes = 128,
      .currentBytes = 128,
      .peakBytes = 128,
      .totalAllocatedBytes = 192,
      .totalFreedBytes = 64,
  };
  EXPECT_EQ(snapshot.total, expectedTotalUsage);
  EXPECT_EQ(snapshot.activeAllocations, 1);
  recordGpuMemoryFree(0x5590);
}

TEST(GpuMemoryTrackerTest, CurrentTrackerIsThreadLocal) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  {
    ScopedGpuMemoryContext scope{gpuMemoryTracker.get()};
    std::thread allocationThread{[] {
      recordGpuMemoryAllocation(
          GpuMemoryResourceType::kIbSlotCounter, 0x5600, 64, 64);
    }};
    allocationThread.join();
  }

  EXPECT_EQ(gpuMemoryTracker->snapshot().activeAllocations, 0);
  EXPECT_EQ(gpuMemoryTracker->snapshot().total.totalAllocatedBytes, 0);
}

TEST(GpuMemoryTrackerTest, RejectsDuplicateBackingAcrossTrackers) {
  auto ownerTracker = std::make_shared<GpuMemoryTracker>(kContext);
  auto duplicateTracker = std::make_shared<GpuMemoryTracker>(kContext);
  ScopedGpuMemoryContext ownerScope{ownerTracker.get()};
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kCommonTransportDispatchTable, 0x5700, 64, 64);
  {
    ScopedGpuMemoryContext duplicateScope{duplicateTracker.get()};
    recordGpuMemoryAllocation(
        GpuMemoryResourceType::kIbgdaAtomicReturnSink, 0x5700, 128, 128);
    recordGpuMemoryFree(0x5700);
  }

  const GpuMemoryUsage expectedOwnerUsage{
      .peakBytes = 64,
      .totalAllocatedBytes = 64,
      .totalFreedBytes = 64,
  };
  EXPECT_EQ(ownerTracker->snapshot().total, expectedOwnerUsage);
  EXPECT_EQ(duplicateTracker->snapshot().total, GpuMemoryUsage{});
  EXPECT_EQ(duplicateTracker->snapshot().duplicateAllocations, 1);
}

TEST(GpuMemoryTrackerTest, AllocationRegistryDoesNotOwnTracker) {
  std::weak_ptr<GpuMemoryTracker> weakTracker;
  {
    auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
    weakTracker = gpuMemoryTracker;
    ScopedGpuMemoryContext scope{gpuMemoryTracker.get()};
    recordGpuMemoryAllocation(
        GpuMemoryResourceType::kCommonTransportDispatchTable, 0x5800, 64, 64);
  }
  EXPECT_TRUE(weakTracker.expired());

  auto replacementTracker = std::make_shared<GpuMemoryTracker>(kContext);
  ScopedGpuMemoryContext scope{replacementTracker.get()};
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kIbSlotCounter, 0x5800, 128, 128);
  EXPECT_EQ(replacementTracker->snapshot().total.currentBytes, 128);
  EXPECT_EQ(replacementTracker->snapshot().duplicateAllocations, 0);
  recordGpuMemoryFree(0x5800);
}

TEST(GpuMemoryTrackerTest, ContextCanBeUpdatedAcrossReconfigure) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  const GpuMemoryContext updated{
      .commHash = 0x5678,
      .rank = 3,
      .device = 2,
  };

  gpuMemoryTracker->updateContext(updated);

  EXPECT_EQ(gpuMemoryTracker->snapshot().context, updated);
}

TEST(GpuMemoryTrackerTest, FormatsAggregateAndResourceBreakdown) {
  auto gpuMemoryTracker = std::make_shared<GpuMemoryTracker>(kContext);
  ScopedGpuMemoryContext scope{gpuMemoryTracker.get()};
  recordGpuMemoryAllocation(
      GpuMemoryResourceType::kCommonTransportDispatchTable,
      0x6000,
      1024 * 1024,
      1024 * 1024);

  const auto report =
      formatGpuMemorySnapshot("final", gpuMemoryTracker->snapshot());
  EXPECT_NE(
      report.find("MCCL GPU memory [final] commHash=1234 rank=2 gpu=1"),
      std::string::npos);
  EXPECT_NE(report.find("common: 1.00 MiB (1048576 bytes)"), std::string::npos);
  EXPECT_NE(
      report.find("total accounted GPU memory: current=1.00 MiB"),
      std::string::npos);
  EXPECT_NE(
      report.find("resource common.transport_dispatch_table: current=1.00 MiB"),
      std::string::npos);
  recordGpuMemoryFree(0x6000);
}

TEST(GpuMemoryTrackerTest, ResourceMetadataIsStable) {
  EXPECT_EQ(
      gpuMemoryComponentName(GpuMemoryComponent::kUnclassified),
      "unclassified");
  EXPECT_EQ(gpuMemoryComponentName(GpuMemoryComponent::kIbgda), "ibgda");
  EXPECT_EQ(
      gpuMemoryResourceTypeName(GpuMemoryResourceType::kUnclassified),
      "unclassified");
  EXPECT_EQ(
      gpuMemoryResourceTypeComponent(GpuMemoryResourceType::kUnclassified),
      GpuMemoryComponent::kUnclassified);
  EXPECT_EQ(
      gpuMemoryResourceTypeComponent(
          GpuMemoryResourceType::kMcclAllGatherOverlapCounters),
      GpuMemoryComponent::kMccl);
  EXPECT_EQ(
      gpuMemoryResourceTypeComponent(
          GpuMemoryResourceType::kCommonTransportDispatchTable),
      GpuMemoryComponent::kCommon);
  EXPECT_EQ(
      gpuMemoryResourceTypeComponent(GpuMemoryResourceType::kNvlP2pSignal),
      GpuMemoryComponent::kNvl);
  EXPECT_EQ(
      gpuMemoryResourceTypeComponent(GpuMemoryResourceType::kIbEagerSendBuffer),
      GpuMemoryComponent::kIb);
  EXPECT_EQ(
      gpuMemoryResourceTypeName(
          GpuMemoryResourceType::kIbgdaPeerTransportArray),
      "ibgda.peer_transport_array");
  EXPECT_EQ(
      gpuMemoryResourceTypeComponent(
          GpuMemoryResourceType::kIbgdaPeerTransportArray),
      GpuMemoryComponent::kIbgda);
  for (std::size_t i = 0; i < kGpuMemoryResourceTypeCount; ++i) {
    const auto resource = static_cast<GpuMemoryResourceType>(i);
    EXPECT_FALSE(gpuMemoryResourceTypeName(resource).empty());
    EXPECT_NE(gpuMemoryResourceTypeName(resource), "unknown");
    EXPECT_NE(
        gpuMemoryResourceTypeComponent(resource), GpuMemoryComponent::kCount);
  }
}

} // namespace
} // namespace meta::comms::memtrace
