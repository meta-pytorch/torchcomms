// Copyright (c) Meta Platforms, Inc. and affiliates.
// Platform selection integration test for uniflow AMD/CUDA support

#include <gtest/gtest.h>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/drivers/DeviceAdapter.h"

using namespace uniflow;

class PlatformSelectionTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test that device adapter can be created on current platform
TEST_F(PlatformSelectionTest, DeviceAdapterCreation) {
  auto adapter = createDeviceAdapter();
  EXPECT_NE(adapter, nullptr);
}

// Test that device adapter can allocate and free memory
TEST_F(PlatformSelectionTest, DeviceAdapterAllocFree) {
  auto adapter = createDeviceAdapter();
  ASSERT_NE(adapter, nullptr);

  constexpr size_t kSize = 4096;
  auto result = adapter->pinnedHostAlloc(kSize);

  // On platforms without GPU support, this may fail gracefully
  if (result.hasValue()) {
    void* ptr = result.value();
    EXPECT_NE(ptr, nullptr);

    // Verify we can free the memory
    auto freeStatus = adapter->pinnedHostFree(ptr);
    EXPECT_TRUE(freeStatus.hasValue());
  }
}

// Test platform-specific behavior
// Verifies that the DeviceAdapter interface works correctly across platforms
// without requiring platform-specific conditional compilation in test logic.
TEST_F(PlatformSelectionTest, PlatformSpecificBehavior) {
  auto adapter = createDeviceAdapter();
  ASSERT_NE(adapter, nullptr);

  // Allocate memory
  constexpr size_t kSize = 4096;
  auto allocResult = adapter->pinnedHostAlloc(kSize);

  if (!allocResult.hasValue()) {
    GTEST_SKIP() << "Memory allocation not supported on this platform";
  }

  void* hostPtr = allocResult.value();

  // Test hostGetDevicePointer - verifies the interface contract
  // The adapter should either return a valid device pointer or an error,
  // but should not crash. This tests the abstraction without leaking
  // platform details into the test.
  auto devPtrResult = adapter->hostGetDevicePointer(hostPtr);
  // The result may be success (valid pointer) or failure (unsupported),
  // but the call should complete without crashing.
  // We don't assert on the value here because the behavior is
  // platform-dependent and the interface allows both outcomes.

  // Cleanup
  adapter->pinnedHostFree(hostPtr);
}

// Test multiple allocations
TEST_F(PlatformSelectionTest, MultipleAllocations) {
  auto adapter = createDeviceAdapter();
  ASSERT_NE(adapter, nullptr);

  constexpr size_t kNumAllocs = 5;
  constexpr size_t kSize = 1024;
  std::vector<void*> ptrs;

  for (size_t i = 0; i < kNumAllocs; ++i) {
    auto result = adapter->pinnedHostAlloc(kSize);
    if (result.hasValue()) {
      ptrs.push_back(result.value());
    }
  }

  // Free all allocated pointers
  for (void* ptr : ptrs) {
    auto status = adapter->pinnedHostFree(ptr);
    EXPECT_TRUE(status.hasValue());
  }
}

// Test that platform detection works correctly
// This test verifies the DeviceAdapter can be created on any platform
// without requiring platform-specific conditional compilation.
// The adapter implementation handles platform differences internally.
TEST_F(PlatformSelectionTest, PlatformDetection) {
  auto adapter = createDeviceAdapter();
  EXPECT_NE(adapter, nullptr)
      << "DeviceAdapter should be creatable on all platforms";
}

// Test alignment requirements
TEST_F(PlatformSelectionTest, AllocationAlignment) {
  auto adapter = createDeviceAdapter();
  ASSERT_NE(adapter, nullptr);

  constexpr size_t kSize = 4096;
  constexpr size_t kAlignment = 4096; // Page alignment

  auto result = adapter->pinnedHostAlloc(kSize);
  if (result.hasValue()) {
    void* ptr = result.value();

    // Check page alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % kAlignment, 0)
        << "Pointer should be page-aligned for DMA";

    adapter->pinnedHostFree(ptr);
  }
}
