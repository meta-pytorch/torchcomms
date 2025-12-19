// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <cstdint>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::test {

using namespace comms::pipes;

// =============================================================================
// Basic Properties Tests
// =============================================================================

__global__ void testBasicPropertiesKernel(
    const uint32_t* data,
    uint32_t size,
    uint32_t* results,
    uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    DeviceSpan<const uint32_t> span(data, size);

    // Test size()
    if (span.size() != size) {
      atomicAdd(errorCount, 1);
    }
    results[0] = span.size();

    // Test empty()
    if (span.empty() != (size == 0)) {
      atomicAdd(errorCount, 1);
    }
    results[1] = span.empty() ? 1 : 0;

    // Test data()
    if (span.data() != data) {
      atomicAdd(errorCount, 1);
    }
  }
}

void testBasicProperties(
    const uint32_t* data_d,
    uint32_t size,
    uint32_t* results_d,
    uint32_t* errorCount_d) {
  testBasicPropertiesKernel<<<1, 1>>>(data_d, size, results_d, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Element Access Tests
// =============================================================================

__global__ void testElementAccessKernel(
    const uint32_t* data,
    uint32_t size,
    uint32_t* results,
    uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    DeviceSpan<const uint32_t> span(data, size);

    // Test operator[]
    bool indexAccessOk = true;
    for (uint32_t i = 0; i < size; i++) {
      if (span[i] != data[i]) {
        indexAccessOk = false;
        break;
      }
    }
    if (!indexAccessOk) {
      atomicAdd(errorCount, 1);
    }

    // Test front()
    if (size > 0 && span.front() != data[0]) {
      atomicAdd(errorCount, 1);
    }
    results[0] = size > 0 ? span.front() : 0;

    // Test back()
    if (size > 0 && span.back() != data[size - 1]) {
      atomicAdd(errorCount, 1);
    }
    results[1] = size > 0 ? span.back() : 0;
  }
}

void testElementAccess(
    const uint32_t* data_d,
    uint32_t size,
    uint32_t* results_d,
    uint32_t* errorCount_d) {
  testElementAccessKernel<<<1, 1>>>(data_d, size, results_d, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Iterator Tests
// =============================================================================

__global__ void testIteratorKernel(
    const uint32_t* data,
    uint32_t size,
    uint32_t* sum,
    uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    DeviceSpan<const uint32_t> span(data, size);

    // Test begin() and end()
    if (span.begin() != data) {
      atomicAdd(errorCount, 1);
    }
    if (span.end() != data + size) {
      atomicAdd(errorCount, 1);
    }

    // Test range-based for loop (uses begin/end)
    uint32_t total = 0;
    for (uint32_t val : span) {
      total += val;
    }
    *sum = total;

    // Verify sum manually
    uint32_t expectedSum = 0;
    for (uint32_t i = 0; i < size; i++) {
      expectedSum += data[i];
    }
    if (total != expectedSum) {
      atomicAdd(errorCount, 1);
    }
  }
}

void testIterator(
    const uint32_t* data_d,
    uint32_t size,
    uint32_t* sum_d,
    uint32_t* errorCount_d) {
  testIteratorKernel<<<1, 1>>>(data_d, size, sum_d, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Subspan Tests
// =============================================================================

__global__ void testSubspanKernel(
    const uint32_t* data,
    uint32_t size,
    uint32_t* results,
    uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    DeviceSpan<const uint32_t> span(data, size);

    // Test subspan(offset, count)
    if (size >= 6) {
      auto sub = span.subspan(2, 3); // Elements at indices 2, 3, 4
      if (sub.size() != 3) {
        atomicAdd(errorCount, 1);
      }
      if (sub.data() != data + 2) {
        atomicAdd(errorCount, 1);
      }
      if (sub[0] != data[2] || sub[1] != data[3] || sub[2] != data[4]) {
        atomicAdd(errorCount, 1);
      }
      results[0] = sub.size();
      results[1] = sub[0];
    }

    // Test subspan(offset) - from offset to end
    if (size >= 4) {
      auto sub = span.subspan(2); // Elements from index 2 to end
      if (sub.size() != size - 2) {
        atomicAdd(errorCount, 1);
      }
      if (sub.data() != data + 2) {
        atomicAdd(errorCount, 1);
      }
      results[2] = sub.size();
    }

    // Test first(count)
    if (size >= 3) {
      auto sub = span.first(3);
      if (sub.size() != 3) {
        atomicAdd(errorCount, 1);
      }
      if (sub.data() != data) {
        atomicAdd(errorCount, 1);
      }
      if (sub[0] != data[0] || sub[1] != data[1] || sub[2] != data[2]) {
        atomicAdd(errorCount, 1);
      }
      results[3] = sub.size();
    }

    // Test last(count)
    if (size >= 3) {
      auto sub = span.last(3);
      if (sub.size() != 3) {
        atomicAdd(errorCount, 1);
      }
      if (sub.data() != data + size - 3) {
        atomicAdd(errorCount, 1);
      }
      results[4] = sub.size();
      results[5] = sub[0]; // Should be data[size-3]
    }
  }
}

void testSubspan(
    const uint32_t* data_d,
    uint32_t size,
    uint32_t* results_d,
    uint32_t* errorCount_d) {
  testSubspanKernel<<<1, 1>>>(data_d, size, results_d, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Factory Function Tests
// =============================================================================

__global__ void testMakeDeviceSpanKernel(
    const uint32_t* data,
    uint32_t size,
    uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto span = make_device_span(data, size);

    if (span.data() != data) {
      atomicAdd(errorCount, 1);
    }
    if (span.size() != size) {
      atomicAdd(errorCount, 1);
    }

    // Verify element access works
    for (uint32_t i = 0; i < size; i++) {
      if (span[i] != data[i]) {
        atomicAdd(errorCount, 1);
        break;
      }
    }
  }
}

void testMakeDeviceSpan(
    const uint32_t* data_d,
    uint32_t size,
    uint32_t* errorCount_d) {
  testMakeDeviceSpanKernel<<<1, 1>>>(data_d, size, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Const Conversion Tests
// =============================================================================

__global__ void
testConstConversionKernel(uint32_t* data, uint32_t size, uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Create mutable span
    DeviceSpan<uint32_t> mutableSpan(data, size);

    // Convert to const span (implicit conversion)
    DeviceSpan<const uint32_t> constSpan = mutableSpan;

    // Verify they point to the same data
    if (constSpan.data() != mutableSpan.data()) {
      atomicAdd(errorCount, 1);
    }
    if (constSpan.size() != mutableSpan.size()) {
      atomicAdd(errorCount, 1);
    }

    // Verify element access
    for (uint32_t i = 0; i < size; i++) {
      if (constSpan[i] != mutableSpan[i]) {
        atomicAdd(errorCount, 1);
        break;
      }
    }
  }
}

void testConstConversion(
    uint32_t* data_d,
    uint32_t size,
    uint32_t* errorCount_d) {
  testConstConversionKernel<<<1, 1>>>(data_d, size, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Empty Span Tests
// =============================================================================

__global__ void testEmptySpanKernel(uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Default constructor
    DeviceSpan<uint32_t> emptySpan;

    if (!emptySpan.empty()) {
      atomicAdd(errorCount, 1);
    }
    if (emptySpan.size() != 0) {
      atomicAdd(errorCount, 1);
    }
    if (emptySpan.data() != nullptr) {
      atomicAdd(errorCount, 1);
    }
    if (emptySpan.begin() != emptySpan.end()) {
      atomicAdd(errorCount, 1);
    }

    // Constructor with nullptr and 0
    DeviceSpan<uint32_t> emptySpan2(nullptr, 0);
    if (!emptySpan2.empty()) {
      atomicAdd(errorCount, 1);
    }
  }
}

void testEmptySpan(uint32_t* errorCount_d) {
  testEmptySpanKernel<<<1, 1>>>(errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Mutable Span Write Tests
// =============================================================================

__global__ void testMutableSpanWriteKernel(
    uint32_t* data,
    uint32_t size,
    uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    DeviceSpan<uint32_t> span(data, size);

    // Write through span
    for (uint32_t i = 0; i < size; i++) {
      span[i] = i * 10;
    }

    // Verify writes went to underlying data
    for (uint32_t i = 0; i < size; i++) {
      if (data[i] != i * 10) {
        atomicAdd(errorCount, 1);
        break;
      }
    }
  }
}

void testMutableSpanWrite(
    uint32_t* data_d,
    uint32_t size,
    uint32_t* errorCount_d) {
  testMutableSpanWriteKernel<<<1, 1>>>(data_d, size, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
