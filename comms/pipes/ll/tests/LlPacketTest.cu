// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstdint>

#include "comms/pipes/ll/LlPacket.cuh"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::test {

using namespace comms::pipes;

// =============================================================================
// Size Tests
// =============================================================================

__global__ void test_ll_line_size_kernel(uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (sizeof(LlLine) != 16) {
      printf(
          "LlLine size mismatch: expected 16, got %llu\n",
          (unsigned long long)sizeof(LlLine));
      atomicAdd(errorCount, 1);
    }
  }
}

void test_ll_line_size(uint32_t* errorCount_d) {
  test_ll_line_size_kernel<<<1, 1>>>(errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Flag Read/Write Tests
// =============================================================================

__global__ void test_ll_flag_read_write_kernel(
    LlLine* line,
    uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Write a line with flag_value = 42
    LlLine out;
    out.data1 = 0xAAAAAAAA;
    out.flag1 = 42;
    out.data2 = 0xBBBBBBBB;
    out.flag2 = 42;
    ll_store_line(line, out);

    // Read it back
    LlLine in;
    ll_load_line(line, in);

    if (in.data1 != 0xAAAAAAAA) {
      printf(
          "data1 mismatch: expected 0xAAAAAAAA, got 0x%X\n",
          (unsigned)in.data1);
      atomicAdd(errorCount, 1);
    }
    if (in.flag1 != 42) {
      printf("flag1 mismatch: expected 42, got %u\n", (unsigned)in.flag1);
      atomicAdd(errorCount, 1);
    }
    if (in.data2 != 0xBBBBBBBB) {
      printf(
          "data2 mismatch: expected 0xBBBBBBBB, got 0x%X\n",
          (unsigned)in.data2);
      atomicAdd(errorCount, 1);
    }
    if (in.flag2 != 42) {
      printf("flag2 mismatch: expected 42, got %u\n", (unsigned)in.flag2);
      atomicAdd(errorCount, 1);
    }

    // Test ACK
    ll_store_ack(line);
    ll_load_line(line, in);
    if (in.flag1 != kLlReadyToWrite || in.flag2 != kLlReadyToWrite) {
      printf(
          "ACK flag mismatch: expected 0x%X, got flag1=0x%X flag2=0x%X\n",
          (unsigned)kLlReadyToWrite,
          (unsigned)in.flag1,
          (unsigned)in.flag2);
      atomicAdd(errorCount, 1);
    }
    if (in.data1 != 0 || in.data2 != 0) {
      printf(
          "ACK data mismatch: expected 0, got data1=0x%X data2=0x%X\n",
          (unsigned)in.data1,
          (unsigned)in.data2);
      atomicAdd(errorCount, 1);
    }
  }
}

void test_ll_flag_read_write(void* line_d, uint32_t* errorCount_d) {
  test_ll_flag_read_write_kernel<<<1, 1>>>(
      static_cast<LlLine*>(line_d), errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Num Lines Tests
// =============================================================================

__global__ void test_ll_num_lines_kernel(uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (ll_num_lines(0) != 0)
      atomicAdd(errorCount, 1);
    if (ll_num_lines(1) != 1)
      atomicAdd(errorCount, 1);
    if (ll_num_lines(7) != 1)
      atomicAdd(errorCount, 1);
    if (ll_num_lines(8) != 1)
      atomicAdd(errorCount, 1);
    if (ll_num_lines(9) != 2)
      atomicAdd(errorCount, 1);
    if (ll_num_lines(16) != 2)
      atomicAdd(errorCount, 1);
    if (ll_num_lines(17) != 3)
      atomicAdd(errorCount, 1);
    // 64KB = 65536 bytes -> 65536/8 = 8192 lines
    if (ll_num_lines(65536) != 8192)
      atomicAdd(errorCount, 1);
  }
}

void test_ll_num_lines(uint32_t* errorCount_d) {
  test_ll_num_lines_kernel<<<1, 1>>>(errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Buffer Size Tests
// =============================================================================

__global__ void test_ll_buffer_size_kernel(uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (ll_buffer_size(0) != 0)
      atomicAdd(errorCount, 1);
    // 8 bytes -> 1 line -> 16 bytes
    if (ll_buffer_size(8) != 16)
      atomicAdd(errorCount, 1);
    // 16 bytes -> 2 lines -> 32 bytes
    if (ll_buffer_size(16) != 32)
      atomicAdd(errorCount, 1);
    // 64KB -> 8192 lines -> 131072 bytes
    if (ll_buffer_size(65536) != 131072)
      atomicAdd(errorCount, 1);
  }
}

void test_ll_buffer_size(uint32_t* errorCount_d) {
  test_ll_buffer_size_kernel<<<1, 1>>>(errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Buffer Payload Capacity Tests
// =============================================================================

__global__ void test_ll_buffer_payload_capacity_kernel(uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // 0 bytes buffer -> 0 capacity
    if (ll_buffer_payload_capacity(0) != 0)
      atomicAdd(errorCount, 1);
    // 16 bytes buffer -> 1 line -> 8 bytes payload
    if (ll_buffer_payload_capacity(16) != 8)
      atomicAdd(errorCount, 1);
    // 32 bytes buffer -> 2 lines -> 16 bytes payload
    if (ll_buffer_payload_capacity(32) != 16)
      atomicAdd(errorCount, 1);
    // 131072 bytes buffer -> 8192 lines -> 65536 bytes payload
    if (ll_buffer_payload_capacity(131072) != 65536)
      atomicAdd(errorCount, 1);
    // 15 bytes buffer (not a full line) -> 0 lines -> 0 payload
    if (ll_buffer_payload_capacity(15) != 0)
      atomicAdd(errorCount, 1);
  }
}

void test_ll_buffer_payload_capacity(uint32_t* errorCount_d) {
  test_ll_buffer_payload_capacity_kernel<<<1, 1>>>(errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Flag Initialization Test (cudaMemset 0xFF -> kLlReadyToWrite)
// =============================================================================

__global__ void test_ll_flag_init_kernel(LlLine* line, uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    LlLine in;
    ll_load_line(line, in);
    if (in.flag1 != kLlReadyToWrite) {
      printf(
          "Flag1 init mismatch: expected 0x%X, got 0x%X\n",
          (unsigned)kLlReadyToWrite,
          (unsigned)in.flag1);
      atomicAdd(errorCount, 1);
    }
    if (in.flag2 != kLlReadyToWrite) {
      printf(
          "Flag2 init mismatch: expected 0x%X, got 0x%X\n",
          (unsigned)kLlReadyToWrite,
          (unsigned)in.flag2);
      atomicAdd(errorCount, 1);
    }
  }
}

void test_ll_flag_init(void* line_d, uint32_t* errorCount_d) {
  test_ll_flag_init_kernel<<<1, 1>>>(
      static_cast<LlLine*>(line_d), errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// can_use_ll Tests
// =============================================================================

__global__ void test_can_use_ll_kernel(
    const char* aligned_ptr,
    uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // nbytes == 0 always eligible
    if (!can_use_ll(nullptr, 0))
      atomicAdd(errorCount, 1);
    if (!can_use_ll(aligned_ptr + 1, 0))
      atomicAdd(errorCount, 1);

    // Aligned + multiple of 8
    if (!can_use_ll(aligned_ptr, 8))
      atomicAdd(errorCount, 1);
    if (!can_use_ll(aligned_ptr, 16))
      atomicAdd(errorCount, 1);
    if (!can_use_ll(aligned_ptr, 64))
      atomicAdd(errorCount, 1);

    // Aligned + NOT multiple of 8
    if (can_use_ll(aligned_ptr, 1))
      atomicAdd(errorCount, 1);
    if (can_use_ll(aligned_ptr, 7))
      atomicAdd(errorCount, 1);
    if (can_use_ll(aligned_ptr, 9))
      atomicAdd(errorCount, 1);

    // Misaligned (ptr+1) + multiple of 8
    if (can_use_ll(aligned_ptr + 1, 8))
      atomicAdd(errorCount, 1);

    // Misaligned + not multiple of 8
    if (can_use_ll(aligned_ptr + 1, 9))
      atomicAdd(errorCount, 1);
  }
}

void test_can_use_ll(const char* aligned_ptr_d, uint32_t* errorCount_d) {
  test_can_use_ll_kernel<<<1, 1>>>(aligned_ptr_d, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// ll_flag Tests
// =============================================================================

__global__ void test_ll_flag_kernel(uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // ll_flag(0) = 1
    if (ll_flag(0) != 1)
      atomicAdd(errorCount, 1);
    // ll_flag(1) = 2
    if (ll_flag(1) != 2)
      atomicAdd(errorCount, 1);
    // ll_flag(41) = 42
    if (ll_flag(41) != 42)
      atomicAdd(errorCount, 1);
    // Flag values should never be 0 or kLlReadyToWrite
    if (ll_flag(0) == 0)
      atomicAdd(errorCount, 1);
    if (ll_flag(0) == kLlReadyToWrite)
      atomicAdd(errorCount, 1);
  }
}

void test_ll_flag(uint32_t* errorCount_d) {
  test_ll_flag_kernel<<<1, 1>>>(errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
