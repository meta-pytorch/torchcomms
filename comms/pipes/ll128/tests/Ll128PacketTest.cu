// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstdint>

#include "comms/pipes/ll128/Ll128Packet.cuh"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::test {

using namespace comms::pipes;

// =============================================================================
// Alignment and Size Tests
// =============================================================================

__global__ void test_ll128_packet_alignment_kernel(uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Verify struct size
    if (sizeof(Ll128Packet) != 128) {
      atomicAdd(errorCount, 1);
    }

    // Verify alignment
    if (alignof(Ll128Packet) != 128) {
      atomicAdd(errorCount, 1);
    }

    // Verify an on-stack packet is aligned (note: __shared__ guarantees this)
    __shared__ Ll128Packet pkt;
    uintptr_t addr = reinterpret_cast<uintptr_t>(&pkt);
    if (addr % 128 != 0) {
      atomicAdd(errorCount, 1);
    }
  }
}

void test_ll128_packet_alignment(uint32_t* errorCount_d) {
  test_ll128_packet_alignment_kernel<<<1, 1>>>(errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Flag Read/Write Tests
// =============================================================================

__global__ void test_ll128_flag_read_write_kernel(
    Ll128Packet* pkt,
    uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Write flag = kLl128ReadyToWrite (-1)
    ll128_store_flag(*pkt, kLl128ReadyToWrite);
    int64_t flag = ll128_load_flag(*pkt);
    if (flag != kLl128ReadyToWrite) {
      printf(
          "Flag read/write mismatch: expected %lld, got %lld\n",
          (long long)kLl128ReadyToWrite,
          (long long)flag);
      atomicAdd(errorCount, 1);
    }

    // Write flag = 1 (step ID)
    ll128_store_flag(*pkt, 1);
    flag = ll128_load_flag(*pkt);
    if (flag != 1) {
      printf(
          "Flag read/write mismatch: expected 1, got %lld\n", (long long)flag);
      atomicAdd(errorCount, 1);
    }

    // Write flag = 42
    ll128_store_flag(*pkt, 42);
    flag = ll128_load_flag(*pkt);
    if (flag != 42) {
      printf(
          "Flag read/write mismatch: expected 42, got %lld\n", (long long)flag);
      atomicAdd(errorCount, 1);
    }
  }
}

void test_ll128_flag_read_write(void* packet_d, uint32_t* errorCount_d) {
  test_ll128_flag_read_write_kernel<<<1, 1>>>(
      static_cast<Ll128Packet*>(packet_d), errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Payload Size Calculation Tests
// =============================================================================

__global__ void test_ll128_packet_payload_size_kernel(uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // 0 bytes total → packet 0 has 0 payload
    if (ll128_packet_payload_size(0, 0) != 0) {
      atomicAdd(errorCount, 1);
    }

    // 1 byte total → packet 0 has 1 payload byte
    if (ll128_packet_payload_size(0, 1) != 1) {
      atomicAdd(errorCount, 1);
    }

    // 119 bytes total → packet 0 has 119 payload bytes
    if (ll128_packet_payload_size(0, 119) != 119) {
      atomicAdd(errorCount, 1);
    }

    // Exactly 120 bytes → packet 0 has 120 payload bytes
    if (ll128_packet_payload_size(0, 120) != 120) {
      atomicAdd(errorCount, 1);
    }

    // 121 bytes → packet 0 has 120, packet 1 has 1
    if (ll128_packet_payload_size(0, 121) != 120) {
      atomicAdd(errorCount, 1);
    }
    if (ll128_packet_payload_size(1, 121) != 1) {
      atomicAdd(errorCount, 1);
    }

    // 240 bytes → 2 full packets
    if (ll128_packet_payload_size(0, 240) != 120) {
      atomicAdd(errorCount, 1);
    }
    if (ll128_packet_payload_size(1, 240) != 120) {
      atomicAdd(errorCount, 1);
    }

    // 241 bytes → 2 full + 1 partial
    if (ll128_packet_payload_size(2, 241) != 1) {
      atomicAdd(errorCount, 1);
    }

    // Out-of-range packet → 0
    if (ll128_packet_payload_size(1, 120) != 0) {
      atomicAdd(errorCount, 1);
    }
    if (ll128_packet_payload_size(10, 120) != 0) {
      atomicAdd(errorCount, 1);
    }
  }
}

void test_ll128_packet_payload_size(uint32_t* errorCount_d) {
  test_ll128_packet_payload_size_kernel<<<1, 1>>>(errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Num Packets Tests
// =============================================================================

__global__ void test_ll128_num_packets_kernel(uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (ll128_num_packets(0) != 0) {
      atomicAdd(errorCount, 1);
    }
    if (ll128_num_packets(1) != 1) {
      atomicAdd(errorCount, 1);
    }
    if (ll128_num_packets(119) != 1) {
      atomicAdd(errorCount, 1);
    }
    if (ll128_num_packets(120) != 1) {
      atomicAdd(errorCount, 1);
    }
    if (ll128_num_packets(121) != 2) {
      atomicAdd(errorCount, 1);
    }
    if (ll128_num_packets(240) != 2) {
      atomicAdd(errorCount, 1);
    }
    if (ll128_num_packets(241) != 3) {
      atomicAdd(errorCount, 1);
    }
    // 64KB
    if (ll128_num_packets(65536) != 547) {
      atomicAdd(errorCount, 1);
    }
  }
}

void test_ll128_num_packets(uint32_t* errorCount_d) {
  test_ll128_num_packets_kernel<<<1, 1>>>(errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Slot Pointer Tests
// =============================================================================

__global__ void test_ll128_slot_ptr_kernel(uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __shared__ Ll128Packet pkt;

    auto* pkt_base = reinterpret_cast<volatile uint64_t*>(&pkt);

    // Each lane's slot pointer should point to data[lane] = pkt_base + lane*2
    for (int lane = 0; lane < 8; ++lane) {
      volatile uint64_t* slot = ll128_slot_ptr(pkt, lane);
      volatile uint64_t* expected = pkt_base + lane * 2;
      if (slot != expected) {
        printf(
            "Slot ptr mismatch for lane %d: expected %p, got %p\n",
            lane,
            (void*)expected,
            (void*)slot);
        atomicAdd(errorCount, 1);
      }
    }
  }
}

void test_ll128_slot_ptr(uint32_t* errorCount_d) {
  test_ll128_slot_ptr_kernel<<<1, 1>>>(errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Flag Initialization Test (cudaMemset 0xFF → kLl128ReadyToWrite)
// =============================================================================

__global__ void test_ll128_flag_init_kernel(
    Ll128Packet* pkt,
    uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int64_t flag = ll128_load_flag(*pkt);
    if (flag != kLl128ReadyToWrite) {
      printf(
          "Flag init mismatch: expected %lld (-1), got %lld\n",
          (long long)kLl128ReadyToWrite,
          (long long)flag);
      atomicAdd(errorCount, 1);
    }
  }
}

void test_ll128_flag_init(void* packet_d, uint32_t* errorCount_d) {
  test_ll128_flag_init_kernel<<<1, 1>>>(
      static_cast<Ll128Packet*>(packet_d), errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// can_use_ll128 Tests
// =============================================================================

__global__ void test_can_use_ll128_kernel(
    const char* aligned_ptr,
    uint32_t* errorCount) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // nbytes == 0 always eligible
    if (!can_use_ll128(nullptr, 0))
      atomicAdd(errorCount, 1);
    if (!can_use_ll128(aligned_ptr + 1, 0))
      atomicAdd(errorCount, 1);

    // Aligned + multiple of 16
    if (!can_use_ll128(aligned_ptr, 16))
      atomicAdd(errorCount, 1);
    if (!can_use_ll128(aligned_ptr, 32))
      atomicAdd(errorCount, 1);

    // Aligned + NOT multiple of 16
    if (can_use_ll128(aligned_ptr, 1))
      atomicAdd(errorCount, 1);
    if (can_use_ll128(aligned_ptr, 15))
      atomicAdd(errorCount, 1);
    if (can_use_ll128(aligned_ptr, 17))
      atomicAdd(errorCount, 1);

    // Misaligned (ptr+1) + multiple of 16
    if (can_use_ll128(aligned_ptr + 1, 16))
      atomicAdd(errorCount, 1);

    // Misaligned + not multiple of 16
    if (can_use_ll128(aligned_ptr + 1, 17))
      atomicAdd(errorCount, 1);
  }
}

void test_can_use_ll128(const char* aligned_ptr_d, uint32_t* errorCount_d) {
  test_can_use_ll128_kernel<<<1, 1>>>(aligned_ptr_d, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
