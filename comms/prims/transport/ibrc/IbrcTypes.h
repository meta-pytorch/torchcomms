// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace comms::prims {

inline constexpr std::size_t kIbrcCqPollBatch = 32;
inline constexpr uint32_t kIbrcDefaultCmdQueueDepth = 128;
inline constexpr uint64_t kIbrcInvalidReadySeq = ~uint64_t{0};

// Max work requests a single descriptor posts to its QP: an RDMA_WRITE for the
// data payload plus an ATOMIC_FETCH_AND_ADD for the signal.
inline constexpr uint32_t kIbrcMaxWrsPerDescriptor = 2;

// Sentinel error_queue value for transport-level errors that are not
// attributable to a specific command queue.
inline constexpr uint32_t kIbrcUnknownQueue = ~uint32_t{0};

// Cache-line size: aligns the IBRC POD types and separates the host-mapped
// command-queue producer/consumer indices to avoid false-sharing across the
// GPU<->CPU boundary.
inline constexpr std::size_t kIbrcCacheLineBytes = 64;

enum class IbrcOp : uint16_t {
  PUT = 0,
  SIGNAL = 1,
};

enum IbrcFlags : uint16_t {
  IBRC_HAS_SIGNAL = 1 << 0,
  IBRC_SIGNAL_ADD = 1 << 1,
  IBRC_HAS_COUNTER = 1 << 2,
};

struct alignas(kIbrcCacheLineBytes) IbrcDesc {
  uint64_t local_addr;
  uint64_t remote_addr;
  uint64_t bytes;

  uint64_t signal_addr;
  uint64_t signal_value;

  uint64_t counter_addr;
  uint64_t counter_value;

  uint32_t lkey_device_order;
  uint32_t rkey_device_order;
  uint32_t signal_rkey_device_order;

  uint16_t op;
  uint16_t flags;

  uint8_t padding[48];

  uint64_t ready_seq;
};

static_assert(sizeof(IbrcDesc) == 128);
static_assert(alignof(IbrcDesc) == 64);
static_assert(offsetof(IbrcDesc, ready_seq) == 120);
static_assert(std::is_standard_layout_v<IbrcDesc>);
static_assert(std::is_trivially_copyable_v<IbrcDesc>);

struct alignas(kIbrcCacheLineBytes) IbrcNicStatus {
  uint32_t error;
  uint32_t error_queue;
  uint32_t error_code;
};

static_assert(sizeof(IbrcNicStatus) == 64);
static_assert(alignof(IbrcNicStatus) == 64);
static_assert(std::is_standard_layout_v<IbrcNicStatus>);
static_assert(std::is_trivially_copyable_v<IbrcNicStatus>);

struct IbrcCmdQueueDevice {
  IbrcDesc* descs;
  uint64_t* pi;
  uint64_t* ci;
  IbrcNicStatus* status;
  uint32_t depth;
  uint32_t mask;
};

static_assert(std::is_standard_layout_v<IbrcCmdQueueDevice>);
static_assert(std::is_trivially_copyable_v<IbrcCmdQueueDevice>);

struct IbrcBlockQpState {
  uint32_t put_rr{0};
};

static_assert(std::is_standard_layout_v<IbrcBlockQpState>);
static_assert(std::is_trivially_copyable_v<IbrcBlockQpState>);

} // namespace comms::prims
