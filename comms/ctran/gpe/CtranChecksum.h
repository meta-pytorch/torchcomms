// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/utils/PinnedHostPool.h"
#include "comms/utils/cvars/nccl_cvars.h"

// Currently, we hard-code thread number to be 1024, because
// the kernel template is instantiated with 1024 threads.
static const int CHECKSUM_NUM_THREAD = 1024;

struct ChecksumItem {
  using Self = ChecksumItem;

  static const char* name() {
    return "Checksum";
  }

  void reset() {
    inUse_ = false;
    checksum_ = 0;
  }

  bool inUse() {
    return inUse_;
  }

  void onPop() {
    inUse_ = true;
  }

  bool inUse_{false};
  uint32_t checksum_{0};

  void _() {
    // Make sure ChecksumItem satisfies the PinnedHostItem concept
    static_assert(PinnedHostItem<Self>);

    // The following compile-time check is a hint of the memory usage
    // ChecksumItem uses 8 bytes because of alignment
    // Could reduce it to 4 bytes by making checksum 31 bits
    // if we want to save pinned memory
    static_assert(sizeof(Self) == 8);
  }
};

using ChecksumPool = PinnedHostPool<ChecksumItem>;

// Checksum Interface
struct ChecksumArgs {
  const void* buf{nullptr};
  uint32_t size{0};
  uint32_t* out{nullptr};
};

template <KernelConfig::KernelType T>
struct ChecksumHandler {
  static bool isSampled(int opCount);
  static std::optional<ChecksumArgs> ctranFillChecksumArgs(
      KernelConfig& kernelConfig,
      ChecksumItem* checksumItem,
      const CtranComm* comm);
};

template <>
struct ChecksumHandler<KernelConfig::KernelType::SEND> {
  static bool isSampled(int opCount) {
    return NCCL_CTRAN_SENDRECV_CHECKSUM_SAMPLE_RATE == 0
        ? false
        : opCount % NCCL_CTRAN_SENDRECV_CHECKSUM_SAMPLE_RATE == 0;
  }

  static std::optional<ChecksumArgs> ctranFillChecksumArgs(
      KernelConfig& kernelConfig,
      ChecksumItem* checksumItem,
      [[maybe_unused]] const CtranComm* comm) {
    if (checksumItem == nullptr) {
      return std::nullopt;
    }

    const auto& args = kernelConfig.args.collective.send;
    const auto buf = args.sendbuff;
    if (buf == nullptr) {
      return std::nullopt;
    }

    uint32_t size = args.count * commTypeSize(args.datatype);

    return ChecksumArgs{
        .buf = buf, .size = size, .out = &checksumItem->checksum_};
  }
};

template <>
struct ChecksumHandler<KernelConfig::KernelType::RECV> {
  static bool isSampled(int opCount) {
    return NCCL_CTRAN_SENDRECV_CHECKSUM_SAMPLE_RATE == 0
        ? false
        : opCount % NCCL_CTRAN_SENDRECV_CHECKSUM_SAMPLE_RATE == 0;
  }

  static std::optional<ChecksumArgs> ctranFillChecksumArgs(
      KernelConfig& kernelConfig,
      ChecksumItem* checksumItem,
      [[maybe_unused]] const CtranComm* comm) {
    if (checksumItem == nullptr) {
      return std::nullopt;
    }

    const auto& args = kernelConfig.args.collective.recv;
    const auto buf = args.recvbuff;
    if (buf == nullptr) {
      return std::nullopt;
    }

    uint32_t size = args.count * commTypeSize(args.datatype);

    return ChecksumArgs{
        .buf = buf, .size = size, .out = &checksumItem->checksum_};
  }
};

template <>
struct ChecksumHandler<KernelConfig::KernelType::ALLGATHER> {
  static bool isSampled(int opCount) {
    return NCCL_CTRAN_ALLGATHER_CHECKSUM_SAMPLE_RATE == 0
        ? false
        : opCount % NCCL_CTRAN_ALLGATHER_CHECKSUM_SAMPLE_RATE == 0;
  }

  static std::optional<ChecksumArgs> ctranFillChecksumArgs(
      KernelConfig& kernelConfig,
      ChecksumItem* checksumItem,
      const CtranComm* comm) {
    if (checksumItem == nullptr) {
      return std::nullopt;
    }

    const auto& args = kernelConfig.args.collective.allgather;
    const auto recvbuf = args.recvbuff;
    if (recvbuf == nullptr) {
      return std::nullopt;
    }

    uint32_t msg = args.count * commTypeSize(args.datatype);
    uint32_t size = msg * comm->statex_->nRanks();

    return ChecksumArgs{
        .buf = recvbuf, .size = size, .out = &checksumItem->checksum_};
  }
};

template <int Threads>
extern __global__ void checksumKernel(
    const uint8_t* __restrict__ in,
    const uint32_t size,
    uint32_t* __restrict__ out);

static inline int getChecksumGrid(int buf_size) {
  int gridSize = (buf_size + NCCL_CTRAN_CHECKSUM_BYTES_PER_THREAD_BLOCK - 1) /
      NCCL_CTRAN_CHECKSUM_BYTES_PER_THREAD_BLOCK;
  return std::min(
      std::max(1, gridSize), NCCL_CTRAN_CHECKSUM_MAX_NUM_THREAD_BLOCKS);
}
