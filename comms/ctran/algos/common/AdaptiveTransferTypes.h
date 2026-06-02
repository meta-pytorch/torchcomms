// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/ctran/backends/ib/CtranIbBase.h"

namespace ctran {
namespace algos {

// Transfer mode for adaptive data transfer mechanism.
//
// The adaptive transfer mechanism checks RegCache to determine if a buffer
// is pre-registered:
// - ZERO_COPY: Buffer is registered, use direct RDMA/NVL transfer
// - COPY_BASED: Buffer not registered, use staging buffer with pipelining
enum class TransferMode {
  ZERO_COPY, // Buffer is registered, use direct RDMA/NVL transfer
  COPY_BASED, // Buffer not registered, use staging buffer
};

// Transfer direction for adaptive transfer.
enum class TransferDirection {
  SEND,
  RECV,
};

// Configuration for a single direction (send or recv) of adaptive transfer.
//
// This struct is populated by checkAndSetTransferMode() and used by
// collective implementations to determine the appropriate transfer path.
//
// Default is ZERO_COPY to maintain backward compatibility with existing
// code paths (NVL, TCPDM) that don't use adaptive transfer yet.
//
// Note: Staging buffer management is handled by each collective implementation,
// not by this struct. Collectives should use CtranAlgo::getTmpBufInfo() to
// get staging buffers when mode == COPY_BASED.
struct DirectionalTransferConfig {
  TransferMode mode{
      TransferMode::ZERO_COPY}; // Default to zero-copy for backward compat
  void* memHdl{nullptr}; // Memory registration handle (populated by
                         // checkAndSetTransferMode for zero-copy).
                         // Also used to carry remote buffer address after
                         // resolveRemoteInfo on the recv config.
  CtranIbRemoteAccessKey remoteKey{}; // Remote access key (ZC only, IB backend)
  int numBlocks{0}; // Kernel blocks for copy-based staging D2D

  // Returns true if zero-copy path should be used.
  bool isZeroCopy() const {
    return mode == TransferMode::ZERO_COPY;
  }
};

// Configuration for adaptive transfer with separate send and recv configs.
//
// This struct holds transfer configurations for both directions, allowing
// different transfer modes for send and receive operations.
struct TransferConfig {
  DirectionalTransferConfig send; // Configuration for send direction
  DirectionalTransferConfig recv; // Configuration for recv direction

  // Returns true if send zero-copy path should be used.
  bool isSendZeroCopy() const {
    return send.isZeroCopy();
  }

  // Returns true if recv zero-copy path should be used.
  bool isRecvZeroCopy() const {
    return recv.isZeroCopy();
  }
};

} // namespace algos
} // namespace ctran
