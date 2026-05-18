// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/common/AdaptiveTransferTypes.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/utils/commSpecs.h"

namespace ctran {
namespace algos {

// Check buffer registration status and determine transfer mode.
//
// This function checks RegCache to determine if a buffer is pre-registered:
// - If registered: Sets mode to ZERO_COPY and populates memHdl
// - If not registered: Sets mode to COPY_BASED and triggers async registration
//   for future calls
//
// @param comm The CtranComm instance
// @param buf Buffer to check
// @param len Length of buffer in bytes
// @param backend Backend type (IB, NVL, etc.)
// @param direction Transfer direction (SEND or RECV)
// @param config Output: TransferConfig populated with transfer mode for the
//               specified direction
// @return commSuccess on success, error code otherwise
inline commResult_t checkAndSetTransferMode(
    CtranComm* comm,
    const void* buf,
    size_t len,
    CtranMapperBackend backend,
    TransferDirection direction,
    TransferConfig* config) {
  // Get the directional config based on direction
  DirectionalTransferConfig* dirConfig =
      (direction == TransferDirection::SEND) ? &config->send : &config->recv;

  // Only IB backend is supported in this phase
  if (backend == CtranMapperBackend::IB) {
    auto regCache = ctran::RegCache::getInstance();
    if (!regCache) {
      CLOGF(ERR, "CTRAN-ADAPTIVE: Failed to get RegCache instance");
      return commInternalError;
    }

    // Step 1: Check if buffer is already registered (read-only lookup)
    void* regHdl = regCache->getRegHandle(buf, len);

    if (regHdl != nullptr) {
      // Buffer is registered - use zero-copy path
      // Store the handle directly to avoid lookup in collective
      dirConfig->mode = TransferMode::ZERO_COPY;
      dirConfig->memHdl = regHdl;
      CLOGF_TRACE(
          ALLOC,
          "CTRAN-ADAPTIVE: Buffer {} len {} is registered, using zero-copy (direction={})",
          buf,
          len,
          direction == TransferDirection::SEND ? "SEND" : "RECV");
      return commSuccess;
    }

    // Step 2: Buffer not registered - use copy-based path
    dirConfig->mode = TransferMode::COPY_BASED;
    dirConfig->memHdl = nullptr;

    // Step 3: Trigger async registration for future calls
    FB_COMMCHECK(comm->ctran_->mapper->regAsync(buf, len));

    CLOGF_TRACE(
        ALLOC,
        "CTRAN-ADAPTIVE: Buffer {} len {} not registered, using copy-based, "
        "triggered async registration (direction={})",
        buf,
        len,
        direction == TransferDirection::SEND ? "SEND" : "RECV");

    return commSuccess;
  }

  // NVL and other backends: Not implemented yet, default to zero-copy
  dirConfig->mode = TransferMode::ZERO_COPY;
  dirConfig->memHdl = nullptr;
  return commSuccess;
}

} // namespace algos
} // namespace ctran
