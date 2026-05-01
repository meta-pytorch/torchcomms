// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#include <nccl.h> // @manual=//comms/ncclx:nccl

#include "comms/ctran/Ctran.h"

namespace torch::comms {

/**
 * Abstract interface for the ctran execution path used by TorchCommNCCLX
 * for collectives that have no NCCL baseline (phase 1 of the direct-dispatch
 * design). Calling these collectives necessarily means running ctran, so
 * there is no algorithm choice or support gate at this layer — the API is
 * a thin pass-through that converts NCCL-typed arguments to ctran types
 * and translates `commResult_t` back to `ncclResult_t`.
 *
 * Phase-2 (NCCL-baseline) collectives that need an algorithm enum and a
 * runtime support check will be added to this interface in a subsequent
 * change.
 */
class CtranApi {
 public:
  virtual ~CtranApi() = default;

  /// Returns the per-comm `CtranComm*` hanging off `comm`, or `nullptr` if
  /// ctran isn't initialized for that comm (e.g. `NCCL_CTRAN_ENABLE` not
  /// set). Resolved via the per-NCCLX-version `getCtranCommFromNcclComm`
  /// accessor in `MetaFactory.cc`.
  [[nodiscard]] virtual CtranComm* getCtranComm(ncclComm_t comm) = 0;

  /// Returns true iff persistent allgather can run on this comm. Verifies
  /// ctran is initialized and every remote peer has an assigned ctran
  /// transport backend. Callers should check this before `allGatherPInit`
  /// and surface a clear error if false — otherwise `allGatherPInit` will
  /// fail later with a less informative message.
  [[nodiscard]] virtual bool allGatherPSupport(CtranComm* ctranComm) = 0;

  /// Persistent allgather init. `request` is an opaque handle that must be
  /// passed back to `allGatherPExec` and `allGatherPDestroy`.
  [[nodiscard]] virtual ncclResult_t allGatherPInit(
      void* recvbuff,
      size_t maxRecvCount,
      const std::unordered_map<std::string, std::string>& hints,
      ncclDataType_t datatype,
      CtranComm* ctranComm,
      cudaStream_t stream,
      void** request) = 0;

  [[nodiscard]] virtual ncclResult_t allGatherPExec(
      const void* sendbuff,
      size_t count,
      ncclDataType_t datatype,
      void* request) = 0;

  [[nodiscard]] virtual ncclResult_t allGatherPDestroy(void* request) = 0;
};

/**
 * Default implementation that calls the matching `ctran<Coll>` free
 * function directly. Stateless — a single instance can be shared across
 * TorchCommNCCLX comms.
 */
class DefaultCtranApi : public CtranApi {
 public:
  ~DefaultCtranApi() override = default;

  [[nodiscard]] CtranComm* getCtranComm(ncclComm_t comm) override;

  [[nodiscard]] bool allGatherPSupport(CtranComm* ctranComm) override;

  [[nodiscard]] ncclResult_t allGatherPInit(
      void* recvbuff,
      size_t maxRecvCount,
      const std::unordered_map<std::string, std::string>& hints,
      ncclDataType_t datatype,
      CtranComm* ctranComm,
      cudaStream_t stream,
      void** request) override;

  [[nodiscard]] ncclResult_t allGatherPExec(
      const void* sendbuff,
      size_t count,
      ncclDataType_t datatype,
      void* request) override;

  [[nodiscard]] ncclResult_t allGatherPDestroy(void* request) override;
};

} // namespace torch::comms
