// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/types.h>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/collectives/moe_ep/cpp/shared/Config.h"
#include "comms/prims/collectives/moe_ep/cpp/shared/EventHandle.h"

namespace comms::prims::moe_ep {

class IntranodeRuntime;

/**
 * Buffer — pybind-facing C++ class backing
 * `comms.prims.collectives.moe_ep.moe_ep.Buffer`.
 *
 * Phase 1 (D3): intranode dispatch / combine over NVLink. Phase 2 (D4)
 * adds low_latency_*; Phase 3 (D5) adds internode_*.
 *
 * The Python `Buffer.__init__` constructs this with cosmetic args, then
 * gathers `device_ids` + `ipc_handles` Python-side and feeds them back
 * via `sync()`. The IPC handles are cosmetic on our backend — the
 * underlying `GpuMemHandler` (mode `kCudaIpcUncached`) does its own
 * IBootstrap-driven exchange in the runtime ctor.
 */
class Buffer {
 public:
  Buffer(
      int rank,
      int numRanks,
      std::int64_t numNvlBytes,
      std::int64_t numRdmaBytes,
      bool lowLatencyMode,
      bool explicitlyDestroy,
      bool enableShrink,
      bool useFabric);

  ~Buffer();

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&&) = delete;
  Buffer& operator=(Buffer&&) = delete;

  // ---- Topology accessors ----
  bool is_available() const noexcept {
    return available_;
  }
  int get_num_rdma_ranks() const noexcept;
  int get_rdma_rank() const noexcept;
  int get_root_rdma_rank(bool global) const noexcept;
  int get_local_device_id() const noexcept {
    return deviceId_;
  }

  // ---- IPC / NVSHMEM bootstrap ----
  pybind11::bytearray get_local_ipc_handle() const;
  pybind11::bytearray get_local_nvshmem_unique_id() const;
  /**
   * Finalize bootstrap. Python supplies the gathered `device_ids` +
   * `ipc_handles` from `dist.all_gather_object` calls, plus an optional
   * `root_unique_id` for RDMA. On our backend the IPC exchange is owned by
   * `GpuMemHandler` (kCudaIpcUncached) inside the runtime ctor, so the
   * `ipc_handles` arg is cosmetic.
   */
  void sync(
      const std::vector<int>& deviceIds,
      const std::vector<std::optional<pybind11::bytearray>>& ipcHandles,
      const std::optional<pybind11::bytearray>& rootUniqueId);

  void destroy();

  // ---- Layout ----
  /**
   * `get_dispatch_layout` — pure-compute kernel returning per-rank /
   * per-expert / per-RDMA-rank token counts. See cpp/kernels/Layout.cu.
   *
   * Returns: (num_tokens_per_rank, num_tokens_per_rdma_rank,
   *           num_tokens_per_expert, is_token_in_rank, event)
   */
  std::tuple<
      torch::Tensor,
      std::optional<torch::Tensor>,
      torch::Tensor,
      torch::Tensor,
      std::optional<EventHandle>>
  get_dispatch_layout(
      const torch::Tensor& topkIdx,
      int numExperts,
      const std::optional<EventHandle>& previousEvent,
      bool asyncFinish,
      bool allocateOnCommStream);

  // ---- Intranode dispatch / combine (Phase 1) ----
  /**
   * Returns a 6-tuple:
   *   (recv_x, recv_topk_idx, recv_topk_weights,
   * recv_num_tokens_per_expert_list, handle, event)
   */
  pybind11::tuple intranode_dispatch(
      const pybind11::object& x, // Tensor or (Tensor, Tensor) for FP8
      const pybind11::object& handle,
      const std::optional<torch::Tensor>& numTokensPerRank,
      const std::optional<torch::Tensor>& isTokenInRank,
      const std::optional<torch::Tensor>& numTokensPerExpert,
      const std::optional<torch::Tensor>& topkIdx,
      const std::optional<torch::Tensor>& topkWeights,
      int expertAlignment,
      int numWorstTokens,
      const Config& config,
      const std::optional<EventHandle>& previousEvent,
      bool asyncFinish,
      bool allocateOnCommStream);

  /**
   * Returns (combined_x, combined_topk_weights, event).
   */
  std::tuple<
      torch::Tensor,
      std::optional<torch::Tensor>,
      std::optional<EventHandle>>
  intranode_combine(
      const torch::Tensor& x,
      const std::optional<torch::Tensor>& topkWeights,
      const std::optional<torch::Tensor>& bias0,
      const std::optional<torch::Tensor>& bias1,
      const pybind11::object& handle,
      const Config& config,
      const std::optional<EventHandle>& previousEvent,
      bool asyncFinish,
      bool allocateOnCommStream);

  // ---- Internode (Phase 3 — D5) + Low-latency (Phase 2 — D4) ----
  // These are bound through the same pybind module (so the Python `Buffer`
  // class's full method surface stays addressable), but throw at runtime
  // until the matching kernels land. Concrete signatures + bindings live
  // in PyBindings.cpp.
  [[noreturn]] void notImplemented(const char* methodName) const;

 private:
  std::shared_ptr<meta::comms::IBootstrap> bootstrap_;
  const int rank_;
  const int numRanks_;
  const std::int64_t numNvlBytes_;
  const std::int64_t numRdmaBytes_;
  const bool lowLatencyMode_;
  const bool explicitlyDestroy_;
  const bool enableShrink_;
  const bool useFabric_;

  int deviceId_{-1};
  bool available_{false};
  bool destroyed_{false};

  // Local NVL data buffer + IPC handle. Allocated in the Buffer ctor (not
  // sync) so that `get_local_ipc_handle()` returns a real handle BEFORE
  // the Python `dist.all_gather_object(local_ipc_handle)` happens. Without
  // this, the gathered handles are all zeros and `cudaIpcOpenMemHandle`
  // fails at sync time.
  void* localIpcBuffer_{nullptr};
  mutable cudaIpcMemHandle_t localIpcHandle_{};
  mutable bool localIpcHandleReady_{false};

  // Peer-mapped device pointers opened by `cudaIpcOpenMemHandle` in `sync()`
  // from the IPC handles Python gathered via `dist.all_gather_object`. Entry
  // `rank_` is `localIpcBuffer_`; entry `i != rank_` is the local-process
  // pointer to peer `i`'s NVL data buffer. Closed in the destructor.
  std::vector<void*> peerIpcBuffers_;

  // Owned only after `sync()` for Phase 1 (intranode-only).
  std::unique_ptr<IntranodeRuntime> intranode_;
  // LowLatencyRuntime / InternodeRuntime land in D4 / D5.
};

} // namespace comms::prims::moe_ep
