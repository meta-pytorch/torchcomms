// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "comms/prims/collectives/moe_ep/cpp/Buffer.h"
#include "comms/prims/collectives/moe_ep/cpp/shared/Config.h"
#include "comms/prims/collectives/moe_ep/cpp/shared/EventHandle.h"

namespace py = pybind11;
using namespace comms::prims::moe_ep;

PYBIND11_MODULE(_cpp, m) {
  m.doc() = "Python bindings for comms.prims.collectives.moe_ep MoE EP runtime";

  // ---------------------------------------------------------------------------
  // Module-level constants.
  // ---------------------------------------------------------------------------
  m.attr("NUM_WORKSPACE_BYTES") = py::int_(NUM_WORKSPACE_BYTES);
  m.attr("NUM_MAX_NVL_PEERS") = py::int_(NUM_MAX_NVL_PEERS);
  m.attr("NUM_BUFFER_ALIGNMENT_BYTES") = py::int_(NUM_BUFFER_ALIGNMENT_BYTES);
  // topk_idx_t == int64 — Python side uses this to assert dtype.
  m.attr("topk_idx_t") = py::module_::import("torch").attr("int64");

  // Module attrs / free fns expected by the test files.
  // - `AITER_MOE`: hardcoded `False` per design plan §2.4 (line 166).
  // - `is_sm90_compiled()`: returns False on AMD (matches USE_ROCM
  //   behavior); on NVIDIA we conservatively return False until a
  //   proper compute-cap probe lands.
  m.attr("AITER_MOE") = py::bool_(false);
  m.def("is_sm90_compiled", []() { return false; });

  // Low-latency RDMA buffer size estimator. Phase 2 (D4) lands the LL
  // kernels; this helper is bound now because the Python `Buffer` ctor
  // calls it during construction even when low_latency_mode=False (paths
  // that compute hint sizes for diagnostic output).
  //
  // Per-rank send + recv buffers sized by max-dispatch tokens × hidden bytes ×
  // num_experts / num_ranks × 2 (send + recv) + signal slots.
  m.def(
      "get_low_latency_rdma_size_hint",
      [](int num_max_dispatch_tokens_per_rank,
         int hidden,
         int num_ranks,
         int num_experts) {
        constexpr int kBytesPerToken = 2; // bf16
        const int num_local_experts = num_experts / num_ranks;
        const std::size_t per_token_bytes =
            static_cast<std::size_t>(hidden) * kBytesPerToken;
        // 2× for send + recv, num_ranks peers, num_local_experts slots.
        const std::size_t data =
            static_cast<std::size_t>(num_max_dispatch_tokens_per_rank) *
            per_token_bytes * num_ranks * num_local_experts * 2UL;
        // Plus a small per-(peer × local_expert) signal/state region.
        const std::size_t signal =
            static_cast<std::size_t>(num_ranks) * num_local_experts * 64UL;
        return data + signal + (32UL * 1024UL * 1024UL);
      },
      py::arg("num_max_dispatch_tokens_per_rank"),
      py::arg("hidden"),
      py::arg("num_ranks"),
      py::arg("num_experts"));

  // ---------------------------------------------------------------------------
  // Config — POD with five tunables.
  // ---------------------------------------------------------------------------
  py::class_<Config>(m, "Config")
      .def(
          py::init<int, int, int, int, int>(),
          py::arg("num_sms"),
          py::arg("num_max_nvl_chunked_send_tokens"),
          py::arg("num_max_nvl_chunked_recv_tokens"),
          py::arg("num_max_rdma_chunked_send_tokens") = 6,
          py::arg("num_max_rdma_chunked_recv_tokens") = 128)
      .def_readwrite("num_sms", &Config::num_sms)
      .def_readwrite(
          "num_max_nvl_chunked_send_tokens",
          &Config::num_max_nvl_chunked_send_tokens)
      .def_readwrite(
          "num_max_nvl_chunked_recv_tokens",
          &Config::num_max_nvl_chunked_recv_tokens)
      .def_readwrite(
          "num_max_rdma_chunked_send_tokens",
          &Config::num_max_rdma_chunked_send_tokens)
      .def_readwrite(
          "num_max_rdma_chunked_recv_tokens",
          &Config::num_max_rdma_chunked_recv_tokens)
      .def(
          "get_nvl_buffer_size_hint",
          &Config::getNvlBufferSizeHint,
          py::arg("hidden_bytes"),
          py::arg("num_ranks"))
      .def("get_rdma_buffer_size_hint", &Config::getRdmaBufferSizeHint);

  // ---------------------------------------------------------------------------
  // EventHandle — RAII shared_ptr cudaEvent_t, exposes current_stream_wait()
  // for the Python `EventOverlap` wrapper.
  // ---------------------------------------------------------------------------
  py::class_<EventHandle>(m, "EventHandle")
      .def(py::init<>())
      .def("current_stream_wait", &EventHandle::current_stream_wait);

  // ---------------------------------------------------------------------------
  // Buffer — pybind-facing class. Constructor + topology + IPC bootstrap +
  // layout/dispatch/combine. Phase 2/3 entry points are bound below but
  // throw `notImplemented` until D4 / D5 wire in their kernels.
  // ---------------------------------------------------------------------------
  py::class_<Buffer>(m, "Buffer")
      .def(
          py::init<
              int,
              int,
              std::int64_t,
              std::int64_t,
              bool,
              bool,
              bool,
              bool>(),
          py::arg("rank"),
          py::arg("num_ranks"),
          py::arg("num_nvl_bytes"),
          py::arg("num_rdma_bytes"),
          py::arg("low_latency_mode") = false,
          py::arg("explicitly_destroy") = false,
          py::arg("enable_shrink") = false,
          py::arg("use_fabric") = false)
      // ---- Topology ----
      .def("is_available", &Buffer::is_available)
      .def("get_num_rdma_ranks", &Buffer::get_num_rdma_ranks)
      .def("get_rdma_rank", &Buffer::get_rdma_rank)
      .def("get_root_rdma_rank", &Buffer::get_root_rdma_rank)
      .def("get_local_device_id", &Buffer::get_local_device_id)
      // ---- IPC / NVSHMEM bootstrap ----
      .def("get_local_ipc_handle", &Buffer::get_local_ipc_handle)
      .def("get_local_nvshmem_unique_id", &Buffer::get_local_nvshmem_unique_id)
      .def(
          "sync",
          &Buffer::sync,
          py::arg("device_ids"),
          py::arg("ipc_handles"),
          py::arg("root_unique_id"))
      .def("destroy", &Buffer::destroy)
      // ---- Layout ----
      .def(
          "get_dispatch_layout",
          &Buffer::get_dispatch_layout,
          py::arg("topk_idx"),
          py::arg("num_experts"),
          py::arg("previous_event") = std::nullopt,
          py::arg("async_finish") = false,
          py::arg("allocate_on_comm_stream") = false)
      // ---- Phase 1 (intranode dispatch / combine) ----
      .def(
          "intranode_dispatch",
          &Buffer::intranode_dispatch,
          py::arg("x"),
          py::arg("handle"),
          py::arg("num_tokens_per_rank") = std::nullopt,
          py::arg("is_token_in_rank") = std::nullopt,
          py::arg("num_tokens_per_expert") = std::nullopt,
          py::arg("topk_idx") = std::nullopt,
          py::arg("topk_weights") = std::nullopt,
          py::arg("expert_alignment") = 1,
          py::arg("num_worst_tokens") = 0,
          py::arg("config"),
          py::arg("previous_event") = std::nullopt,
          py::arg("async_finish") = false,
          py::arg("allocate_on_comm_stream") = false)
      .def(
          "intranode_combine",
          &Buffer::intranode_combine,
          py::arg("x"),
          py::arg("topk_weights") = std::nullopt,
          py::arg("bias_0") = std::nullopt,
          py::arg("bias_1") = std::nullopt,
          py::arg("handle"),
          py::arg("config"),
          py::arg("previous_event") = std::nullopt,
          py::arg("async_finish") = false,
          py::arg("allocate_on_comm_stream") = false);
}
