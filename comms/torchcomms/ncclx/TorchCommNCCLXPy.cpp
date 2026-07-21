// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/ncclx/NcclxGlobalApi.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"
#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"

namespace py = pybind11;
using namespace torch::comms;

template <typename T, typename... TOptions>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>, TOptions...>;

PYBIND11_MODULE(_comms_ncclx, m, py::mod_gil_not_used()) {
  m.doc() = "NCCLX specific python bindings for TorchComm";

  py::class_<TorchCommNCCLX, TorchCommBackend, std::shared_ptr<TorchCommNCCLX>>(
      m, "TorchCommNCCLX")
      .def(
          "device_alltoallv_single",
          [](TorchCommNCCLX& self,
             at::Tensor& output,
             const at::Tensor& input,
             const at::Tensor& output_split_sizes,
             const at::Tensor& input_split_sizes,
             bool async_op) {
            return self.device_alltoallv_single(
                output, input, output_split_sizes, input_split_sizes, async_op);
          },
          R"(
All-to-all-v operation where split sizes are device tensors (on GPU).

Unlike all_to_all_v_single where split sizes are host vectors, this API takes
split sizes as CUDA tensors, allowing the GPU to read them directly without
host-device synchronization. Displacements are computed internally by the
kernel as exclusive prefix sums of the counts.

Args:
    output: Output tensor to receive data.
    input: Input tensor containing data to send.
    output_split_sizes: CUDA int64 tensor of receive counts per rank.
    input_split_sizes: CUDA int64 tensor of send counts per rank.
    async_op: Whether to perform the operation asynchronously.

Returns:
    TorchWork object for tracking operation completion.
)",
          py::arg("output"),
          py::arg("input"),
          py::arg("output_split_sizes"),
          py::arg("input_split_sizes"),
          py::arg("async_op"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "comm_dump",
          &TorchCommNCCLX::comm_dump,
          R"(
Dump NCCL communicator internal state as key-value pairs.

Returns a dictionary of communicator metadata including comm hash,
rank, number of ranks, and collective trace information.

Returns:
    dict[str, str]: Key-value pairs of communicator state.
)",
          py::call_guard<py::gil_scoped_release>())
      .def("get_nccl_comm_ptr", &TorchCommNCCLX::getCommPtr)
#ifdef NCCL_REDUCE_SCATTER_QUANTIZE_SUPPORTED
      .def(
          "reduce_scatter_quantized",
          [](TorchCommNCCLX& self,
             at::Tensor& output,
             const at::Tensor& input,
             const ReduceOp& op,
             const at::Tensor& seed,
             bool async_op) {
            return self.reduce_scatter_quantized(
                output, input, op, seed, async_op);
          },
          R"(
Reduce-scatter with stochastic quantization (FP32 input/output, BF16 transport).

Performs a reduce-scatter where data is stochastically rounded from FP32 to BF16
for transport between peers, then reduced in FP32 at the destination. Uses the
PAT algorithm with Philox-based stochastic rounding to provide unbiased precision
reduction.

The input tensor must be FP32, of size (world_size * output.numel()).
The output tensor must be FP32.

Args:
    output: Output tensor (FP32). Will contain the reduced scatter result.
    input: Input tensor (FP32). Size must be output.numel() * world_size.
    op: Reduction operation. Only SUM and AVG are supported.
    seed: A single-element int64 CUDA tensor containing the Philox RNG seed
          for stochastic rounding. Must reside in GPU memory.
    async_op: Whether to perform the operation asynchronously.

Returns:
    TorchWork object for tracking operation completion.
)",
          py::arg("output"),
          py::arg("input"),
          py::arg("op"),
          py::arg("seed"),
          py::arg("async_op"),
          py::call_guard<py::gil_scoped_release>())
#endif
      .def(
          "set_config",
          &TorchCommNCCLX::setConfig,
          R"(
Override NCCL communicator configuration at runtime via key-value hints.

Applies the given hints to the communicator through ncclx::commSetConfig.
Only mutable hints (e.g., algorithm selection) are accepted; immutable hints
and unrecognized keys cause an exception.

Args:
    hints: Dict of key-value configuration hints.
        Supported mutable keys include algorithm overrides such as
        ``sendrecvAlgo``, ``allgatherAlgo``, ``allreduceAlgo``, etc.

Raises:
    RuntimeError: If any hint key is immutable, unrecognized, or the
        underlying ncclx::commSetConfig call fails.
)",
          py::arg("hints"),
          py::call_guard<py::gil_scoped_release>());

  m.def(
      "comm_dump_all",
      [](const std::unordered_map<std::string, std::string>& hints) {
        DefaultNcclxGlobalApi api;
        std::unordered_map<
            std::string,
            std::unordered_map<std::string, std::string>>
            map;
        auto result = api.commDumpAll(map, hints);
        if (result != ncclSuccess) {
          throw std::runtime_error(
              std::string("ncclCommDumpAll failed: ") +
              api.getErrorString(result));
        }
        return map;
      },
      R"(
Dump internal state of all NCCL communicators as nested key-value pairs.

This is a module-level function that does not require a communicator instance.
Returns a dictionary keyed by communicator hash, where each value is a
dictionary of that communicator's internal state.

Args:
    hints: Dict of key-value options controlling the dump behavior.
        Supported hints:
        - "comm_dump::requestFields": semicolon-separated list of output keys.
        - "comm_dump::flush": "1" to flush ring buffers before dumping.
        Empty dict (default) dumps all fields without flushing.

        comm_dump::requestFields are categorized by cost:

        Trivial — read directly from in-memory structs, no serialization:
            commHash, rank, localRank, node, nRanks, localRanks, nNodes,
            commDesc, CT_currentIteration, CT_currentIterationCommTimeUs,
            GlobalInfo::totalCommDurPerIterationUs, memory

        Expensive — requires dumping + JSON serialization of collections:
            CT_pastColls, CT_currentColls, CT_pendingColls,
            processGlobalErrors, GlobalInfo::NetworkPerfInfo

Returns:
    dict[str, dict[str, str]]: Nested key-value pairs of all communicator states.
)",
      py::arg("hints") = std::unordered_map<std::string, std::string>{},
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "init_caching_allocator_hook",
      []() {
        DefaultNcclxGlobalApi api;
        api.initCachingAllocatorHook();
      },
      R"(
Attach the CCA (CUDA Caching Allocator) memory hook for NCCLX.

This initializes the global memory registration hook that automatically
registers/deregisters GPU memory segments with the NCCLX transport layer
(ctran) as they are allocated/freed by PyTorch's CUDACachingAllocator.

This does not require creating a communicator. It is useful for P2P transfer
cases where memory needs to be registered for RDMA without a communicator.

The hook is a process-global singleton -- calling this multiple times is safe
(subsequent calls are no-ops).
)");

#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API
  // Device API methods (get_device_window, register_local_buffer,
  // deregister_local_buffer) are bound on the TorchCommWindow base class
  // in TorchCommPy.cpp and inherited by both GIN and Pipes subclasses.

  // --- GIN backend window class ---
  auto gin_cls = py::class_<
      TorchCommWindowNCCLXGin,
      TorchCommWindow,
      std::shared_ptr<TorchCommWindowNCCLXGin>>(m, "TorchCommWindowNCCLXGin");

  // GIN-specific methods (not on base class)
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
  gin_cls.def(
      "get_nvlink_address",
      [](TorchCommWindowNCCLXGin& self, int peer, int64_t offset) {
        void* ptr = self.get_nvlink_address(peer, static_cast<size_t>(offset));
        return reinterpret_cast<int64_t>(ptr);
      },
      R"(
Get the NVLink-mapped device pointer for a peer's window memory.

Returns the direct NVLink address that can be used to access the peer's
window buffer via NVLink. Returns 0 if the peer is not NVLink-accessible
(e.g., remote node over RDMA).

Prerequisites: Must call tensor_register() first.

Args:
    peer: The world rank of the peer whose address to retrieve.
    offset: Byte offset within the peer's window (default 0).

Returns:
    int: NVLink-mapped device pointer as int64, or 0 if not accessible.
)",
      py::arg("peer"),
      py::arg("offset") = 0,
      py::call_guard<py::gil_scoped_release>());

  gin_cls.def(
      "get_multimem_address",
      [](TorchCommWindowNCCLXGin& self, int64_t offset) {
        void* ptr = self.get_multimem_address(static_cast<size_t>(offset));
        return reinterpret_cast<int64_t>(ptr);
      },
      R"(
Get the NVLS multicast (multimem) device pointer for this window.

Returns the multicast address that can be used with multimem.ld_reduce
(hardware-fused all-reduce) and multimem.st (broadcast) PTX instructions
across all LSA-connected peers.

Requires sm_90+ (Hopper+) hardware with NVLS support.

Prerequisites: Must call tensor_register() first.

Args:
    offset: Byte offset within the window (default 0).

Returns:
    int: Multimem device pointer as int64, or 0 if not supported.
)",
      py::arg("offset") = 0,
      py::call_guard<py::gil_scoped_release>());
#endif

#if defined(ENABLE_PRIMS)
  // --- Pipes backend window class ---
  auto pipes_cls = py::class_<
      TorchCommWindowNCCLXPipes,
      TorchCommWindow,
      std::shared_ptr<TorchCommWindowNCCLXPipes>>(
      m, "TorchCommWindowNCCLXPipes");
#endif

#endif
}
