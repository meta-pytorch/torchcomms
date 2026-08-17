// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/rcclx/TorchCommRCCLX.hpp"

namespace py = pybind11;
using namespace torch::comms;

PYBIND11_MODULE(_comms_rcclx, m, py::mod_gil_not_used()) {
  m.doc() = "RCCLX specific python bindings for TorchComm";

  py::class_<TorchCommRCCLX, TorchCommBackend, std::shared_ptr<TorchCommRCCLX>>(
      m, "TorchCommRCCLX")
      .def(
          "sharded_relay_multi_group_all_reduce",
          [](TorchCommRCCLX& self,
             std::vector<at::Tensor>& tensors,
             const ReduceOp& op,
             const std::vector<std::vector<int64_t>>& all_active_ranks,
             const std::vector<int64_t>& per_group_counts,
             bool async_op) {
            return self.sharded_relay_multi_group_all_reduce(
                tensors, op, all_active_ranks, per_group_counts, async_op);
          },
          R"(
Fused multi-group sharded relay allreduce for 2D sparse parallelism.

This executes multiple allreduce groups in lockstep phases to eliminate
XGMI link contention on MI300x GPUs. All groups proceed through phases
simultaneously:
  - Phase 1: All groups scatter (active -> helpers)
  - Phase 1.5: All helpers accumulate received contributions
  - Phase 2: All groups gather (helpers -> active) + direct exchange
  - Phase 3: All active ranks perform final reduction

This eliminates the bidirectional traffic that occurs when different groups
are in different phases, achieving maximum XGMI link utilization.

Args:
    tensors: List of tensors to allreduce (one per group, modified in-place)
    op: Reduction operation (e.g., ReduceOp.SUM)
    all_active_ranks: List of lists, where each inner list contains the
        active rank IDs for one sparse group. All groups must have the
        same number of active ranks.
    per_group_counts: List of element counts (one per group). This allows
        different groups to have different tensor sizes. Each tensor's
        numel() must match the corresponding count.
    async_op: If True, returns a TorchWork handle for async operation

Returns:
    TorchWork object for operation completion if async_op=True

Example:
    # 2D sparse parallelism with 4 groups on 8 GPUs (different sizes per group)
    tensors = [tensor0, tensor1, tensor2, tensor3]
    all_active_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]
    per_group_counts = [1000000, 2000000, 500000, 1500000]
    comm.sharded_relay_multi_group_all_reduce(
        tensors, ReduceOp.SUM, all_active_ranks, per_group_counts, async_op=True)
)",
          py::arg("tensors"),
          py::arg("op"),
          py::arg("all_active_ranks"),
          py::arg("per_group_counts"),
          py::arg("async_op") = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "sharded_relay_multi_group_reduce_scatter",
          [](TorchCommRCCLX& self,
             std::vector<at::Tensor>& input_tensors,
             std::vector<at::Tensor>& output_tensors,
             const ReduceOp& op,
             const std::vector<std::vector<int64_t>>& all_active_ranks,
             const std::vector<int64_t>& per_group_recv_counts,
             bool async_op) {
            return self.sharded_relay_multi_group_reduce_scatter(
                input_tensors,
                output_tensors,
                op,
                all_active_ranks,
                per_group_recv_counts,
                async_op);
          },
          R"(
Fused multi-group sharded relay reduce-scatter for 2D sparse parallelism.

Reduce-scatter analogue of sharded_relay_multi_group_all_reduce. Executes
multiple reduce-scatter groups in lockstep phases to eliminate XGMI link
contention on MI300x GPUs. Each group has a power-of-two number of active ranks
(2 or 4); the logical collective is a reduce-scatter among them, accelerated by
passthrough helpers that relay sharded chunks of a single output block.

For each active rank, the input holds nActiveRanks x per_group_recv_counts[g]
elements (block[i] is the slice destined for active index i) and the output
holds per_group_recv_counts[g] elements receiving the reduced block[myIndex].

Args:
    input_tensors: List of send tensors (one per group). For an active rank,
        holds nActiveRanks x per_group_recv_counts[g] elements. For a helper
        rank, an nActiveRanks-slot scratch tensor (nActiveRanks x chunkSize).
    output_tensors: List of receive tensors (one per group). For an active
        rank, holds per_group_recv_counts[g] elements. Pass the
        local-contribution block of the input tensor for in-place operation.
        For a helper rank, the same scratch tensor as the input.
    op: Reduction operation (ReduceOp.SUM or ReduceOp.AVG)
    all_active_ranks: List of lists, where each inner list contains the active
        rank IDs for one sparse group. All groups must have the same number of
        active ranks.
    per_group_recv_counts: List of OUTPUT element counts (one per group).
    async_op: If True, returns a TorchWork handle for async operation

Returns:
    TorchWork object for operation completion if async_op=True

Example:
    # 2D sparse parallelism with 4 groups on 8 GPUs
    input_tensors = [in0, in1, in2, in3]    # active: 2 x recvCount elements
    output_tensors = [out0, out1, out2, out3]  # active: recvCount elements
    all_active_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]
    per_group_recv_counts = [1000000, 2000000, 500000, 1500000]
    comm.sharded_relay_multi_group_reduce_scatter(
        input_tensors, output_tensors, ReduceOp.SUM, all_active_ranks,
        per_group_recv_counts, async_op=True)
)",
          py::arg("input_tensors"),
          py::arg("output_tensors"),
          py::arg("op"),
          py::arg("all_active_ranks"),
          py::arg("per_group_recv_counts"),
          py::arg("async_op") = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "sharded_relay_multi_group_all_to_all",
          [](TorchCommRCCLX& self,
             std::vector<at::Tensor>& input_tensors,
             std::vector<at::Tensor>& output_tensors,
             const std::vector<std::vector<int64_t>>& all_active_ranks,
             const std::vector<int64_t>& per_group_segment_counts,
             bool async_op) {
            return self.sharded_relay_multi_group_all_to_all(
                input_tensors,
                output_tensors,
                all_active_ranks,
                per_group_segment_counts,
                async_op);
          },
          R"(
Fused multi-group sharded relay all-to-all for 2D sparse parallelism.

All-to-all analogue of sharded_relay_multi_group_reduce_scatter. Executes
multiple all-to-all groups in lockstep phases to eliminate XGMI link contention
on MI300x GPUs. Each group has exactly 2 active ranks; the logical collective is
a 2-rank all-to-all between them, accelerated by passthrough helpers. There is
NO reduction (pure data movement) and NO reduction op.

For each active rank, the input holds nActiveRanks x per_group_segment_counts[g]
elements (input = [sendSeg[0]|sendSeg[1]], sendSeg[j] destined for active index
j) and the output holds the same number of elements (output =
[recvSeg[0]|recvSeg[1]], recvSeg[i] from active index i).

OUT-OF-PLACE ONLY: input_tensors[g] and output_tensors[g] must be distinct for
the active group (matches native ncclAllToAll); passing aliasing buffers raises.

Args:
    input_tensors: List of send tensors (one per group). Active rank: holds
        nActiveRanks x per_group_segment_counts[g] elements. Helper rank: a
        two-slot scratch tensor.
    output_tensors: List of receive tensors (one per group). Active rank: holds
        nActiveRanks x per_group_segment_counts[g] elements (distinct from
        input). Helper rank: the same scratch tensor as the input.
    all_active_ranks: List of lists of active rank IDs per sparse group.
    per_group_segment_counts: List of per-segment element counts (one per group).
    async_op: If True, returns a TorchWork handle for async operation

Returns:
    TorchWork object for operation completion if async_op=True
)",
          py::arg("input_tensors"),
          py::arg("output_tensors"),
          py::arg("all_active_ranks"),
          py::arg("per_group_segment_counts"),
          py::arg("async_op") = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "sharded_relay_multi_group_all_gather",
          [](TorchCommRCCLX& self,
             std::vector<at::Tensor>& input_tensors,
             std::vector<at::Tensor>& output_tensors,
             const std::vector<std::vector<int64_t>>& all_active_ranks,
             const std::vector<int64_t>& per_group_send_counts,
             bool async_op) {
            return self.sharded_relay_multi_group_all_gather(
                input_tensors,
                output_tensors,
                all_active_ranks,
                per_group_send_counts,
                async_op);
          },
          R"(
Fused multi-group sharded relay all-gather for 2D sparse parallelism.

All-gather analogue of sharded_relay_multi_group_reduce_scatter (and the dual of
it). Executes multiple all-gather groups in lockstep phases to eliminate XGMI
link contention on MI300x GPUs. Each group has exactly 2 active ranks; the
logical collective is a 2-rank all-gather between them, accelerated by
passthrough helpers. There is NO reduction (pure data movement) and NO op.

For each active rank, the input holds per_group_send_counts[g] elements (its
contribution) and the output holds nActiveRanks x per_group_send_counts[g]
elements (output[i x sendCount] receives active index i's contribution).

Supports both in-place and out-of-place. In-place is detected when the active
input aliases output + myActiveIndex x sendCount (standard NCCL all-gather
in-place convention).

Args:
    input_tensors: List of send tensors (one per group). Active rank: holds
        per_group_send_counts[g] elements. Helper rank: a two-slot scratch
        tensor.
    output_tensors: List of receive tensors (one per group). Active rank: holds
        nActiveRanks x per_group_send_counts[g] elements. Helper rank: the same
        scratch tensor as the input.
    all_active_ranks: List of lists of active rank IDs per sparse group.
    per_group_send_counts: List of per-rank contribution counts (one per group).
    async_op: If True, returns a TorchWork handle for async operation

Returns:
    TorchWork object for operation completion if async_op=True
)",
          py::arg("input_tensors"),
          py::arg("output_tensors"),
          py::arg("all_active_ranks"),
          py::arg("per_group_send_counts"),
          py::arg("async_op") = false,
          py::call_guard<py::gil_scoped_release>());
}
