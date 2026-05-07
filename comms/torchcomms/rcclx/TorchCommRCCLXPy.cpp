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
          py::call_guard<py::gil_scoped_release>());
}
