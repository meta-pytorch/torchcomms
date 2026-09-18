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
             bool async_op,
             std::optional<std::vector<at::Tensor>> output_tensors,
             bool low_precision) {
            return self.sharded_relay_multi_group_all_reduce(
                tensors,
                op,
                all_active_ranks,
                per_group_counts,
                async_op,
                output_tensors,
                low_precision);
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
    output_tensors: Optional list of output segment tensors (one list per
        group), parallel to `tensors`. When None (default) the allreduce is
        in-place. When provided, the active group's reduced result is written
        out-of-place into these output tensors while the inputs are preserved;
        the active group's output segments must mirror its input segments in
        count and per-segment numel().
    low_precision: If True, use the low-precision (fp8e4m3) wire format where
        it pays. An internal size-only gate decides -- unsupported dtype,
        counts that are not a multiple of 128, or a message below the measured
        crossover all decline to full precision SILENTLY, so assert engagement
        rather than assuming it. COLLECTIVE: pass the same value on every rank
        of the call, exactly like the dtype and the counts. Ranks that disagree
        disagree on how many bytes cross each link, so the call hangs or
        corrupts rather than degrading.

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
          py::arg("output_tensors") = py::none(),
          py::arg("low_precision") = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "sharded_relay_multi_group_reduce_scatter",
          [](TorchCommRCCLX& self,
             std::vector<at::Tensor>& input_tensors,
             std::vector<at::Tensor>& output_tensors,
             const ReduceOp& op,
             const std::vector<std::vector<int64_t>>& all_active_ranks,
             const std::vector<int64_t>& per_group_recv_counts,
             bool async_op,
             bool low_precision) {
            return self.sharded_relay_multi_group_reduce_scatter(
                input_tensors,
                output_tensors,
                op,
                all_active_ranks,
                per_group_recv_counts,
                async_op,
                low_precision);
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
    low_precision: If True, use the fp8e4m3 wire format where it pays. See
        sharded_relay_multi_group_all_reduce -- same gate, same COLLECTIVE
        contract.

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
          py::arg("low_precision") = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "sharded_relay_multi_group_all_to_all",
          [](TorchCommRCCLX& self,
             std::vector<at::Tensor>& input_tensors,
             std::vector<at::Tensor>& output_tensors,
             const std::vector<std::vector<int64_t>>& all_active_ranks,
             const std::vector<int64_t>& per_group_segment_counts,
             bool async_op,
             bool low_precision) {
            return self.sharded_relay_multi_group_all_to_all(
                input_tensors,
                output_tensors,
                all_active_ranks,
                per_group_segment_counts,
                async_op,
                low_precision);
          },
          R"(
Fused multi-group sharded relay all-to-all for 2D sparse parallelism.

All-to-all analogue of sharded_relay_multi_group_reduce_scatter. Executes
multiple all-to-all groups in lockstep phases to eliminate XGMI link contention
on MI300x GPUs. Each group has a power-of-two number of active ranks (2 or 4);
the logical collective is an all-to-all among them, accelerated by passthrough
helpers (A==2 uses the original path; A>2 uses a flat all-to-all + 2-hop relay).
There is NO reduction (pure data movement) and NO reduction op.

For each active rank, the input holds nActiveRanks x per_group_segment_counts[g]
elements (input = [sendSeg[0]|...|sendSeg[A-1]], sendSeg[j] destined for active
index j) and the output holds the same number of elements (output =
[recvSeg[0]|...|recvSeg[A-1]], recvSeg[i] from active index i).

OUT-OF-PLACE ONLY: input_tensors[g] and output_tensors[g] must be distinct for
the active group (matches native ncclAllToAll); passing aliasing buffers raises.

Args:
    input_tensors: List of send tensors (one per group). Active rank: holds
        nActiveRanks x per_group_segment_counts[g] elements. Helper rank: an
        nActiveRanks-slot scratch tensor.
    output_tensors: List of receive tensors (one per group). Active rank: holds
        nActiveRanks x per_group_segment_counts[g] elements (distinct from
        input). Helper rank: the same scratch tensor as the input.
    all_active_ranks: List of lists of active rank IDs per sparse group.
    per_group_segment_counts: List of per-segment element counts (one per group).
    async_op: If True, returns a TorchWork handle for async operation
    low_precision: If True, use the fp8e4m3 wire format where it pays. See
        sharded_relay_multi_group_all_reduce -- same gate, same COLLECTIVE
        contract.

Returns:
    TorchWork object for operation completion if async_op=True
)",
          py::arg("input_tensors"),
          py::arg("output_tensors"),
          py::arg("all_active_ranks"),
          py::arg("per_group_segment_counts"),
          py::arg("async_op") = false,
          py::arg("low_precision") = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "sharded_relay_multi_group_all_gather",
          [](TorchCommRCCLX& self,
             std::vector<at::Tensor>& input_tensors,
             std::vector<at::Tensor>& output_tensors,
             const std::vector<std::vector<int64_t>>& all_active_ranks,
             const std::vector<int64_t>& per_group_send_counts,
             bool async_op,
             bool low_precision) {
            return self.sharded_relay_multi_group_all_gather(
                input_tensors,
                output_tensors,
                all_active_ranks,
                per_group_send_counts,
                async_op,
                low_precision);
          },
          R"(
Fused multi-group sharded relay all-gather for 2D sparse parallelism.

All-gather analogue of sharded_relay_multi_group_reduce_scatter (and the dual of
it). Executes multiple all-gather groups in lockstep phases to eliminate XGMI
link contention on MI300x GPUs. Each group has a power-of-two number of active
ranks (2 or 4); the logical collective is an all-gather among them, accelerated
by passthrough helpers (A==2 uses the original 2-active passthrough path; A>2
uses the bandwidth-optimal flat scatter->forward relay -- the dual of the
reduce-scatter -- i.e. a direct intra all-to-all woven with a 2-hop offload
through the idle helper GPUs). There is NO reduction (pure data movement) and
NO op.

For each active rank, the input holds per_group_send_counts[g] elements (its
contribution) and the output holds nActiveRanks x per_group_send_counts[g]
elements (output[i x sendCount] receives active index i's contribution).

Supports both in-place and out-of-place. In-place is detected when the active
input aliases output + myActiveIndex x sendCount (standard NCCL all-gather
in-place convention).

Args:
    input_tensors: List of send tensors (one per group). Active rank: holds
        per_group_send_counts[g] elements. Helper rank: an nActiveRanks-slot
        scratch tensor.
    output_tensors: List of receive tensors (one per group). Active rank: holds
        nActiveRanks x per_group_send_counts[g] elements. Helper rank: the same
        scratch tensor as the input.
    all_active_ranks: List of lists of active rank IDs per sparse group.
    per_group_send_counts: List of per-rank contribution counts (one per group).
    async_op: If True, returns a TorchWork handle for async operation
    low_precision: If True, use the fp8e4m3 wire format where it pays. See
        sharded_relay_multi_group_all_reduce -- same gate, same COLLECTIVE
        contract.

Returns:
    TorchWork object for operation completion if async_op=True
)",
          py::arg("input_tensors"),
          py::arg("output_tensors"),
          py::arg("all_active_ranks"),
          py::arg("per_group_send_counts"),
          py::arg("async_op") = false,
          py::arg("low_precision") = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "relay_control_publish",
          [](TorchCommRCCLX& self,
             uint64_t epoch,
             const std::vector<int64_t>& counts,
             int64_t op_code,
             int64_t dtype,
             int64_t timeout_ns,
             int64_t red_op,
             int64_t flags) {
            self.relay_control_publish(
                epoch, counts, op_code, dtype, red_op, flags, timeout_ns);
          },
          R"(
Publish the plan for one relay forward. Rank 0 only.

A communicator is a data plane, not a scheduler: nothing in it can make a helper
process post a collective. So the active side publishes what the forward will do
into a shared-memory segment, and every rank that does not already know the plan
consumes it before enqueueing. Host-only and synchronous -- no stream, no work
handle.

Bounded by timeout_ns rather than waiting forever, so a consumer that has fallen
more than the ring depth behind raises instead of hanging.

Args:
    epoch: Monotonic forward counter. Must increase by 1 per forward.
    counts: One element count per relay call in this forward. Its length is the
        call count, so a forward with a different number of chunks needs no
        separate field.
    op_code: An ncclRelayOp_t value (0 = shutdown, 1 = all_reduce,
        2 = reduce_scatter, 3 = all_gather, 4 = all_to_all).
    dtype: ncclDataType_t value the calls will use.
    timeout_ns: Bound on the wait for ring space. Precedes red_op and flags
        because those two carry defaults and a defaulted parameter cannot come
        before a required one -- listed here in the order a POSITIONAL caller
        must pass them, which is the order py::arg enforces below.
    red_op: ncclRedOp_t value, ignored for the non-reducing collectives.
    flags: Reserved; pass 0.

Raises:
    RuntimeError: if publishing fails, including a rank other than 0 calling it
        and a plan with more calls than NCCL_RELAY_CONTROL_MAX_CALLS.
)",
          py::arg("epoch"),
          py::arg("counts"),
          py::arg("op_code"),
          py::arg("dtype"),
          py::arg("timeout_ns"),
          py::arg("red_op") = 0,
          py::arg("flags") = 0,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "relay_control_consume",
          [](TorchCommRCCLX& self, uint64_t epoch, int64_t timeout_ns) {
            return self.relay_control_consume(epoch, timeout_ns);
          },
          R"(
Consume the plan for one relay forward. Called by every rank that does not
already know it -- in practice the helpers.

Blocks until the publisher has written `epoch`, then returns it. The counts
buffer is owned and sized by the wrapper from NCCL_RELAY_CONTROL_MAX_CALLS, so
the caller passes nothing and gets back a right-sized list.

Args:
    epoch: The forward to read. Must match what the publisher wrote.
    timeout_ns: Bound on the wait. A publisher that stops raises here rather
        than leaving this rank blocked forever.

Returns:
    (op_code, dtype, red_op, flags, counts) -- counts has one entry per relay
    call, so len(counts) is the number of calls to enqueue.

Raises:
    RuntimeError: on timeout, on a plan too large for the wrapper's buffer, or
        if the publisher signalled an abort.
)",
          py::arg("epoch"),
          py::arg("timeout_ns"),
          py::call_guard<py::gil_scoped_release>());
}
