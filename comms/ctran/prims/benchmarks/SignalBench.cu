// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/prims/ThreadGroup.cuh"
#include "comms/ctran/prims/benchmarks/SignalBench.cuh"

namespace ctran::prims::benchmark {

__global__ void signalAddBenchKernel(
    SignalState* remote,
    SignalState* local,
    int nSteps,
    bool useBlockGroups) {
  auto group = useBlockGroups ? make_block_group() : make_warp_group();
  for (int step = 1; step <= nSteps; ++step) {
    remote[group.group_id].signal(group, SignalOp::SIGNAL_ADD, 1);
    local[group.group_id].wait_until(group, CmpOp::CMP_EQ, step);
  }
}

__global__ void signalSetBenchKernel(
    SignalState* remote,
    SignalState* local,
    int nSteps,
    bool useBlockGroups) {
  auto group = useBlockGroups ? make_block_group() : make_warp_group();
  for (int step = 1; step <= nSteps; ++step) {
    remote[group.group_id].signal(group, SignalOp::SIGNAL_SET, step);
    local[group.group_id].wait_until(group, CmpOp::CMP_EQ, step);
  }
}

} // namespace ctran::prims::benchmark
