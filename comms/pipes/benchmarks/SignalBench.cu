// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/benchmarks/SignalBench.cuh"

namespace comms::pipes::benchmark {

__global__ void signalAddBenchKernel(
    Signal* remote,
    Signal* local,
    int nSteps,
    bool useBlockGroups) {
  auto group = useBlockGroups ? make_block_group() : make_warp_group();
  for (int step = 1; step <= nSteps; ++step) {
    remote[group.group_id].signal(group, SignalOp::SIGNAL_ADD, 1);
    local[group.group_id].wait_until(group, CmpOp::CMP_EQ, step);
  }
}

__global__ void signalSetBenchKernel(
    Signal* remote,
    Signal* local,
    int nSteps,
    bool useBlockGroups) {
  auto group = useBlockGroups ? make_block_group() : make_warp_group();
  for (int step = 1; step <= nSteps; ++step) {
    remote[group.group_id].signal(group, SignalOp::SIGNAL_SET, step);
    local[group.group_id].wait_until(group, CmpOp::CMP_EQ, step);
  }
}

} // namespace comms::pipes::benchmark
