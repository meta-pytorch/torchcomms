// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/tests/cudagraph/DeviceVerify.h"
#include "comms/utils/CudaRAII.h"
#include "comms/utils/test_utils/CudaGraphTestUtils.h"

class CtranGpe;

namespace ctran::testing {

struct CaptureContext {
  CtranComm* comm;
  cudaStream_t stream;
  int rank;
  int numRanks;
};

struct CapturedGraph {
  ~CapturedGraph();
  CapturedGraph() = default;

  CapturedGraph(const CapturedGraph&) = delete;
  CapturedGraph& operator=(const CapturedGraph&) = delete;
  CapturedGraph(CapturedGraph&& other) noexcept;
  CapturedGraph& operator=(CapturedGraph&& other) noexcept;

  cudaGraph_t graph{nullptr};
  cudaGraphExec_t graphExec{nullptr};
  cudaStream_t captureStream{nullptr};
};

using CaptureFn = std::function<void(CaptureContext& ctx)>;
using ResetFn = std::function<void(cudaStream_t stream)>;
using VerifyFn = std::function<void()>;
using DeviceVerifyFn =
    std::function<void(cudaStream_t stream, unsigned int* mismatchCount_d)>;
using GraphAssertionsFn =
    std::function<void(const std::vector<GraphTopology>& topos)>;

// Schedule types for pipeline-style execution (fork-join across steps).
enum class ScheduleEntryType { Eager, Graph };

struct ScheduleEntry {
  ScheduleEntryType type;
  CaptureFn fn;
};

using ScheduleStep = std::vector<ScheduleEntry>;
using Schedule = std::vector<ScheduleStep>;

// RAII guard that snapshots GPE pool sizes on construction and asserts
// they haven't grown unboundedly on destruction.
class GpePoolGuard {
 public:
  GpePoolGuard(CtranGpe* gpe, size_t maxGrowth);
  ~GpePoolGuard();

  GpePoolGuard(const GpePoolGuard&) = delete;
  GpePoolGuard& operator=(const GpePoolGuard&) = delete;

 private:
  CtranGpe* gpe_;
  size_t maxGrowth_;
  size_t initialFlags_;
  size_t initialElems_;
  size_t initialChecksums_;
  size_t initialSyncs_;
};

// Builder for CTRAN CUDA graph tests with schedule-based pipeline execution.
//
// Supports two APIs:
//   addCapture(fn)    — sugar for a single Graph step
//   addSchedule(sched) — full pipeline with Eager/Graph fork-join steps
//
// Usage (simple):
//   CtranGraphTestBuilder(comm, rank, nRanks)
//       .addCapture([](CaptureContext& ctx) { ... })
//       .withReset([]() { ... })
//       .withVerify([]() { ... })
//       .run();
//
// Usage (pipeline):
//   CtranGraphTestBuilder(comm, rank, nRanks)
//       .addSchedule({
//           {{Graph, captureFn}},                  // step 1: graph only
//           {{Eager, eagerFn}, {Graph, captureFn}}, // step 2: concurrent
//           {{Eager, eagerFn}},                    // step 3: eager only
//       })
//       .withReset([]() { ... })
//       .withVerify([]() { ... })
//       .run();
class CtranGraphTestBuilder {
 public:
  CtranGraphTestBuilder(CtranComm* comm, int rank, int numRanks);
  CtranGraphTestBuilder(const CtranGraphTestBuilder&) = delete;
  CtranGraphTestBuilder& operator=(const CtranGraphTestBuilder&) = delete;
  CtranGraphTestBuilder(CtranGraphTestBuilder&&) = default;
  CtranGraphTestBuilder& operator=(CtranGraphTestBuilder&&) = default;

  CtranGraphTestBuilder& withNumReplays(int n);

  CtranGraphTestBuilder& addCapture(cudaStream_t stream, CaptureFn captureFn);
  CtranGraphTestBuilder& addCapture(CaptureFn captureFn);

  CtranGraphTestBuilder& addSchedule(Schedule schedule);

  CtranGraphTestBuilder& withReset(ResetFn resetFn);
  CtranGraphTestBuilder& withVerify(VerifyFn verifyFn);
  CtranGraphTestBuilder& withDeviceVerify(DeviceVerifyFn fn);
  CtranGraphTestBuilder& withGraphAssertions(GraphAssertionsFn fn);
  CtranGraphTestBuilder& withResourcePoolCheck(bool enable);

  void run();

  const std::vector<std::unique_ptr<CapturedGraph>>& capturedGraphs() const {
    return graphs_;
  }

 private:
  void scheduleReplay();
  void destroy();
  void verifyGpeLeak();

  CtranComm* comm_;
  int rank_;
  int numRanks_;
  int numReplays_{3};
  bool checkResourcePools_{true};

  Schedule schedule_;

  // Captured graphs (populated by scheduleReplay, consumed by
  // destroy/verifyGpeLeak)
  std::vector<std::unique_ptr<CapturedGraph>> graphs_;

  // Per-substep graph state for schedule replay (indexed by [step][substep]).
  // Graph entries get captured on first replay; Eager entries leave these null.
  std::vector<std::vector<std::unique_ptr<CapturedGraph>>> scheduleGraphs_;

  ResetFn resetFn_;
  VerifyFn verifyFn_;
  DeviceVerifyFn deviceVerifyFn_;
  GraphAssertionsFn graphAssertionsFn_;
  std::optional<DeviceMismatchCounter> mismatchCounter_;
};

} // namespace ctran::testing
