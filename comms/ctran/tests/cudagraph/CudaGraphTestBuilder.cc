// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/tests/cudagraph/CudaGraphTestBuilder.h"

#include <chrono>
#include <optional>
#include <thread>

#include <gtest/gtest.h>

#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/interfaces/ICtran.h"

namespace ctran::testing {

GpePoolGuard::GpePoolGuard(CtranGpe* gpe, size_t maxGrowth)
    : gpe_(gpe),
      maxGrowth_(maxGrowth),
      initialFlags_(gpe->numInUseKernelFlags()),
      initialElems_(gpe->numInUseKernelElems()),
      initialChecksums_(gpe->numInUseChecksums()),
      initialSyncs_(gpe->numInUseGpeKernelSyncs()) {}

GpePoolGuard::~GpePoolGuard() {
  EXPECT_LE(gpe_->numInUseKernelFlags(), initialFlags_ + maxGrowth_)
      << "KernelFlag pool growing unboundedly";
  EXPECT_LE(gpe_->numInUseKernelElems(), initialElems_ + maxGrowth_)
      << "KernelElem pool growing unboundedly";
  EXPECT_LE(gpe_->numInUseChecksums(), initialChecksums_ + maxGrowth_)
      << "Checksum pool growing unboundedly";
  EXPECT_LE(gpe_->numInUseGpeKernelSyncs(), initialSyncs_ + maxGrowth_)
      << "GpeKernelSync pool growing unboundedly";
}

CapturedGraph::~CapturedGraph() {
  if (graphExec) {
    cudaGraphExecDestroy(graphExec);
  }
  if (graph) {
    cudaGraphDestroy(graph);
  }
}

CapturedGraph::CapturedGraph(CapturedGraph&& other) noexcept
    : graph(other.graph),
      graphExec(other.graphExec),
      captureStream(other.captureStream) {
  other.graph = nullptr;
  other.graphExec = nullptr;
  other.captureStream = nullptr;
}

CapturedGraph& CapturedGraph::operator=(CapturedGraph&& other) noexcept {
  if (this != &other) {
    if (graphExec) {
      cudaGraphExecDestroy(graphExec);
    }
    if (graph) {
      cudaGraphDestroy(graph);
    }
    graph = other.graph;
    graphExec = other.graphExec;
    captureStream = other.captureStream;
    other.graph = nullptr;
    other.graphExec = nullptr;
    other.captureStream = nullptr;
  }
  return *this;
}

CtranGraphTestBuilder::CtranGraphTestBuilder(
    CtranComm* comm,
    int rank,
    int numRanks)
    : comm_(comm), rank_(rank), numRanks_(numRanks) {}

CtranGraphTestBuilder& CtranGraphTestBuilder::withNumReplays(int n) {
  numReplays_ = n;
  return *this;
}

CtranGraphTestBuilder& CtranGraphTestBuilder::addCapture(
    cudaStream_t,
    CaptureFn captureFn) {
  // addCapture is sugar for a single-step single-substep Graph schedule entry.
  // The stream parameter is ignored — scheduleReplay creates its own streams.
  schedule_.push_back({{ScheduleEntryType::Graph, std::move(captureFn)}});
  return *this;
}

CtranGraphTestBuilder& CtranGraphTestBuilder::addCapture(CaptureFn captureFn) {
  return addCapture(nullptr, std::move(captureFn));
}

CtranGraphTestBuilder& CtranGraphTestBuilder::addSchedule(Schedule schedule) {
  EXPECT_TRUE(schedule_.empty())
      << "addSchedule() cannot be combined with addCapture()";
  schedule_ = std::move(schedule);
  return *this;
}

CtranGraphTestBuilder& CtranGraphTestBuilder::withReset(ResetFn resetFn) {
  resetFn_ = std::move(resetFn);
  return *this;
}

CtranGraphTestBuilder& CtranGraphTestBuilder::withVerify(VerifyFn verifyFn) {
  verifyFn_ = std::move(verifyFn);
  return *this;
}

CtranGraphTestBuilder& CtranGraphTestBuilder::withDeviceVerify(
    DeviceVerifyFn fn) {
  deviceVerifyFn_ = std::move(fn);
  mismatchCounter_.emplace();
  return *this;
}

CtranGraphTestBuilder& CtranGraphTestBuilder::withGraphAssertions(
    GraphAssertionsFn fn) {
  graphAssertionsFn_ = std::move(fn);
  return *this;
}

CtranGraphTestBuilder& CtranGraphTestBuilder::withResourcePoolCheck(
    bool enable) {
  checkResourcePools_ = enable;
  return *this;
}

void CtranGraphTestBuilder::run() {
  ASSERT_FALSE(schedule_.empty())
      << "Must call addCapture() or addSchedule() before run()";
  scheduleReplay();
  if (mismatchCounter_) {
    mismatchCounter_->assertZero();
  }
  destroy();
  verifyGpeLeak();
}

void CtranGraphTestBuilder::scheduleReplay() {
  // Allocate per-substep streams and graph holders (initially null).
  // Graph entries are captured on the first replay iteration.
  std::vector<std::vector<meta::comms::CudaStream>> stepStreams;
  scheduleGraphs_.resize(schedule_.size());
  stepStreams.reserve(schedule_.size());
  for (size_t s = 0; s < schedule_.size(); ++s) {
    auto& step = schedule_[s];
    scheduleGraphs_[s].resize(step.size());
    stepStreams.emplace_back();
    stepStreams.back().reserve(step.size());
    for (size_t j = 0; j < step.size(); ++j) {
      stepStreams.back().emplace_back(cudaStreamNonBlocking);
    }
  }

  size_t numGraphEntries = 0;
  for (auto& step : schedule_) {
    for (auto& entry : step) {
      if (entry.type == ScheduleEntryType::Graph) {
        ++numGraphEntries;
      }
    }
  }

  // Pool guard is created after the first iteration (when graphs are captured)
  // so the snapshot reflects the post-capture state.
  std::optional<GpePoolGuard> poolGuard;
  cudaEvent_t iterDoneEvent = nullptr;

  for (int iter = 0; iter < numReplays_; ++iter) {
    // Reset buffers. The first stream from step 0 is used for async reset
    // so it's ordered before the schedule steps on that stream.
    cudaStream_t firstStream = stepStreams[0][0].get();

    // Wait for previous iteration's verify to complete before resetting
    // buffers (verify may run on a different stream in multi-step schedules).
    if (iterDoneEvent) {
      ASSERT_EQ(
          cudaStreamWaitEvent(firstStream, iterDoneEvent, 0), cudaSuccess);
      cudaEventDestroy(iterDoneEvent);
      iterDoneEvent = nullptr;
    }

    if (resetFn_) {
      resetFn_(firstStream);
    }

    std::vector<cudaEvent_t> prevEvents;

    for (size_t s = 0; s < schedule_.size(); ++s) {
      auto& step = schedule_[s];
      std::vector<cudaEvent_t> stepEvents;
      stepEvents.reserve(step.size());

      for (size_t j = 0; j < step.size(); ++j) {
        cudaStream_t stream = stepStreams[s][j].get();

        // Wait on all events from the previous step.
        for (auto& ev : prevEvents) {
          ASSERT_EQ(cudaStreamWaitEvent(stream, ev, 0), cudaSuccess);
        }

        if (step[j].type == ScheduleEntryType::Eager) {
          // Eager: run lambda directly on the stream.
          CaptureContext ctx{comm_, stream, rank_, numRanks_};
          step[j].fn(ctx);
        } else {
          // Graph: capture on first iteration, replay on subsequent.
          auto& cg = scheduleGraphs_[s][j];
          if (!cg) {
            cg = std::make_unique<CapturedGraph>();
            cg->captureStream = stream;

            ASSERT_EQ(
                cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                cudaSuccess);

            CaptureContext ctx{comm_, stream, rank_, numRanks_};
            step[j].fn(ctx);

            ASSERT_EQ(cudaStreamEndCapture(stream, &cg->graph), cudaSuccess);
            ASSERT_NE(cg->graph, nullptr);

            ASSERT_EQ(
                cudaGraphInstantiate(
                    &cg->graphExec, cg->graph, nullptr, nullptr, 0),
                cudaSuccess);
            ASSERT_NE(cg->graphExec, nullptr);
          }
          ASSERT_EQ(cudaGraphLaunch(cg->graphExec, stream), cudaSuccess);
        }

        // Record an event after this substep completes.
        cudaEvent_t ev;
        ASSERT_EQ(cudaEventCreate(&ev), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(ev, stream), cudaSuccess);
        stepEvents.push_back(ev);
      }

      // Destroy previous step events and move current to prev.
      for (auto& ev : prevEvents) {
        cudaEventDestroy(ev);
      }
      prevEvents = std::move(stepEvents);
    }

    // Device-side verify: launch comparison kernel on the last step's stream.
    // No device sync needed — the kernel is ordered after the schedule steps.
    cudaStream_t lastStream = stepStreams.back().back().get();
    if (deviceVerifyFn_) {
      // Ensure all steps complete before verify (join all prevEvents to
      // lastStream).
      for (auto& ev : prevEvents) {
        ASSERT_EQ(cudaStreamWaitEvent(lastStream, ev, 0), cudaSuccess);
      }
      deviceVerifyFn_(lastStream, mismatchCounter_->ptr());

      // Record event so the next iteration's reset waits for verify to finish
      // reading the buffers (verify and reset may be on different streams).
      ASSERT_EQ(cudaEventCreate(&iterDoneEvent), cudaSuccess);
      ASSERT_EQ(cudaEventRecord(iterDoneEvent, lastStream), cudaSuccess);
    }

    // Clean up remaining events.
    for (auto& ev : prevEvents) {
      cudaEventDestroy(ev);
    }

    // After first iteration: sync for graph assertions and pool guard setup.
    // Subsequent iterations run fully async (no device sync).
    if (iter == 0) {
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

      if (graphAssertionsFn_) {
        std::vector<GraphTopology> topos;
        for (auto& stepGraphs : scheduleGraphs_) {
          for (auto& cg : stepGraphs) {
            if (cg) {
              topos.push_back(getGraphTopology(cg->graph));
            }
          }
        }
        if (!topos.empty()) {
          graphAssertionsFn_(topos);
        }
      }
      if (checkResourcePools_ && comm_->ctran_ && numGraphEntries > 0) {
        poolGuard.emplace(comm_->ctran_->gpe.get(), numGraphEntries);
      }
    }
  }

  if (iterDoneEvent) {
    cudaEventDestroy(iterDoneEvent);
  }

  // One final sync after all iterations.
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Legacy CPU-side verify (runs after final sync).
  if (verifyFn_) {
    verifyFn_();
  }

  // Move captured graphs into graphs_ so destroy() and verifyGpeLeak() work.
  for (auto& stepGraphs : scheduleGraphs_) {
    for (auto& cg : stepGraphs) {
      if (cg) {
        graphs_.push_back(std::move(cg));
      }
    }
  }
  scheduleGraphs_.clear();
}

void CtranGraphTestBuilder::destroy() {
  graphs_.clear();
  cudaDeviceSynchronize();
}

void CtranGraphTestBuilder::verifyGpeLeak() {
  if (!comm_->ctran_) {
    return;
  }
  // GPE pools use lazy reclaim — items may not be returned to the free pool
  // immediately after graph destroy. Spin until all counters reach zero,
  // with a timeout to catch real leaks.
  constexpr int kMaxSpinMs = 5000;
  constexpr int kSpinIntervalMs = 10;
  int elapsedMs = 0;
  while (elapsedMs < kMaxSpinMs) {
    if (comm_->ctran_->gpe->numInUseKernelFlags() == 0 &&
        comm_->ctran_->gpe->numInUseKernelElems() == 0 &&
        comm_->ctran_->gpe->numInUseChecksums() == 0 &&
        comm_->ctran_->gpe->numInUseGpeKernelSyncs() == 0) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kSpinIntervalMs));
    elapsedMs += kSpinIntervalMs;
  }
  EXPECT_EQ(comm_->ctran_->gpe->numInUseKernelFlags(), 0)
      << "KernelFlag leak detected after graph destroy";
  EXPECT_EQ(comm_->ctran_->gpe->numInUseKernelElems(), 0)
      << "KernelElem leak detected after graph destroy";
  EXPECT_EQ(comm_->ctran_->gpe->numInUseChecksums(), 0)
      << "Checksum leak detected after graph destroy";
  EXPECT_EQ(comm_->ctran_->gpe->numInUseGpeKernelSyncs(), 0)
      << "GpeKernelSync leak detected after graph destroy";
}

} // namespace ctran::testing
