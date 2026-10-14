// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Separate binary from GraphCollTraceUT: the globaltimer calibration singleton
// is never reset, so nothing here may touch it before the test runs. That makes
// CollTrace's warmup thread pay first-time calibration while a capture is open.

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/colltrace/CollTrace.h"
#include "comms/utils/colltrace/GraphCudaWaitEvent.h"
#include "comms/utils/cvars/nccl_cvars.h"

using meta::comms::colltrace::CollTrace;
using meta::comms::colltrace::CollTraceConfig;
using meta::comms::colltrace::GraphCudaWaitEvent;
using meta::comms::colltrace::ICollMetadata;
using meta::comms::colltrace::ICollTracePlugin;

namespace {

constexpr auto kDeadline = std::chrono::seconds(30);

class SimpleMetadata : public ICollMetadata {
 public:
  std::size_t hash() const override {
    return 0;
  }
  bool equals(const ICollMetadata&) const noexcept override {
    return true;
  }
  std::string_view getMetadataType() const noexcept override {
    return "test";
  }
  folly::dynamic toDynamic() const noexcept override {
    return folly::dynamic::object("type", "test");
  }
  void fromDynamic(const folly::dynamic&) noexcept override {}
};

TEST(GraphCollTraceColdCalibrationTest, ColdWarmupOverlapsOpenCapture) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaSetDevice(0);
  if (!meta::comms::colltrace::graphColltraceSupported(
          "GraphCollTraceColdCalibrationTest")) {
    GTEST_SKIP() << "graph colltrace unsupported on this device (needs sm_90+)";
  }
  EnvRAII<bool> cvarGuard(NCCL_COLLTRACE_TRACE_CUDA_GRAPH, true);
  EnvRAII<bool> asyncWarmupGuard(NCCL_COLLTRACE_ASYNC_GRAPH_WARMUP, true);
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  auto gate = std::make_shared<std::promise<void>>();
  auto gateFuture = gate->get_future().share();
  // The setup function leaves the warmup thread in global capture mode, so
  // warmupGraphTracing's own guard is what keeps calibration out of the
  // capture.
  auto colltrace = std::make_shared<CollTrace>(
      CollTraceConfig{
          .maxCheckCancelInterval = std::chrono::milliseconds{1},
          .beforeGraphWarmupHook =
              [gateFuture]() {
                if (gateFuture.wait_for(kDeadline) !=
                    std::future_status::ready) {
                  ADD_FAILURE() << "warmup gate never released";
                }
              }},
      CommLogData{},
      []() -> meta::comms::CommsMaybeVoid {
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        cudaSetDevice(0);
        return folly::unit;
      },
      std::vector<std::unique_ptr<ICollTracePlugin>>{});

  auto captureBegun = std::make_shared<std::promise<void>>();
  auto captureBegunFuture = captureBegun->get_future();
  auto recordFuture =
      std::async(std::launch::async, [stream, colltrace, captureBegun]() {
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        cudaSetDevice(0);
        auto beginErr =
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        captureBegun->set_value();
        auto result = colltrace->recordCollective(
            std::make_unique<SimpleMetadata>(),
            std::make_unique<GraphCudaWaitEvent>(stream));
        cudaGraph_t graph = nullptr;
        auto endErr = cudaStreamEndCapture(stream, &graph);
        return std::tuple{beginErr, std::move(result), endErr, graph};
      });

  ASSERT_EQ(captureBegunFuture.wait_for(kDeadline), std::future_status::ready);
  gate->set_value();
  ASSERT_EQ(recordFuture.wait_for(kDeadline), std::future_status::ready)
      << "graph record did not return after cold warmup was released";
  auto [beginErr, result, endErr, graph] = recordFuture.get();

  EXPECT_EQ(beginErr, cudaSuccess);
  std::string err;
  if (result.hasError()) {
    err = result.error().message;
  }
  ASSERT_TRUE(result.hasValue()) << err;
  EXPECT_EQ(endErr, cudaSuccess);
  EXPECT_NE(graph, nullptr);
  if (graph != nullptr) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaGraphDestroy(graph);
  }
  colltrace.reset();
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamDestroy(stream);
}

} // namespace
