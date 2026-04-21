// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/GraphCaptureSideStream.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

using meta::comms::GraphSideStream;

namespace {

// Tests inject captured non-event graph nodes via ``cudaMemsetAsync`` rather
// than a custom __global__ kernel — it's simpler and doesn't depend on which
// CUDA archs the test target compiles for.

class GraphSideStreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }

  static std::vector<cudaGraphNode_t> getNodes(cudaGraph_t graph) {
    size_t n = 0;
    EXPECT_EQ(cudaGraphGetNodes(graph, nullptr, &n), cudaSuccess);
    std::vector<cudaGraphNode_t> nodes(n);
    if (n > 0) {
      EXPECT_EQ(cudaGraphGetNodes(graph, nodes.data(), &n), cudaSuccess);
    }
    return nodes;
  }

  static std::vector<cudaGraphNode_t> getSuccs(cudaGraphNode_t node) {
    size_t n = 0;
    EXPECT_EQ(cudaGraphNodeGetDependentNodes(node, nullptr, &n), cudaSuccess);
    std::vector<cudaGraphNode_t> out(n);
    if (n > 0) {
      EXPECT_EQ(
          cudaGraphNodeGetDependentNodes(node, out.data(), &n), cudaSuccess);
    }
    return out;
  }

  static std::vector<cudaGraphNode_t> getPreds(cudaGraphNode_t node) {
    size_t n = 0;
    EXPECT_EQ(cudaGraphNodeGetDependencies(node, nullptr, &n), cudaSuccess);
    std::vector<cudaGraphNode_t> out(n);
    if (n > 0) {
      EXPECT_EQ(
          cudaGraphNodeGetDependencies(node, out.data(), &n), cudaSuccess);
    }
    return out;
  }

  static cudaGraphNodeType nodeType(cudaGraphNode_t node) {
    cudaGraphNodeType t;
    EXPECT_EQ(cudaGraphNodeGetType(node, &t), cudaSuccess);
    return t;
  }
};

TEST_F(GraphSideStreamTest, ConstructAndDestruct) {
  GraphSideStream side;
  EXPECT_NE(side.get(), nullptr);
}

// When the caller's stream is NOT currently under graph capture, fork_from
// should fall back to invoking ``fn`` with the caller's stream directly.
TEST_F(GraphSideStreamTest, ForkFromFallsBackWhenNotCapturing) {
  GraphSideStream side;
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  cudaStream_t invoked_with = nullptr;
  int invocation_count = 0;
  EXPECT_EQ(
      side.fork_from(
          stream,
          [&](cudaStream_t passed) {
            ++invocation_count;
            invoked_with = passed;
          }),
      cudaSuccess);
  EXPECT_EQ(invocation_count, 1);
  EXPECT_EQ(invoked_with, stream)
      << "fallback path must pass the caller's stream, not the side stream";

  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// End-to-end: during graph capture, fork_from should:
//   - Invoke the user fn with the SIDE stream.
//   - Produce a captured graph where the side-stream work is NOT a
//     predecessor of the caller's subsequent ops.
//   - Keep ``cudaStreamEndCapture`` happy (rejoin node present in graph).
TEST_F(GraphSideStreamTest, ForkFromRoutesWorkOffMainCriticalPath) {
  GraphSideStream side;

  cudaStream_t main = nullptr;
  ASSERT_EQ(cudaStreamCreate(&main), cudaSuccess);

  int* dev_counter = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_counter, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_counter, 0, sizeof(int)), cudaSuccess);

  cudaEvent_t ext_event = nullptr;
  ASSERT_EQ(
      cudaEventCreateWithFlags(&ext_event, cudaEventDisableTiming),
      cudaSuccess);

  ASSERT_EQ(
      cudaStreamBeginCapture(main, cudaStreamCaptureModeThreadLocal),
      cudaSuccess);

  ASSERT_EQ(
      cudaMemsetAsync(dev_counter, 0, sizeof(int), main),
      cudaSuccess); // kernel1

  cudaStream_t invoked_with = nullptr;
  ASSERT_EQ(
      side.fork_from(
          main,
          [&](cudaStream_t s) {
            invoked_with = s;
            (void)cudaEventRecordWithFlags(
                ext_event, s, cudaEventRecordExternal);
          }),
      cudaSuccess);
  EXPECT_EQ(invoked_with, side.get())
      << "under active capture, fn must run on the side stream";

  ASSERT_EQ(
      cudaMemsetAsync(dev_counter, 0, sizeof(int), main),
      cudaSuccess); // kernel2

  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(main, &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);

  // Classify nodes. We injected two cudaMemsetAsync ops as "main stream
  // anchors" (kernel1 / kernel2 in role) and a single external EVENT_RECORD
  // via the side stream.
  auto nodes = getNodes(graph);
  cudaGraphNode_t kernel1 = nullptr;
  cudaGraphNode_t kernel2 = nullptr;
  cudaGraphNode_t event_record = nullptr;
  for (auto n : nodes) {
    auto t = nodeType(n);
    if (t == cudaGraphNodeTypeMemset) {
      if (kernel1 == nullptr) {
        kernel1 = n;
      } else {
        kernel2 = n;
      }
    } else if (t == cudaGraphNodeTypeEventRecord) {
      // Capture the user-issued external record (the first EVENT_RECORD
      // encountered — fork/rejoin records also exist on side but are not
      // what the test asserts on structurally).
      if (event_record == nullptr) {
        event_record = n;
      }
    }
  }
  ASSERT_NE(kernel1, nullptr);
  ASSERT_NE(kernel2, nullptr);
  ASSERT_NE(event_record, nullptr);

  // kernel2 must depend directly on kernel1 (and NOT transitively via the
  // event record node).
  auto k2_preds = getPreds(kernel2);
  bool kernel2_depends_on_kernel1_directly = false;
  for (auto p : k2_preds) {
    EXPECT_NE(p, event_record)
        << "kernel2 must not depend on the external event record";
    if (p == kernel1) {
      kernel2_depends_on_kernel1_directly = true;
    }
  }
  EXPECT_TRUE(kernel2_depends_on_kernel1_directly)
      << "kernel2 should have a direct kernel1 edge after rewind";

  // And the event record itself must NOT have kernel2 as a descendant.
  auto er_succs = getSuccs(event_record);
  for (auto s : er_succs) {
    EXPECT_NE(s, kernel2)
        << "event record must not be a predecessor of kernel2";
  }

  EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  EXPECT_EQ(cudaEventDestroy(ext_event), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(main), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_counter), cudaSuccess);
}

// Instantiating and replaying the captured graph after fork_from must also
// execute cleanly — guards against accidentally breaking the DAG structure
// such that cudaGraphInstantiate fails.
TEST_F(GraphSideStreamTest, CapturedGraphInstantiatesAndReplays) {
  GraphSideStream side;
  cudaStream_t main = nullptr;
  ASSERT_EQ(cudaStreamCreate(&main), cudaSuccess);
  int* dev_counter = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_counter, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_counter, 0, sizeof(int)), cudaSuccess);
  cudaEvent_t ext_event = nullptr;
  ASSERT_EQ(
      cudaEventCreateWithFlags(&ext_event, cudaEventDisableTiming),
      cudaSuccess);

  ASSERT_EQ(
      cudaStreamBeginCapture(main, cudaStreamCaptureModeThreadLocal),
      cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(dev_counter, 0, sizeof(int), main), cudaSuccess);
  ASSERT_EQ(
      side.fork_from(
          main,
          [&](cudaStream_t s) {
            (void)cudaEventRecordWithFlags(
                ext_event, s, cudaEventRecordExternal);
          }),
      cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(dev_counter, 0, sizeof(int), main), cudaSuccess);

  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(main, &graph), cudaSuccess);

  cudaGraphExec_t exec = nullptr;
  ASSERT_EQ(
      cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(exec, main), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(main), cudaSuccess);

  EXPECT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  EXPECT_EQ(cudaEventDestroy(ext_event), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(main), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_counter), cudaSuccess);
}

} // namespace
