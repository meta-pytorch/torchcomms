// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/GraphCaptureSideStream.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <set>
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

  // CUDA 13 folded the optional edge-data output into the dependency queries
  // themselves; HIP and earlier CUDA runtimes keep the three-argument form.
  static cudaError_t
  dependentNodes(cudaGraphNode_t node, cudaGraphNode_t* out, size_t* n) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
    return hipGraphNodeGetDependentNodes(node, out, n);
#elif CUDART_VERSION >= 13000
    return cudaGraphNodeGetDependentNodes(node, out, nullptr, n);
#else
    return cudaGraphNodeGetDependentNodes(node, out, n);
#endif
  }

  static cudaError_t
  dependencies(cudaGraphNode_t node, cudaGraphNode_t* out, size_t* n) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
    return hipGraphNodeGetDependencies(node, out, n);
#elif CUDART_VERSION >= 13000
    return cudaGraphNodeGetDependencies(node, out, nullptr, n);
#else
    return cudaGraphNodeGetDependencies(node, out, n);
#endif
  }

  static std::vector<cudaGraphNode_t> getSuccs(cudaGraphNode_t node) {
    size_t n = 0;
    EXPECT_EQ(dependentNodes(node, nullptr, &n), cudaSuccess);
    std::vector<cudaGraphNode_t> out(n);
    if (n > 0) {
      EXPECT_EQ(dependentNodes(node, out.data(), &n), cudaSuccess);
    }
    return out;
  }

  static std::vector<cudaGraphNode_t> getPreds(cudaGraphNode_t node) {
    size_t n = 0;
    EXPECT_EQ(dependencies(node, nullptr, &n), cudaSuccess);
    std::vector<cudaGraphNode_t> out(n);
    if (n > 0) {
      EXPECT_EQ(dependencies(node, out.data(), &n), cudaSuccess);
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

// Simulates two back-to-back async collectives inside a single graph capture,
// mirroring the TorchWorkNCCLX lifecycle:
//
//   Per collective:
//     recordStart    → fork(start_event external) onto side stream
//     NCCL kernel    → main stream
//     recordEnd      → fork(end_event external) onto side stream
//                    → eventRecord(sync_event) on main stream
//     overlap compute → main stream (runs BEFORE work.wait — this is the
//                       compute that should overlap with the collective and
//                       must NOT be blocked by the external event records)
//     work.wait()    → cudaStreamWaitEvent(main, sync_event)
//   post-wait compute → main stream
//
// Verifies:
//   - Overlap compute is NOT serialized by side-stream event records
//   - NCCL kernels ARE ancestors of post-wait compute
//   - Dep event and side stream are reused correctly across fork_from calls
//   - The graph instantiates and replays cleanly
TEST_F(GraphSideStreamTest, AsyncCollectiveLifecycleDoesNotBlockCompute) {
  GraphSideStream side;

  cudaStream_t main = nullptr;
  ASSERT_EQ(cudaStreamCreate(&main), cudaSuccess);

  int* dev_buf = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_buf, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev_buf, 0, sizeof(int)), cudaSuccess);

  // Separate buffer for overlap compute so memset nodes are distinguishable
  // by address if needed.
  int* overlap_buf = nullptr;
  ASSERT_EQ(cudaMalloc(&overlap_buf, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(overlap_buf, 0, sizeof(int)), cudaSuccess);

  // Two collectives, each with start + end external events and a sync event.
  constexpr int kNumCollectives = 2;
  cudaEvent_t start_events[kNumCollectives];
  cudaEvent_t end_events[kNumCollectives];
  cudaEvent_t sync_events[kNumCollectives];
  for (int i = 0; i < kNumCollectives; ++i) {
    ASSERT_EQ(
        cudaEventCreateWithFlags(&start_events[i], cudaEventDisableTiming),
        cudaSuccess);
    ASSERT_EQ(
        cudaEventCreateWithFlags(&end_events[i], cudaEventDisableTiming),
        cudaSuccess);
    ASSERT_EQ(
        cudaEventCreateWithFlags(&sync_events[i], cudaEventDisableTiming),
        cudaSuccess);
  }

  ASSERT_EQ(
      cudaStreamBeginCapture(main, cudaStreamCaptureModeThreadLocal),
      cudaSuccess);

  for (int c = 0; c < kNumCollectives; ++c) {
    // recordStart: fork start_event onto side stream
    ASSERT_EQ(
        side.fork_from(
            main,
            [&, c](cudaStream_t s) {
              (void)cudaEventRecordWithFlags(
                  start_events[c], s, cudaEventRecordExternal);
            }),
        cudaSuccess);

    // NCCL collective kernel (memset as stand-in)
    ASSERT_EQ(cudaMemsetAsync(dev_buf, 0, sizeof(int), main), cudaSuccess);

    // recordEnd: fork end_event onto side stream
    ASSERT_EQ(
        side.fork_from(
            main,
            [&, c](cudaStream_t s) {
              (void)cudaEventRecordWithFlags(
                  end_events[c], s, cudaEventRecordExternal);
            }),
        cudaSuccess);

    // recordEnd: sync_event stays on main stream (the join point)
    ASSERT_EQ(cudaEventRecord(sync_events[c], main), cudaSuccess);

    // Overlap compute: runs AFTER the collective launches but BEFORE
    // work.wait(). This is the compute that benefits from the side-stream
    // optimization — if the external event records were on the main stream,
    // their release fences would block this compute at replay time.
    ASSERT_EQ(cudaMemsetAsync(overlap_buf, 0, sizeof(int), main), cudaSuccess);

    // work.wait(): downstream ops wait on sync_event
    ASSERT_EQ(cudaStreamWaitEvent(main, sync_events[c], 0), cudaSuccess);
  }

  // Post-wait compute after both collectives complete
  ASSERT_EQ(cudaMemsetAsync(dev_buf, 0, sizeof(int), main), cudaSuccess);

  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(main, &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);

  // Classify nodes.
  auto nodes = getNodes(graph);
  std::vector<cudaGraphNode_t> memset_nodes;
  for (auto n : nodes) {
    if (nodeType(n) == cudaGraphNodeTypeMemset) {
      memset_nodes.push_back(n);
    }
  }

  // 2 NCCL kernels + 2 overlap computes + 1 post-wait compute = 5 memsets
  ASSERT_EQ(memset_nodes.size(), 5u);

  // Sort memsets topologically (fewer ancestors = earlier).
  std::sort(memset_nodes.begin(), memset_nodes.end(), [&](auto a, auto b) {
    return getPreds(a).size() < getPreds(b).size();
  });

  // For each overlap compute node, verify no EVENT_RECORD ancestor exists.
  // The overlap compute nodes are the ones that would be stalled if the
  // external event records were on the main stream instead of the side stream.
  for (auto& memset_node : memset_nodes) {
    // BFS ancestors of this memset node.
    std::vector<cudaGraphNode_t> bfs = {memset_node};
    std::set<cudaGraphNode_t> ancestors;
    while (!bfs.empty()) {
      auto cur = bfs.back();
      bfs.pop_back();
      if (!ancestors.insert(cur).second) {
        continue;
      }
      for (auto p : getPreds(cur)) {
        bfs.push_back(p);
      }
    }

    // No memset node should have an EVENT_RECORD ancestor. The side-stream
    // external records and fork/rejoin dep-event records must all be off
    // the critical path.
    for (auto a : ancestors) {
      EXPECT_NE(nodeType(a), cudaGraphNodeTypeEventRecord)
          << "memset node must not have any event record node as ancestor "
             "(side-stream external records should be off the critical path)";
    }
  }

  // The NCCL kernel memsets must be ancestors of the post-wait compute
  // (the last memset in topological order).
  auto& post_wait_compute = memset_nodes.back();
  std::vector<cudaGraphNode_t> bfs = {post_wait_compute};
  std::set<cudaGraphNode_t> post_wait_ancestors;
  while (!bfs.empty()) {
    auto cur = bfs.back();
    bfs.pop_back();
    if (!post_wait_ancestors.insert(cur).second) {
      continue;
    }
    for (auto p : getPreds(cur)) {
      bfs.push_back(p);
    }
  }
  for (size_t i = 0; i < memset_nodes.size() - 1; ++i) {
    EXPECT_TRUE(post_wait_ancestors.count(memset_nodes[i]))
        << "memset " << i << " must be an ancestor of post-wait compute";
  }

  // Verify that the side-stream event record nodes form a totally ordered
  // chain (they're all on the same captured side stream).
  //
  // Identify side-stream event records as EVENT_RECORD nodes that are NOT
  // ancestors of any memset node (main-stream ops).  The main-stream
  // dep_event records ARE ancestors of memsets; the side-stream external
  // records and rejoin dep_event records are not.
  std::set<cudaGraphNode_t> main_stream_ancestors;
  for (auto& mn : memset_nodes) {
    std::vector<cudaGraphNode_t> bfs_m = {mn};
    while (!bfs_m.empty()) {
      auto cur = bfs_m.back();
      bfs_m.pop_back();
      if (!main_stream_ancestors.insert(cur).second) {
        continue;
      }
      for (auto p : getPreds(cur)) {
        bfs_m.push_back(p);
      }
    }
  }

  std::vector<cudaGraphNode_t> side_event_records;
  for (auto n : nodes) {
    if (nodeType(n) == cudaGraphNodeTypeEventRecord &&
        main_stream_ancestors.find(n) == main_stream_ancestors.end()) {
      side_event_records.push_back(n);
    }
  }

  // All side-stream event records must form a total order (single chain).
  // Verify by checking that every pair has an ancestor relationship.
  // Helper: compute the full ancestor set of a node.
  auto getAncestors = [&](cudaGraphNode_t node) {
    std::set<cudaGraphNode_t> ancestors;
    std::vector<cudaGraphNode_t> q = {node};
    while (!q.empty()) {
      auto cur = q.back();
      q.pop_back();
      if (!ancestors.insert(cur).second) {
        continue;
      }
      for (auto p : getPreds(cur)) {
        q.push_back(p);
      }
    }
    return ancestors;
  };

  for (size_t i = 0; i < side_event_records.size(); ++i) {
    for (size_t j = i + 1; j < side_event_records.size(); ++j) {
      auto ancestors_j = getAncestors(side_event_records[j]);
      auto ancestors_i = getAncestors(side_event_records[i]);
      bool i_before_j = ancestors_j.count(side_event_records[i]) > 0;
      bool j_before_i = ancestors_i.count(side_event_records[j]) > 0;
      EXPECT_TRUE(i_before_j || j_before_i)
          << "side-stream event records " << i << " and " << j
          << " must be ordered (one must be an ancestor of the other)";
    }
  }

  // Verify the graph is valid by instantiating and replaying, and that
  // the external events are signaled after replay (watchdog can query them).
  cudaGraphExec_t exec = nullptr;
  ASSERT_EQ(
      cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(exec, main), cudaSuccess);

  // Wait on each external event directly. If the external records fired
  // during graph replay, cudaEventSynchronize returns immediately.
  // If they didn't, the test hangs (timeout).
  for (int i = 0; i < kNumCollectives; ++i) {
    EXPECT_EQ(cudaEventSynchronize(start_events[i]), cudaSuccess)
        << "start_event[" << i
        << "] must be signaled after graph replay (wait should not block)";
    EXPECT_EQ(cudaEventSynchronize(end_events[i]), cudaSuccess)
        << "end_event[" << i
        << "] must be signaled after graph replay (wait should not block)";
  }

  ASSERT_EQ(cudaStreamSynchronize(main), cudaSuccess);

  EXPECT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  for (int i = 0; i < kNumCollectives; ++i) {
    EXPECT_EQ(cudaEventDestroy(start_events[i]), cudaSuccess);
    EXPECT_EQ(cudaEventDestroy(end_events[i]), cudaSuccess);
    EXPECT_EQ(cudaEventDestroy(sync_events[i]), cudaSuccess);
  }
  EXPECT_EQ(cudaStreamDestroy(main), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_buf), cudaSuccess);
  EXPECT_EQ(cudaFree(overlap_buf), cudaSuccess);
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

#if CUDART_VERSION >= 12050
// An 8-SM green context on device 0.
void makeGreenContext(CUgreenCtx* green_ctx) {
  CUdevice dev = 0;
  ASSERT_EQ(cuDeviceGet(&dev, 0), CUDA_SUCCESS);
  CUdevResource sm = {};
  ASSERT_EQ(
      cuDeviceGetDevResource(dev, &sm, CU_DEV_RESOURCE_TYPE_SM), CUDA_SUCCESS);
  unsigned int groups = 1;
  CUdevResource green_sm = {};
  CUdevResource remaining = {};
  ASSERT_EQ(
      cuDevSmResourceSplitByCount(
          &green_sm, &groups, &sm, &remaining, 0, /*minCount=*/8),
      CUDA_SUCCESS);
  CUdevResourceDesc desc = nullptr;
  ASSERT_EQ(cuDevResourceGenerateDesc(&desc, &green_sm, 1), CUDA_SUCCESS);
  ASSERT_EQ(
      cuGreenCtxCreate(green_ctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM),
      CUDA_SUCCESS);
}

// A stream in a green context forks onto a side stream in that same green
// context, created on first use mid-capture and reused afterwards, so the
// captured fork never leaves the caller's context.
TEST_F(GraphSideStreamTest, ForkFromGreenContextStaysInGreenContext) {
  CUgreenCtx green_ctx = nullptr;
  ASSERT_NO_FATAL_FAILURE(makeGreenContext(&green_ctx));
  CUstream main = nullptr;
  ASSERT_EQ(
      cuGreenCtxStreamCreate(&main, green_ctx, CU_STREAM_NON_BLOCKING, 0),
      CUDA_SUCCESS);

  {
    GraphSideStream side;
    int* dev_counter = nullptr;
    ASSERT_EQ(cudaMalloc(&dev_counter, sizeof(int)), cudaSuccess);
    std::vector<cudaStream_t> forked;

    ASSERT_EQ(
        cudaStreamBeginCapture(main, cudaStreamCaptureModeGlobal), cudaSuccess);
    for (int i = 0; i < 2; ++i) {
      ASSERT_EQ(
          cudaMemsetAsync(dev_counter, 0, sizeof(int), main), cudaSuccess);
      ASSERT_EQ(
          side.fork_from(
              main,
              [&](cudaStream_t s) {
                forked.push_back(s);
                (void)cudaMemsetAsync(dev_counter, 1, sizeof(int), s);
              }),
          cudaSuccess);
    }
    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(main, &graph), cudaSuccess);

    ASSERT_EQ(forked.size(), 2u);
    EXPECT_NE(forked[0], side.get());
    EXPECT_EQ(forked[0], forked[1]);
    CUcontext ctx = nullptr;
    CUgreenCtx forked_green_ctx = nullptr;
    ASSERT_EQ(
        cuStreamGetCtx_v2(forked[0], &ctx, &forked_green_ctx), CUDA_SUCCESS);
    EXPECT_EQ(forked_green_ctx, green_ctx);

    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(exec, main), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(main), cudaSuccess);

    EXPECT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    EXPECT_EQ(cudaFree(dev_counter), cudaSuccess);
  }

  EXPECT_EQ(cuStreamDestroy(main), CUDA_SUCCESS);
  EXPECT_EQ(cuGreenCtxDestroy(green_ctx), CUDA_SUCCESS);
}

// The first fork from a device-0 green stream may happen while another device
// is current; the side stream's event must still belong to device 0, or this
// fork and every later one fail to record it.
TEST_F(GraphSideStreamTest, ForkFromGreenContextWithAnotherDeviceCurrent) {
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  if (device_count < 2) {
    GTEST_SKIP() << "Needs two CUDA devices";
  }
  CUgreenCtx green_ctx = nullptr;
  ASSERT_NO_FATAL_FAILURE(makeGreenContext(&green_ctx));
  CUstream main = nullptr;
  ASSERT_EQ(
      cuGreenCtxStreamCreate(&main, green_ctx, CU_STREAM_NON_BLOCKING, 0),
      CUDA_SUCCESS);

  {
    GraphSideStream side;
    int* dev_counter = nullptr;
    ASSERT_EQ(cudaMalloc(&dev_counter, sizeof(int)), cudaSuccess);
    int forks = 0;
    auto fork = [&](cudaStream_t s) {
      ++forks;
      (void)cudaMemsetAsync(dev_counter, 1, sizeof(int), s);
    };

    ASSERT_EQ(
        cudaStreamBeginCapture(main, cudaStreamCaptureModeGlobal), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(dev_counter, 0, sizeof(int), main), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    EXPECT_EQ(side.fork_from(main, fork), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_EQ(side.fork_from(main, fork), cudaSuccess);
    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(main, &graph), cudaSuccess);
    EXPECT_EQ(forks, 2);

    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(exec, main), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(main), cudaSuccess);

    EXPECT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    EXPECT_EQ(cudaFree(dev_counter), cudaSuccess);
  }

  EXPECT_EQ(cuStreamDestroy(main), CUDA_SUCCESS);
  EXPECT_EQ(cuGreenCtxDestroy(green_ctx), CUDA_SUCCESS);
}

// Two collectives captured on streams of two different green contexts fork
// onto two side streams; the second fork's work must still run after the
// first's, as it did when every fork shared one side stream (graph-monitor
// replay counters precede the START events of later collectives).
TEST_F(GraphSideStreamTest, ForksAcrossGreenContextsKeepForkOrder) {
  CUgreenCtx green_a = nullptr;
  CUgreenCtx green_b = nullptr;
  ASSERT_NO_FATAL_FAILURE(makeGreenContext(&green_a));
  ASSERT_NO_FATAL_FAILURE(makeGreenContext(&green_b));
  CUstream main_a = nullptr;
  CUstream main_b = nullptr;
  ASSERT_EQ(
      cuGreenCtxStreamCreate(&main_a, green_a, CU_STREAM_NON_BLOCKING, 0),
      CUDA_SUCCESS);
  ASSERT_EQ(
      cuGreenCtxStreamCreate(&main_b, green_b, CU_STREAM_NON_BLOCKING, 0),
      CUDA_SUCCESS);
  cudaEvent_t join = nullptr;
  ASSERT_EQ(
      cudaEventCreateWithFlags(&join, cudaEventDisableTiming), cudaSuccess);

  {
    GraphSideStream side;
    int* first = nullptr;
    int* second = nullptr;
    ASSERT_EQ(cudaMalloc(&first, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&second, sizeof(int)), cudaSuccess);

    ASSERT_EQ(
        cudaStreamBeginCapture(main_a, cudaStreamCaptureModeGlobal),
        cudaSuccess);
    ASSERT_EQ(cudaEventRecord(join, main_a), cudaSuccess);
    ASSERT_EQ(cudaStreamWaitEvent(main_b, join), cudaSuccess);
    ASSERT_EQ(
        side.fork_from(
            main_a,
            [&](cudaStream_t s) {
              (void)cudaMemsetAsync(first, 1, sizeof(int), s);
            }),
        cudaSuccess);
    ASSERT_EQ(
        side.fork_from(
            main_b,
            [&](cudaStream_t s) {
              (void)cudaMemsetAsync(second, 1, sizeof(int), s);
            }),
        cudaSuccess);
    ASSERT_EQ(cudaEventRecord(join, main_b), cudaSuccess);
    ASSERT_EQ(cudaStreamWaitEvent(main_a, join), cudaSuccess);
    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(main_a, &graph), cudaSuccess);

    auto memset_to = [&](const int* dst) -> cudaGraphNode_t {
      for (cudaGraphNode_t node : getNodes(graph)) {
        if (nodeType(node) != cudaGraphNodeTypeMemset) {
          continue;
        }
        cudaMemsetParams params{};
        EXPECT_EQ(cudaGraphMemsetNodeGetParams(node, &params), cudaSuccess);
        if (params.dst == dst) {
          return node;
        }
      }
      return nullptr;
    };
    cudaGraphNode_t first_node = memset_to(first);
    cudaGraphNode_t second_node = memset_to(second);
    ASSERT_NE(first_node, nullptr);
    ASSERT_NE(second_node, nullptr);
    std::set<cudaGraphNode_t> seen;
    std::vector<cudaGraphNode_t> frontier{second_node};
    while (!frontier.empty()) {
      cudaGraphNode_t node = frontier.back();
      frontier.pop_back();
      for (cudaGraphNode_t pred : getPreds(node)) {
        if (seen.insert(pred).second) {
          frontier.push_back(pred);
        }
      }
    }
    EXPECT_TRUE(seen.count(first_node))
        << "the second fork's work must depend on the first fork's";

    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(exec, main_a), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(main_a), cudaSuccess);

    EXPECT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    EXPECT_EQ(cudaFree(first), cudaSuccess);
    EXPECT_EQ(cudaFree(second), cudaSuccess);
  }

  EXPECT_EQ(cudaEventDestroy(join), cudaSuccess);
  EXPECT_EQ(cuStreamDestroy(main_a), CUDA_SUCCESS);
  EXPECT_EQ(cuStreamDestroy(main_b), CUDA_SUCCESS);
  EXPECT_EQ(cuGreenCtxDestroy(green_a), CUDA_SUCCESS);
  EXPECT_EQ(cuGreenCtxDestroy(green_b), CUDA_SUCCESS);
}

// One helper outlives a green context. A later green context, even one the
// driver places at the retired context's address, must get side resources of
// its own rather than the retired context's.
TEST_F(GraphSideStreamTest, ForkFromRecreatedGreenContext) {
  GraphSideStream side;
  int* dev_counter = nullptr;
  ASSERT_EQ(cudaMalloc(&dev_counter, sizeof(int)), cudaSuccess);
  auto capture_one_fork = [&](CUgreenCtx green_ctx) {
    CUstream main = nullptr;
    ASSERT_EQ(
        cuGreenCtxStreamCreate(&main, green_ctx, CU_STREAM_NON_BLOCKING, 0),
        CUDA_SUCCESS);
    ASSERT_EQ(
        cudaStreamBeginCapture(main, cudaStreamCaptureModeGlobal), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(dev_counter, 0, sizeof(int), main), cudaSuccess);
    EXPECT_EQ(
        side.fork_from(
            main,
            [&](cudaStream_t s) {
              (void)cudaMemsetAsync(dev_counter, 1, sizeof(int), s);
            }),
        cudaSuccess);
    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(main, &graph), cudaSuccess);
    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(exec, main), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(main), cudaSuccess);
    EXPECT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    EXPECT_EQ(cuStreamDestroy(main), CUDA_SUCCESS);
  };

  CUgreenCtx retired = nullptr;
  ASSERT_NO_FATAL_FAILURE(makeGreenContext(&retired));
  ASSERT_NO_FATAL_FAILURE(capture_one_fork(retired));
  const CUgreenCtx retired_address = retired;
  ASSERT_EQ(cuGreenCtxDestroy(retired), CUDA_SUCCESS);

  // Prefer a context at the retired address, the case a pointer-keyed cache
  // gets wrong; keep the last one created either way.
  CUgreenCtx recreated = nullptr;
  for (int i = 0; i < 16; ++i) {
    ASSERT_NO_FATAL_FAILURE(makeGreenContext(&recreated));
    if (recreated == retired_address) {
      break;
    }
    ASSERT_EQ(cuGreenCtxDestroy(recreated), CUDA_SUCCESS);
    recreated = nullptr;
  }
  if (recreated == nullptr) {
    ASSERT_NO_FATAL_FAILURE(makeGreenContext(&recreated));
  }
  ASSERT_NO_FATAL_FAILURE(capture_one_fork(recreated));

  EXPECT_EQ(cuGreenCtxDestroy(recreated), CUDA_SUCCESS);
  EXPECT_EQ(cudaFree(dev_counter), cudaSuccess);
}
#endif

} // namespace
