// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/GraphCaptureSideStream.h"

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

} // namespace
