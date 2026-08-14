// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Host-side unit tests for the side-stream GPE host-node path
// (NCCL_CTRAN_GPE_HOST_NODE_SIDE_STREAM). These exercise the spine bookkeeping
// that ctran::gpe::HostNodeSpine owns -- per-graph keying, lazy
// creation, reclamation of finished captures -- plus the recorded graph
// topology the knob is for, using a bare capture with stand-in nodes so no
// comm, window or IB backend is required.
//
// The end-to-end shape on a real ctwin AllGatherP (HOST out-degree 2 and N-1
// HOST->HOST edges with the knob on, versus out-degree 1 and 0 such edges with
// it off) is asserted by
// CtranAllgatherCtwinHostSpineTest.HostNodeWiring in
// comms/ctran/tests/CtranDistAllgatherCtwinTests.cc, which needs 8 ranks and a
// vnode topology; this TU covers the mechanism on a single device.

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "comms/ctran/utils/CudaGraphUtils.h"

namespace {

void hostFn(void* /*unused*/) {}

// Mirrors what CtranGpe::Impl does per captured submit when the knob is on:
// record the HOST node on the spine stream, publish its tip, then join the tip
// into the user stream BEFORE the collective's first node is issued.
void addSpineHostNode(
    cudaStream_t userStream,
    cudaStream_t spineStream,
    cudaEvent_t tip) {
  ASSERT_EQ(cudaLaunchHostFunc(spineStream, hostFn, nullptr), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(tip, spineStream), cudaSuccess);
  ASSERT_EQ(cudaStreamWaitEvent(userStream, tip, 0), cudaSuccess);
}

struct Census {
  size_t nodes{0};
  size_t edges{0};
  size_t hosts{0};
  int maxHostOutDeg{0};
  int hostToHost{0};
};

Census censusOf(cudaGraph_t graph) {
  Census c{};
  EXPECT_EQ(cudaGraphGetNodes(graph, nullptr, &c.nodes), cudaSuccess);
  std::vector<cudaGraphNode_t> nodes(c.nodes);
  EXPECT_EQ(cudaGraphGetNodes(graph, nodes.data(), &c.nodes), cudaSuccess);

  EXPECT_EQ(cudaGraphGetEdges(graph, nullptr, nullptr, &c.edges), cudaSuccess);
  std::vector<cudaGraphNode_t> from(c.edges), to(c.edges);
  EXPECT_EQ(
      cudaGraphGetEdges(graph, from.data(), to.data(), &c.edges), cudaSuccess);

  std::unordered_set<cudaGraphNode_t> hostNodes;
  for (cudaGraphNode_t n : nodes) {
    cudaGraphNodeType t{};
    EXPECT_EQ(cudaGraphNodeGetType(n, &t), cudaSuccess);
    if (t == cudaGraphNodeTypeHost) {
      hostNodes.insert(n);
    }
  }
  c.hosts = hostNodes.size();

  std::unordered_map<cudaGraphNode_t, int> outDeg;
  for (size_t e = 0; e < c.edges; ++e) {
    outDeg[from[e]]++;
    if (hostNodes.count(from[e]) != 0u && hostNodes.count(to[e]) != 0u) {
      c.hostToHost++;
    }
  }
  for (cudaGraphNode_t h : hostNodes) {
    c.maxHostOutDeg = std::max(c.maxHostOutDeg, outDeg[h]);
  }
  return c;
}

class GpeHostNodeSideStreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "no CUDA device";
    }
  }
};

// The inline shape (knob off) leaves each HOST node a pass-through link: no
// HOST depends on another, so nothing orders the collectives relative to each
// other.
TEST_F(GpeHostNodeSideStreamTest, InlineHostNodesAreIndependent) {
  constexpr int kNumCollectives = 4;
  cudaStream_t user{};
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&user, cudaStreamNonBlocking), cudaSuccess);
  int* buf{};
  ASSERT_EQ(cudaMalloc(&buf, sizeof(int)), cudaSuccess);

  cudaGraph_t graph{};
  ASSERT_EQ(
      cudaStreamBeginCapture(user, cudaStreamCaptureModeGlobal), cudaSuccess);
  for (int i = 0; i < kNumCollectives; ++i) {
    ASSERT_EQ(cudaLaunchHostFunc(user, hostFn, nullptr), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(buf, i, sizeof(int), user), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamEndCapture(user, &graph), cudaSuccess);

  const Census c = censusOf(graph);
  EXPECT_EQ(c.hosts, static_cast<size_t>(kNumCollectives));
  EXPECT_EQ(c.hostToHost, 0);
  EXPECT_EQ(c.maxHostOutDeg, 1);

  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  ASSERT_EQ(cudaFree(buf), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(user), cudaSuccess);
}

// The spine shape (knob on): HOST[i]'s only predecessor is HOST[i-1], and it
// also parents the collective's first node, so out-degree reaches 2 and there
// are exactly N-1 HOST->HOST edges. That single serializing edge per collective
// is the whole mechanism -- it is the shape NCCL's captured host callbacks
// already have.
TEST_F(GpeHostNodeSideStreamTest, SpineChainsHostNodesAcrossCollectives) {
  constexpr int kNumCollectives = 4;
  cudaStream_t user{}, spine{};
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&user, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&spine, cudaStreamNonBlocking), cudaSuccess);
  cudaEvent_t tip{};
  ASSERT_EQ(
      cudaEventCreateWithFlags(&tip, cudaEventDisableTiming), cudaSuccess);
  int* buf{};
  ASSERT_EQ(cudaMalloc(&buf, sizeof(int)), cudaSuccess);

  cudaGraph_t graph{};
  ASSERT_EQ(
      cudaStreamBeginCapture(user, cudaStreamCaptureModeGlobal), cudaSuccess);
  // Join the spine into the graph and CLEAR the dependency set it inherited,
  // so HOST[0] does not depend on anything already on the user stream.
  ASSERT_EQ(cudaEventRecord(tip, user), cudaSuccess);
  ASSERT_EQ(cudaStreamWaitEvent(spine, tip, 0), cudaSuccess);
#if CUDART_VERSION >= 13000
  ASSERT_EQ(
      cudaStreamUpdateCaptureDependencies(
          spine, nullptr, nullptr, 0, cudaStreamSetCaptureDependencies),
      cudaSuccess);
#else
  ASSERT_EQ(
      cudaStreamUpdateCaptureDependencies(
          spine, nullptr, 0, cudaStreamSetCaptureDependencies),
      cudaSuccess);
#endif
  for (int i = 0; i < kNumCollectives; ++i) {
    addSpineHostNode(user, spine, tip);
    ASSERT_EQ(cudaMemsetAsync(buf, i, sizeof(int), user), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamEndCapture(user, &graph), cudaSuccess);

  const Census c = censusOf(graph);
  EXPECT_EQ(c.hosts, static_cast<size_t>(kNumCollectives));
  // Capture absorbs the tip's cudaEventRecord into a plain dependency edge
  // rather than emitting an EVENT_RECORD node, so the spine is HOST->HOST.
  EXPECT_EQ(c.hostToHost, kNumCollectives - 1);
  EXPECT_EQ(c.maxHostOutDeg, 2);

  // The graph must still instantiate and replay.
  cudaGraphExec_t exec{};
  // The 5-arg (legacy) form is the portable one: HIP only provides that
  // signature.
  ASSERT_EQ(
      cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(exec, user), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(user), cudaSuccess);

  ASSERT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(tip), cudaSuccess);
  ASSERT_EQ(cudaFree(buf), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(spine), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(user), cudaSuccess);
}

// Two graphs captured concurrently must not share a spine: each keeps its own
// HOST chain, so neither graph's host nodes depend on the other's. This is why
// the implementation keys spines by capture id (as NCCL keys its capture
// streams by graphId) instead of holding one spine per comm.
TEST_F(GpeHostNodeSideStreamTest, ConcurrentCapturesGetIndependentSpines) {
  constexpr int kNumCollectives = 3;
  cudaStream_t userA{}, userB{}, spineA{}, spineB{};
  for (cudaStream_t* s : {&userA, &userB, &spineA, &spineB}) {
    ASSERT_EQ(cudaStreamCreateWithFlags(s, cudaStreamNonBlocking), cudaSuccess);
  }
  cudaEvent_t tipA{}, tipB{};
  ASSERT_EQ(
      cudaEventCreateWithFlags(&tipA, cudaEventDisableTiming), cudaSuccess);
  ASSERT_EQ(
      cudaEventCreateWithFlags(&tipB, cudaEventDisableTiming), cudaSuccess);
  int* buf{};
  ASSERT_EQ(cudaMalloc(&buf, sizeof(int)), cudaSuccess);

  ASSERT_EQ(
      cudaStreamBeginCapture(userA, cudaStreamCaptureModeGlobal), cudaSuccess);
  ASSERT_EQ(
      cudaStreamBeginCapture(userB, cudaStreamCaptureModeGlobal), cudaSuccess);

  // Distinct capture ids => distinct spines.
  ctran::utils::cudagraph::StreamCaptureInfo infoA{}, infoB{};
  ASSERT_EQ(
      ctran::utils::cudagraph::getStreamCaptureInfo(userA, infoA), cudaSuccess);
  ASSERT_EQ(
      ctran::utils::cudagraph::getStreamCaptureInfo(userB, infoB), cudaSuccess);
  EXPECT_EQ(infoA.status, cudaStreamCaptureStatusActive);
  EXPECT_EQ(infoB.status, cudaStreamCaptureStatusActive);
  EXPECT_NE(infoA.id, infoB.id);

  for (auto [user, spine, tip] :
       {std::tuple{userA, spineA, tipA}, std::tuple{userB, spineB, tipB}}) {
    ASSERT_EQ(cudaEventRecord(tip, user), cudaSuccess);
    ASSERT_EQ(cudaStreamWaitEvent(spine, tip, 0), cudaSuccess);
#if CUDART_VERSION >= 13000
    ASSERT_EQ(
        cudaStreamUpdateCaptureDependencies(
            spine, nullptr, nullptr, 0, cudaStreamSetCaptureDependencies),
        cudaSuccess);
#else
    ASSERT_EQ(
        cudaStreamUpdateCaptureDependencies(
            spine, nullptr, 0, cudaStreamSetCaptureDependencies),
        cudaSuccess);
#endif
  }
  for (int i = 0; i < kNumCollectives; ++i) {
    addSpineHostNode(userA, spineA, tipA);
    ASSERT_EQ(cudaMemsetAsync(buf, i, sizeof(int), userA), cudaSuccess);
    addSpineHostNode(userB, spineB, tipB);
    ASSERT_EQ(cudaMemsetAsync(buf, i, sizeof(int), userB), cudaSuccess);
  }

  cudaGraph_t graphA{}, graphB{};
  ASSERT_EQ(cudaStreamEndCapture(userA, &graphA), cudaSuccess);
  ASSERT_EQ(cudaStreamEndCapture(userB, &graphB), cudaSuccess);

  // Each graph carries its own complete spine; nothing leaked across.
  for (cudaGraph_t g : {graphA, graphB}) {
    const Census c = censusOf(g);
    EXPECT_EQ(c.hosts, static_cast<size_t>(kNumCollectives));
    EXPECT_EQ(c.hostToHost, kNumCollectives - 1);
    EXPECT_EQ(c.maxHostOutDeg, 2);
  }

  ASSERT_EQ(cudaGraphDestroy(graphA), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graphB), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(tipA), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(tipB), cudaSuccess);
  ASSERT_EQ(cudaFree(buf), cudaSuccess);
  for (cudaStream_t s : {userA, userB, spineA, spineB}) {
    ASSERT_EQ(cudaStreamDestroy(s), cudaSuccess);
  }
}

// A spine stream whose capture has ended is reclaimable: cudaStreamIsCapturing
// reports it inactive, which is the signal HostNodeSpine::getOrCreate
// uses to destroy the entry instead of leaking one stream + event per captured
// graph.
TEST_F(GpeHostNodeSideStreamTest, FinishedCaptureIsDetectedAsReclaimable) {
  cudaStream_t user{}, spine{};
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&user, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&spine, cudaStreamNonBlocking), cudaSuccess);
  cudaEvent_t tip{};
  ASSERT_EQ(
      cudaEventCreateWithFlags(&tip, cudaEventDisableTiming), cudaSuccess);

  cudaGraph_t graph{};
  ASSERT_EQ(
      cudaStreamBeginCapture(user, cudaStreamCaptureModeGlobal), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(tip, user), cudaSuccess);
  ASSERT_EQ(cudaStreamWaitEvent(spine, tip, 0), cudaSuccess);
  addSpineHostNode(user, spine, tip);

  cudaStreamCaptureStatus status{};
  ASSERT_EQ(cudaStreamIsCapturing(spine, &status), cudaSuccess);
  EXPECT_EQ(status, cudaStreamCaptureStatusActive) << "active during capture";

  ASSERT_EQ(cudaStreamEndCapture(user, &graph), cudaSuccess);

  ASSERT_EQ(cudaStreamIsCapturing(spine, &status), cudaSuccess);
  EXPECT_EQ(status, cudaStreamCaptureStatusNone)
      << "inactive after EndCapture -> entry is reclaimable";

  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(tip), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(spine), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(user), cudaSuccess);
}

} // namespace
