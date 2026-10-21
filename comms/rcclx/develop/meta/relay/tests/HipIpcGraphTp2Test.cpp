/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <hip/hip_runtime.h>
#include <poll.h>
#include <sys/socket.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include "comms/rcclx/develop/meta/relay/hipipc/hipipc_relay.h"
#include "comms/rcclx/develop/meta/relay/registered_allreduce_relay_args.h"

std::pair<size_t, size_t> hipipcTestEventCounts();
namespace {
using namespace rcclx::relay::hipipc;
constexpr size_t kBytes = 2ULL << 20;
void expect(bool value, const char* message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}
void check(hipError_t error) {
  if (error != hipSuccess) {
    throw std::runtime_error(hipGetErrorString(error));
  }
}
template <class F>
void rejects(F&& function) {
  bool failed = false;
  try {
    function();
  } catch (const std::exception&) {
    failed = true;
  }
  expect(failed, "graph API accepted an invalid operation");
}
template <class T>
T exchange(int socket, const T& mine) {
  expect(
      send(socket, &mine, sizeof(mine), MSG_NOSIGNAL) == sizeof(mine),
      "graph control send");
  pollfd descriptor{socket, POLLIN, 0};
  expect(
      poll(&descriptor, 1, 45000) == 1 && (descriptor.revents & POLLIN),
      "graph control deadline");
  T peer{};
  expect(
      recv(socket, &peer, sizeof(peer), MSG_TRUNC) == sizeof(peer),
      "graph control length");
  return peer;
}
uint16_t bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, 4);
  return (bits + 0x7fffu + ((bits >> 16) & 1)) >> 16;
}
void fill(void* pointer, float value) {
  check(hipMemsetD16(
      static_cast<hipDeviceptr_t>(pointer), bf16(value), kBytes / 2));
  check(hipStreamSynchronize(nullptr));
}
void output(void* pointer, float value) {
  std::vector<uint16_t> result(kBytes / 2);
  check(hipMemcpy(result.data(), pointer, kBytes, hipMemcpyDeviceToHost));
  expect(
      std::all_of(
          result.begin(),
          result.end(),
          [value](uint16_t actual) { return actual == bf16(value); }),
      "graph raw BF16 output mismatch");
}
Owner allocate(size_t bytes, bool pool, hipStream_t stream) {
  void* pointer = nullptr;
  check(
      pool ? hipMallocAsync(&pointer, bytes, stream)
           : hipMalloc(&pointer, bytes));
  check(hipStreamSynchronize(stream));
  return {pointer, [pool, stream](void* p) {
            if (pool) {
              (void)hipFreeAsync(p, stream);
            } else {
              (void)hipFree(p);
            }
          }};
}
struct Buffers {
  std::vector<Owner> storage;
};
uint32_t calls(const WorkspaceDescriptor& peer) {
  hipIpcMemHandle_t handle{};
  std::memcpy(&handle, peer.flags.data(), sizeof(handle));
  void* base = nullptr;
  check(hipIpcOpenMemHandle(&base, handle, hipIpcMemLazyEnablePeerAccess));
  rcclx::relay::RegisteredRelayFlags flags{};
  check(hipMemcpy(
      &flags,
      static_cast<unsigned char*>(base) + 256,
      sizeof(flags),
      hipMemcpyDeviceToHost));
  check(hipIpcCloseMemHandle(base));
  return flags.calls.value;
}
std::vector<hipGraphNode_t> nodes(hipGraph_t graph) {
  size_t count = 0;
  check(hipGraphGetNodes(graph, nullptr, &count));
  std::vector<hipGraphNode_t> result(count);
  check(hipGraphGetNodes(graph, result.data(), &count));
  return result;
}
rcclx::relay::RegisteredRelayArgs arguments(const hipKernelNodeParams& params) {
  rcclx::relay::RegisteredRelayArgs result{};
  if (params.kernelParams) {
    std::memcpy(&result, params.kernelParams[0], sizeof(result));
  } else {
    expect(
        params.extra && params.extra[0] == HIP_LAUNCH_PARAM_BUFFER_POINTER &&
            params.extra[2] == HIP_LAUNCH_PARAM_BUFFER_SIZE,
        "unknown HIP canonical argument representation");
    expect(
        *static_cast<size_t*>(params.extra[3]) == sizeof(result),
        "canonical argument size");
    std::memcpy(&result, params.extra[1], sizeof(result));
  }
  return result;
}
void retire(
    Workspace& workspace,
    int socket,
    const std::vector<RegistrationId>& ids) {
  workspace.drain();
  exchange(socket, uint64_t{1});
  workspace.closeImports(ids);
  exchange(socket, uint64_t{2});
  workspace.retire(ids);
}
void close(Workspace& workspace, int socket) {
  workspace.drain();
  exchange(socket, uint64_t{1});
  workspace.closeImports();
  exchange(socket, uint64_t{2});
  workspace.close();
}
} // namespace

void runHipIpcGraphCases(
    rcclx::relay::hipipc::Workspace& workspace,
    int rank,
    int socket,
    hipStream_t execution,
    const std::string& mode) {
  using namespace rcclx::relay::hipipc;
  const auto peerWorkspace = exchange(
      socket, decodeRecord<WorkspaceDescriptor>(workspace.localDescriptor()));
  const uint32_t epoch = calls(peerWorkspace);
  hipStream_t capture = nullptr;
  hipEvent_t order = nullptr;
  check(hipStreamCreateWithFlags(&capture, hipStreamNonBlocking));
  check(hipEventCreateWithFlags(&order, hipEventDisableTiming));
  auto releases = std::make_shared<unsigned>(0);
  auto group = std::shared_ptr<Buffers>(
      new Buffers, [releases, &workspace](Buffers* buffers) {
        try {
          (void)workspace.localDescriptor();
        } catch (const std::exception&) {
          // Whole close has already transitioned the workspace to Closed.
        }
        expect(!workspace.poisoned(), "captured owner freed while poisoned");
        ++*releases;
        delete buffers;
      });
  std::weak_ptr<Buffers> weak = group;
  const bool pool =
      mode == "graph_staged" || (mode == "graph_mixed" && rank == 1);
  group->storage.push_back(allocate(2 * kBytes + 8192, pool, execution));
  auto* first = static_cast<unsigned char*>(group->storage[0].get()) + 4112;
  auto* second = first + kBytes;
  std::array<void*, 3> inputs{first, second, nullptr}, outputs{};
  for (auto& out : outputs) {
    group->storage.push_back(allocate(kBytes, false, execution));
    out = group->storage.back().get();
  }
  inputs[2] = outputs[0];
  fill(first, rank + 1);
  fill(second, 2 * (rank + 1));
  check(hipEventRecord(order, execution));
  check(hipStreamWaitEvent(capture, order, 0));
  const auto eventCounts = hipipcTestEventCounts();
  check(hipStreamBeginCapture(capture, hipStreamCaptureModeRelaxed));
  std::vector<CapturedOp> ops;
  if (mode == "graph_capture_failure" || mode == "graph_stream") {
    rejects([&] {
      workspace.captureAllReduce(
          first,
          mode == "graph_stream" ? outputs[0] : first,
          kBytes,
          group,
          mode == "graph_stream" ? execution : capture);
    });
    hipGraph_t failed = nullptr;
    check(hipStreamEndCapture(capture, &failed));
    const auto after = hipipcTestEventCounts();
    expect(after == eventCounts, "native capture created or recorded an event");
    group.reset();
    expect(
        workspace.poisoned() && !weak.expired() && *releases == 0,
        "capture failure dropped input/output ownership");
    const auto noWork = hipipcTestEventCounts();
    rejects([&] { workspace.close(); });
    expect(
        hipipcTestEventCounts() == noWork, "poisoned close submitted GPU work");
    exchange(socket, uint64_t{1});
    return;
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    ops.push_back(workspace.captureAllReduce(
        inputs[i], outputs[i], kBytes, group, capture));
  }
  hipGraph_t graph = nullptr;
  check(hipStreamEndCapture(capture, &graph));
  expect(
      hipipcTestEventCounts() == eventCounts, "native CAP event create/record");
  check(hipEventRecord(order, capture));
  check(hipStreamWaitEvent(execution, order, 0));
  check(hipStreamSynchronize(execution));
  std::vector<CapturedDescriptor> local, peer;
  for (auto op : ops) {
    local.push_back(decodeRecord<CapturedDescriptor>(workspace.describe(op)));
    peer.push_back(exchange(socket, local.back()));
    if (mode == "graph_peer" && peer.size() == 2) {
      rejects([&] { workspace.bindPeer(op, encodeRecord(peer.front())); });
      group.reset();
      expect(
          workspace.poisoned() && !weak.expired(),
          "stale binding released captured owners");
      return;
    }
    workspace.bindPeer(op, encodeRecord(peer.back()));
  }
  expect(
      local[0].role == (pool ? CaptureRole::Staged : CaptureRole::Direct),
      "wrong typed staged/direct allocation role");
  if (!pool) {
    expect(
        local[0].input.offset == 4112 &&
            local[0].input.base == local[1].input.base &&
            local[0].input.allocationGeneration ==
                local[1].input.allocationGeneration,
        "captured allocation/view alias truth");
  }
  rejects([&] { workspace.finalizeCapture(graph, {}); });
  expect(!workspace.poisoned(), "empty finalize poisoned a healthy workspace");
  if (mode == "graph_finalize" || mode == "graph_finalize_partial" ||
      mode == "graph_finalize_order" || mode == "graph_finalize_duplicate") {
    hipGraph_t clone = nullptr;
    auto invalid = ops;
    if (mode == "graph_finalize") {
      check(hipGraphClone(&clone, graph));
    } else if (mode == "graph_finalize_partial") {
      invalid.pop_back();
    } else if (mode == "graph_finalize_duplicate") {
      invalid[1] = invalid[0];
    } else {
      std::swap(invalid[0], invalid[1]);
    }
    const auto before = nodes(graph);
    rejects([&] { workspace.finalizeCapture(clone ? clone : graph, invalid); });
    expect(nodes(graph) == before, "invalid finalize changed graph nodes");
    group.reset();
    expect(
        workspace.poisoned() && !weak.expired(),
        "invalid finalization released captured owners");
    return;
  }
  workspace.finalizeCapture(graph, ops);
  expect(
      calls(peerWorkspace) == epoch,
      "capture/finalize reset or advanced live flags");
  size_t copies = 0, kernels = 0;
  for (auto node : nodes(graph)) {
    hipGraphNodeType type{};
    check(hipGraphNodeGetType(node, &type));
    copies += type == hipGraphNodeTypeMemcpy;
    if (type == hipGraphNodeTypeKernel) {
      ++kernels;
      hipKernelNodeParams params{};
      check(hipGraphKernelNodeGetParams(node, &params));
      const auto args = arguments(params);
      auto found = std::find(outputs.begin(), outputs.end(), args.output);
      expect(found != outputs.end(), "unknown canonical graph output");
      const size_t i = found - outputs.begin();
      const bool direct = local[i].role == CaptureRole::Direct &&
          peer[i].role == CaptureRole::Direct;
      expect(
          !direct || args.mine == inputs[i],
          "direct graph still reads shared staging");
    }
  }
  size_t expectedCopies = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    expectedCopies += local[i].role == CaptureRole::Staged ||
        peer[i].role == CaptureRole::Staged;
  }
  expect(
      copies == expectedCopies && kernels == 3,
      "unnecessary direct copies or missing canonical nodes");
  std::vector<RegistrationId> ids;
  for (auto op : ops) {
    ids.push_back(op.id);
  }
  const auto stale = ops.front();
  ops.clear();
  group.reset();
  expect(!weak.expired(), "GC/token drops freed captured input or output");
  if (mode == "graph_bare") {
    check(hipGraphDestroy(graph));
    retire(workspace, socket, ids);
    expect(
        weak.expired() && *releases == 1,
        "bare graph retirement leaked owners");
    close(workspace, socket);
    check(hipEventDestroy(order));
    check(hipStreamDestroy(capture));
    return;
  }
  hipGraphExec_t executable = nullptr;
  check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  for (int replay = 1; replay <= 4; ++replay) {
    fill(first, replay * (rank + 1));
    fill(second, 2 * replay * (rank + 1));
    check(hipGraphLaunch(executable, execution));
    workspace.drain();
    output(outputs[0], 3 * replay);
    output(outputs[1], 6 * replay);
    output(outputs[2], 6 * replay);
    expect(
        calls(peerWorkspace) == epoch + 3 * replay, "graph replay flag epochs");
  }
  // Queue another replay, then disable replay and drain before destroying HIP
  // handles. Owners stay native-rooted despite all public tokens being dropped.
  check(hipGraphLaunch(executable, execution));
  expect(
      !weak.expired() && *releases == 0,
      "queued graph lost its storage owners");
  workspace.drain();
  check(hipGraphExecDestroy(executable));
  check(hipGraphDestroy(graph));
  if (mode == "graph_close") {
    close(workspace, socket);
    expect(
        weak.expired() && *releases == 1,
        "whole close leaked captured output owners");
    check(hipEventDestroy(order));
    check(hipStreamDestroy(capture));
    return;
  }
  if (mode == "graph_stale") {
    retire(workspace, socket, {ids.front()});
    expect(!weak.expired(), "partial captured alias retirement freed group");
    rejects([&] { workspace.describe(stale); });
    expect(
        workspace.poisoned() && !weak.expired(),
        "stale op did not retain whole state");
    return;
  }
  retire(workspace, socket, ids);
  expect(
      weak.expired() && *releases == 1,
      "batched captured retirement failed outside-lock owner release");
  auto input = allocate(kBytes, false, execution),
       out = allocate(kBytes, false, execution);
  fill(input.get(), rank + 1);
  workspace.allReduceStaged(input.get(), out.get(), kBytes, execution);
  workspace.drain();
  output(out.get(), 3);
  expect(
      calls(peerWorkspace) == epoch + 16,
      "eager after graph retirement flag epoch");
  close(workspace, socket);
  check(hipEventDestroy(order));
  check(hipStreamDestroy(capture));
}
