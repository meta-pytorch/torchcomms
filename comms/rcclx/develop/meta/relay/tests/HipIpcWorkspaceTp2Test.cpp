/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// HIP-only workers exec before initializing HIP; no Torch/NCCL dependency.
#include "comms/rcclx/develop/meta/relay/hipipc/hipipc_relay.h"
#include "comms/rcclx/develop/meta/relay/registered_allreduce_relay_args.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef HIPIPC_TEST_NATIVE_DIR
#define HIPIPC_TEST_NATIVE_DIR ""
#endif

// Test-only runtime interposition: inject a launch failure after bootstrap.
static bool failLaunch = false;
static std::atomic<bool> holdCompletion{false}, completionQueried{false};
extern "C" hipError_t hipEventQuery(hipEvent_t event) {
  if (holdCompletion.load()) {
    completionQueried.store(true);
    return hipErrorNotReady;
  }
  static auto real = reinterpret_cast<decltype(&hipEventQuery)>(
      dlsym(RTLD_NEXT, "hipEventQuery"));
  return real(event);
}
extern "C" hipError_t hipLaunchKernel(
    const void* function,
    dim3 grid,
    dim3 block,
    void** args,
    size_t shared,
    hipStream_t stream) {
  static auto realLaunch = reinterpret_cast<decltype(&hipLaunchKernel)>(
      dlsym(RTLD_NEXT, "hipLaunchKernel"));
  // The canonical launcher reads hipGetLastError: let HIP set its real TLS
  // error by rejecting an invalid block rather than faking a return code.
  return realLaunch ? realLaunch(
                          function,
                          grid,
                          failLaunch ? dim3{0, 0, 0} : block,
                          args,
                          shared,
                          stream)
                    : hipErrorSharedObjectSymbolNotFound;
}

namespace {
using namespace rcclx::relay::hipipc;
constexpr size_t kBytes = 2ULL << 20;
void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}
void hipCheck(hipError_t status) {
  if (status != hipSuccess) {
    throw std::runtime_error(hipGetErrorString(status));
  }
}
template <class F>
void rejects(F&& function) {
  bool threw = false;
  try {
    function();
  } catch (const std::exception&) {
    threw = true;
  }
  expect(threw, "native API accepted invalid operation");
}
template <class T>
T exchange(int socket, const T& value) {
  expect(
      send(socket, &value, sizeof(value), MSG_NOSIGNAL) == sizeof(value),
      "control send failed");
  pollfd descriptor{socket, POLLIN, 0};
  expect(
      poll(&descriptor, 1, 45000) == 1 && (descriptor.revents & POLLIN),
      "control receive deadline exceeded");
  T peer{};
  expect(
      recv(socket, &peer, sizeof(peer), MSG_TRUNC) == sizeof(peer),
      "control record length mismatch or peer died");
  return peer;
}
Owner allocation(
    size_t bytes,
    std::shared_ptr<unsigned> releases = {},
    Workspace* workspace = nullptr) {
  int device = -1;
  hipCheck(hipGetDevice(&device));
  void* pointer = nullptr;
  hipCheck(hipMalloc(&pointer, bytes));
  return {pointer, [device, releases, workspace](void* p) {
            // Reentry deadlocks if native drops an owner while holding its
            // state lock.
            if (workspace) {
              try {
                (void)workspace->localDescriptor();
              } catch (const std::exception&) {
                // Whole close has already transitioned the workspace to Closed.
              }
              expect(
                  !workspace->poisoned(),
                  "healthy retirement unexpectedly poisoned");
            }
            if (releases) {
              ++*releases;
            }
            int previous = -1;
            (void)hipGetDevice(&previous);
            (void)hipSetDevice(device);
            (void)hipFree(p);
            (void)hipSetDevice(previous);
          }};
}
uint16_t bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1);
  return bits >> 16;
}
void fill(void* pointer, uint16_t value, size_t bytes = kBytes) {
  hipCheck(
      hipMemsetD16(static_cast<hipDeviceptr_t>(pointer), value, bytes / 2));
  // Test producers use the default stream; EXEC is explicitly nonblocking.
  hipCheck(hipStreamSynchronize(nullptr));
}
void checkOutput(void* pointer, uint16_t value, size_t bytes = kBytes) {
  std::vector<uint16_t> host(bytes / 2);
  hipCheck(hipMemcpy(host.data(), pointer, bytes, hipMemcpyDeviceToHost));
  const auto mismatch =
      std::find_if(host.begin(), host.end(), [value](uint16_t actual) {
        return actual != value;
      });
  if (mismatch != host.end()) {
    std::fprintf(
        stderr,
        "HIPIPC mismatch element %zu: expected %04x actual %04x\n",
        static_cast<size_t>(mismatch - host.begin()),
        value,
        *mismatch);
    throw std::runtime_error("TP2 output differs in raw BF16 bits");
  }
}
void bootstrap(Workspace& workspace, int socket, hipStream_t stream, int rank) {
  const auto local =
      decodeRecord<WorkspaceDescriptor>(workspace.localDescriptor());
  workspace.connect(encodeRecord(exchange(socket, local)));
  // The fast rank may enter warmup while the slow rank still reads its
  // canaries.
  if (rank == 1) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  workspace.probe(stream);
  exchange(socket, uint64_t{1});
}
void bind(Workspace& workspace, RegistrationId id, int socket) {
  const auto local =
      decodeRecord<RegistrationDescriptor>(workspace.describe(id));
  workspace.bindPeer(id, encodeRecord(exchange(socket, local)));
}
void retire(
    Workspace& workspace,
    const std::vector<RegistrationId>& ids,
    int socket) {
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
  workspace.close();
}
std::vector<unsigned char> flagSnapshot(const WorkspaceDescriptor& peer) {
  hipIpcMemHandle_t handle{};
  std::memcpy(&handle, peer.flags.data(), sizeof(handle));
  void* base = nullptr;
  hipCheck(hipIpcOpenMemHandle(&base, handle, hipIpcMemLazyEnablePeerAccess));
  std::vector<unsigned char> bytes(sizeof(rcclx::relay::RegisteredRelayFlags));
  hipCheck(hipMemcpy(
      bytes.data(),
      static_cast<unsigned char*>(base) + 256,
      bytes.size(),
      hipMemcpyDeviceToHost));
  hipCheck(hipIpcCloseMemHandle(base));
  return bytes;
}
void normalCases(
    Workspace& workspace,
    int rank,
    int socket,
    hipStream_t stream,
    int peerDevice) {
  const auto local =
      decodeRecord<WorkspaceDescriptor>(workspace.localDescriptor());
  const auto peer = exchange(socket, local);
  const auto flagsBefore = flagSnapshot(peer);
  auto releases = std::make_shared<unsigned>(0);
  auto owner = allocation(3 * kBytes + 8192, releases, &workspace);
  std::weak_ptr<void> weak = owner;
  auto* first = static_cast<unsigned char*>(owner.get()) + 4112;
  auto* second = first + kBytes;
  auto* out = second + kBytes;
  fill(first, bf16(rank + 1));
  fill(second, bf16(2 * (rank + 1)));
  const auto r0 = workspace.registerInput(first, kBytes, owner);
  const auto r1 = workspace.registerInput(second, kBytes, Owner(owner, second));
  auto d0 = decodeRecord<RegistrationDescriptor>(workspace.describe(r0));
  auto d1 = decodeRecord<RegistrationDescriptor>(workspace.describe(r1));
  expect(
      !(r0 == r1) && d0.offset == 4112 && d0.base == d1.base &&
          d0.allocationGeneration == d1.allocationGeneration,
      "alias identity/base/displacement lost");
  bind(workspace, r0, socket);
  bind(workspace, r1, socket);
  expect(
      flagSnapshot(peer) == flagsBefore, "registration reset live relay flags");
  workspace.allReduceDirect(r0, out, kBytes, stream);
  workspace.drain();
  checkOutput(out, bf16(3));
  checkOutput(first, bf16(rank + 1));
  retire(workspace, {r0}, socket);
  workspace.allReduceDirect(r1, out, kBytes, stream);
  workspace.drain();
  checkOutput(out, bf16(6));
  const auto repeated = workspace.registerInput(first, kBytes, owner);
  const auto repeatedDesc =
      decodeRecord<RegistrationDescriptor>(workspace.describe(repeated));
  expect(
      !(repeated == r0) &&
          repeatedDesc.allocationGeneration == d1.allocationGeneration,
      "repeat registration confused allocation and registration generation");
  bind(workspace, repeated, socket);
  owner.reset();
  expect(!weak.expired(), "registration failed to retain dropped owner");
  workspace.drain();
  exchange(socket, uint64_t{1});
  workspace.closeImports({r1, repeated});
  expect(
      *releases == 0 && !weak.expired(),
      "import closure released exported owner");
  exchange(socket, uint64_t{2});
  workspace.retire({repeated, r1});
  expect(*releases == 1 && weak.expired(), "final alias owner leaked");

  auto reused = allocation(kBytes + 8192);
  auto* view = static_cast<unsigned char*>(reused.get()) + 4112;
  uint64_t lastGeneration = 0;
  for (int i = 0; i < 4; ++i) {
    fill(view, bf16(rank + 1));
    const auto id = workspace.registerInput(view, kBytes, reused);
    const auto d = decodeRecord<RegistrationDescriptor>(workspace.describe(id));
    expect(
        d.allocationGeneration != lastGeneration,
        "retired base reused generation");
    lastGeneration = d.allocationGeneration;
    bind(workspace, id, socket);
    auto result = allocation(kBytes);
    workspace.allReduceDirect(id, result.get(), kBytes, stream);
    workspace.drain();
    checkOutput(result.get(), bf16(3));
    retire(workspace, {id}, socket);
  }
  reused.reset();
  auto fresh = allocation(kBytes);
  const auto freshId = workspace.registerInput(fresh.get(), kBytes, fresh);
  expect(
      decodeRecord<RegistrationDescriptor>(workspace.describe(freshId))
              .allocationGeneration != lastGeneration,
      "fresh allocation reused generation");
  bind(workspace, freshId, socket);
  retire(workspace, {freshId}, socket);

  std::vector<Owner> sources, outputs;
  for (size_t i = 0; i < 64; ++i) {
    sources.push_back(allocation(kBytes + 16));
    outputs.push_back(allocation(kBytes));
    auto* unaligned = static_cast<unsigned char*>(sources.back().get()) + 2;
    fill(unaligned, bf16((rank + 1) * (i + 1)));
  }
  for (size_t i = 0; i < sources.size(); ++i) {
    workspace.allReduceStaged(
        static_cast<unsigned char*>(sources[i].get()) + 2,
        outputs[i].get(),
        kBytes,
        stream);
  }
  workspace.drain();
  for (size_t i = 0; i < outputs.size(); ++i) {
    checkOutput(outputs[i].get(), bf16(3 * (i + 1)));
  }
  // A bounded non-vector-tail payload exercises the unchanged canonical
  // builder.
  auto tailInput = allocation(kBytes + 16);
  auto tailOutput = allocation(kBytes + 16);
  fill(tailInput.get(), bf16(rank + 1), kBytes + 16);
  workspace.allReduceStaged(
      tailInput.get(), tailOutput.get(), kBytes + 16, stream);
  workspace.drain();
  checkOutput(tailOutput.get(), bf16(3), kBytes + 16);
  auto finalReleases = std::make_shared<unsigned>(0);
  auto finalOwner = allocation(kBytes, finalReleases, &workspace);
  const auto finalId =
      workspace.registerInput(finalOwner.get(), kBytes, finalOwner);
  bind(workspace, finalId, socket);
  finalOwner.reset();
  int original = -1;
  hipCheck(hipGetDevice(&original));
  hipCheck(hipSetDevice(peerDevice));
  close(workspace, socket);
  int current = -1;
  hipCheck(hipGetDevice(&current));
  expect(
      current == peerDevice, "workspace control cleanup changed caller device");
  hipCheck(hipSetDevice(original));
  expect(
      *finalReleases == 1,
      "whole-workspace close failed to release owner outside locks");
}
void unsupportedCase(
    Workspace& workspace,
    int rank,
    int socket,
    hipStream_t stream) {
  void* pointer = nullptr;
  hipCheck(hipMallocAsync(&pointer, kBytes, stream));
  hipCheck(hipStreamSynchronize(stream));
  Owner owner(pointer, [stream](void* p) { (void)hipFreeAsync(p, stream); });
  bool unsupported = false;
  try {
    workspace.registerInput(pointer, kBytes, owner);
  } catch (const UnsupportedAllocation&) {
    unsupported = true;
  }
  expect(
      unsupported && !workspace.poisoned(),
      "unsupported pool allocation poisoned workspace");
  auto output = allocation(kBytes);
  fill(pointer, bf16(rank + 1));
  workspace.allReduceStaged(pointer, output.get(), kBytes, stream);
  workspace.drain();
  checkOutput(output.get(), bf16(3));
  owner.reset();
  hipCheck(hipStreamSynchronize(stream));
  close(workspace, socket);
}
std::shared_ptr<unsigned> faultCase(
    Workspace& workspace,
    int rank,
    int socket,
    hipStream_t stream,
    const std::string& mode) {
  auto releases = std::make_shared<unsigned>(0);
  auto input = allocation(2 * kBytes, releases);
  auto output = allocation(kBytes);
  fill(input.get(), bf16(rank + 1));
  fill(output.get(), 0xa5a5);
  const auto id = workspace.registerInput(input.get(), kBytes, input);
  const auto peer = exchange(
      socket, decodeRecord<RegistrationDescriptor>(workspace.describe(id)));
  workspace.bindPeer(id, encodeRecord(peer));
  if (mode == "generation") {
    const auto newId = workspace.registerInput(input.get(), kBytes, input);
    retire(workspace, {id}, socket);
    rejects([&] { workspace.bindPeer(newId, encodeRecord(peer)); });
  } else if (mode == "stale") {
    workspace.registerInput(input.get(), kBytes, input);
    retire(workspace, {id}, socket);
    rejects(
        [&] { workspace.allReduceDirect(id, output.get(), kBytes, stream); });
  } else if (mode == "stream") {
    hipStream_t wrong = nullptr;
    hipCheck(hipStreamCreateWithFlags(&wrong, hipStreamNonBlocking));
    rejects(
        [&] { workspace.allReduceDirect(id, output.get(), kBytes, wrong); });
    hipCheck(hipStreamDestroy(wrong));
  } else if (mode == "overlap") {
    rejects([&] {
      workspace.allReduceDirect(
          id, static_cast<unsigned char*>(input.get()) + 16, kBytes, stream);
    });
  } else if (mode == "staged_overlap") {
    rejects([&] {
      workspace.allReduceStaged(input.get(), input.get(), kBytes, stream);
    });
  } else if (mode == "ack") {
    rejects([&] { workspace.retire({id}); });
  } else if (mode == "capture") {
    hipCheck(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
    rejects(
        [&] { workspace.allReduceDirect(id, output.get(), kBytes, stream); });
    hipGraph_t graph = nullptr;
    hipCheck(hipStreamEndCapture(stream, &graph));
    hipCheck(hipGraphDestroy(graph));
  } else if (mode == "launch") {
    failLaunch = true;
    rejects(
        [&] { workspace.allReduceDirect(id, output.get(), kBytes, stream); });
    failLaunch = false;
  } else if (mode == "poison" || mode == "destructor") {
    if (mode == "poison") {
      holdCompletion.store(true);
      std::atomic<bool> drainRejected{false};
      std::thread waiting([&] {
        try {
          workspace.drain();
        } catch (const std::exception&) {
          drainRejected.store(true);
        }
      });
      while (!completionQueried.load()) {
        std::this_thread::yield();
      }
      const auto before = std::chrono::steady_clock::now();
      workspace.poison();
      expect(
          workspace.poisoned(),
          "poison was not visible during completion wait");
      expect(
          std::chrono::steady_clock::now() - before < std::chrono::seconds(1),
          "poison blocked behind GPU completion");
      waiting.join();
      holdCompletion.store(false);
      expect(drainRejected.load(), "poison did not interrupt completion wait");
    }
  } else {
    throw std::runtime_error("unknown fault mode");
  }
  if (mode != "destructor") {
    expect(workspace.poisoned(), "native fault did not poison");
  }
  input.reset();
  expect(*releases == 0, "fault released registered exported owner");
  exchange(socket, uint64_t{1});
  return releases;
}
std::array<int, 8> parsePool(const std::string& text) {
  std::array<int, 8> pool{};
  const char* cursor = text.c_str();
  for (size_t i = 0; i < pool.size(); ++i) {
    expect(
        *cursor >= '0' && *cursor <= '9', "pool needs eight device ordinals");
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(cursor, &end, 10);
    expect(!errno && value >= 0 && value <= 1024, "invalid device ordinal");
    pool[i] = value;
    expect(
        std::find(pool.begin(), pool.begin() + i, pool[i]) == pool.begin() + i,
        "pool contains duplicate devices");
    expect(
        i == pool.size() - 1 ? *end == 0 : *end == ',',
        "pool needs eight ordinals");
    cursor = end + (i == pool.size() - 1 ? 0 : 1);
  }
  return pool;
}
void worker(
    int rank,
    int socket,
    const std::string& mode,
    const std::array<int, 8>& pool) {
  hipCheck(hipSetDevice(pool[rank]));
  Config config;
  config.device = pool[rank];
  config.peerDevice = pool[1 - rank];
  config.capacity = 4ULL << 20;
  config.helperDevices.assign(pool.begin() + 2, pool.end());
  hipStream_t stream = nullptr;
  hipCheck(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  expect(hipipc_relay_abi_version() == kAbiVersion, "native ABI mismatch");
  auto workspace = Workspace::create(config);
  if (mode == "abi" || mode == "capacity" || mode == "helper_order" ||
      mode == "import") {
    auto peer = exchange(
        socket,
        decodeRecord<WorkspaceDescriptor>(workspace->localDescriptor()));
    if (mode == "abi") {
      ++peer.version;
    } else if (mode == "capacity") {
      peer.capacity -= 16;
    } else if (mode == "helper_order") {
      std::swap(peer.helperBus[0], peer.helperBus[1]);
    } else {
      peer.flags = peer.input;
    }
    rejects([&] {
      workspace->connect(encodeRecord(peer));
      if (mode == "import") {
        workspace->probe(stream);
      }
    });
    expect(workspace->poisoned(), "invalid descriptor failed to poison");
    exchange(socket, uint64_t{1});
    return;
  }
  bootstrap(*workspace, socket, stream, rank);
  if (mode == "normal") {
    normalCases(*workspace, rank, socket, stream, config.peerDevice);
  } else if (mode == "unsupported") {
    unsupportedCase(*workspace, rank, socket, stream);
  } else {
    const auto releases = faultCase(*workspace, rank, socket, stream, mode);
    workspace.reset();
    expect(*releases == 0, "destructor freed unacknowledged exported owner");
    exchange(socket, uint64_t{2});
  }
  if (mode == "normal" || mode == "unsupported") {
    hipCheck(hipStreamDestroy(stream));
  }
}
std::string canonicalPath(const std::string& path) {
  expect(!path.empty() && path.front() == '/', "bundle path must be absolute");
  std::unique_ptr<char, decltype(&std::free)> resolved(
      realpath(path.c_str(), nullptr), &std::free);
  expect(static_cast<bool>(resolved), "bundle path cannot be resolved");
  return resolved.get();
}
void verifyLibrary(const std::string& native) {
  Dl_info loaded{};
  void* marker = dlsym(RTLD_DEFAULT, "hipipc_relay_abi_version");
  expect(
      marker && dladdr(marker, &loaded) && loaded.dli_fname,
      "cannot identify loaded workspace library");
  expect(
      canonicalPath(loaded.dli_fname) ==
          canonicalPath(native + "/libhipipc_relay.so"),
      "worker loaded a stale DSO");
  std::fprintf(stderr, "HIPIPC fresh worker DSO: %s\n", loaded.dli_fname);
}
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 7 && std::string(argv[1]) == "--worker") {
      verifyLibrary(argv[5]);
      int runtime = 0;
      hipCheck(hipRuntimeGetVersion(&runtime));
      Dl_info provider{};
      expect(
          dladdr(dlsym(RTLD_DEFAULT, "hipRuntimeGetVersion"), &provider) &&
              provider.dli_fname,
          "cannot identify HIP runtime provider");
      std::fprintf(
          stderr, "HIPIPC runtime %d: %s\n", runtime, provider.dli_fname);
      worker(
          std::stoi(argv[2]), std::stoi(argv[3]), argv[4], parsePool(argv[6]));
      _exit(0);
    }
    const std::string mode = argc > 1 ? argv[1] : "normal";
    std::string native = HIPIPC_TEST_NATIVE_DIR;
    for (int i = 2; i < argc; ++i) {
      expect(
          std::string(argv[i]) == "--native-dir" && i + 1 < argc,
          "usage: test MODE [--native-dir ABS_BUNDLE]");
      native = argv[++i];
    }
    native = canonicalPath(native);
    const char* authorization =
        std::getenv("HIPIPC_TEST_EXCLUSIVE_GFX950_POOL");
    expect(authorization, "exclusive gfx950 pool authorization required");
    const std::string poolText = authorization;
    (void)parsePool(poolText);
    const auto executable = canonicalPath("/proc/self/exe");
    int sockets[2]{};
    expect(
        socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sockets) == 0,
        "socketpair failed");
    pid_t children[2]{};
    for (int rank = 0; rank < 2; ++rank) {
      children[rank] = fork();
      if (children[rank] == 0) {
        ::close(sockets[1 - rank]);
        const auto rankText = std::to_string(rank);
        const auto socketText = std::to_string(sockets[rank]);
        const char* previous = std::getenv("LD_LIBRARY_PATH");
        const std::string search =
            native + (previous ? ":" + std::string(previous) : "");
        if (setenv("LD_LIBRARY_PATH", search.c_str(), 1)) {
          _exit(1);
        }
        execl(
            executable.c_str(),
            executable.c_str(),
            "--worker",
            rankText.c_str(),
            socketText.c_str(),
            mode.c_str(),
            native.c_str(),
            poolText.c_str(),
            static_cast<char*>(nullptr));
        _exit(1);
      }
      if (children[rank] < 0) {
        if (rank == 1) {
          kill(children[0], SIGKILL);
          (void)waitpid(children[0], nullptr, 0);
        }
        return 1;
      }
    }
    ::close(sockets[0]);
    ::close(sockets[1]);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(180);
    bool success = true;
    size_t remaining = 2;
    while (remaining && std::chrono::steady_clock::now() < deadline) {
      for (auto& child : children) {
        if (child <= 0) {
          continue;
        }
        int status = 0;
        if (waitpid(child, &status, WNOHANG) == child) {
          success &= WIFEXITED(status) && WEXITSTATUS(status) == 0;
          child = 0;
          --remaining;
        }
      }
      if (!success) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    for (auto child : children) {
      if (child > 0) {
        kill(child, SIGKILL);
        (void)waitpid(child, nullptr, 0);
        success = false;
      }
    }
    return success ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "HIPIPC native test: %s\n", error.what());
    if (argc > 1 && std::string(argv[1]) == "--worker") {
      _exit(1);
    }
    return 1;
  }
}
