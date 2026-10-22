/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "hipipc_relay.h"

#include <hip/hip_runtime.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>

#include "comms/rcclx/develop/meta/relay/registered_allreduce_relay_kernels.h"
#include "hipipc_probe.h"

namespace rcclx::relay::hipipc {
namespace {
static_assert(sizeof(hipIpcMemHandle_t) == sizeof(IpcHandle));
static_assert(sizeof(void*) == 8);
constexpr size_t kFlagPrefix = 256;
static_assert(kFlagPrefix % alignof(RegisteredRelayFlags) == 0);
constexpr size_t kFlagBytes = kFlagPrefix + sizeof(RegisteredRelayFlags);
constexpr size_t kProbeSites = 128;
constexpr auto kCompletionTimeout = std::chrono::seconds(30);
constexpr RegisteredRelayLaunchPolicy
    kPolicy{256ULL << 10, 256ULL << 10, 1, 8, 8, 2};

uint64_t nextIdentity() {
  static std::atomic<uint64_t> serial{1};
  static const uint64_t seed =
      std::chrono::steady_clock::now().time_since_epoch().count() ^
      (static_cast<uint64_t>(getpid()) << 32);
  for (;;) {
    uint64_t value = seed + serial.fetch_add(1, std::memory_order_relaxed);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    if (value) {
      return value;
    }
  }
}
void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}
void check(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + hipGetErrorString(status));
  }
}
class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    check(hipGetDevice(&previous_), "get current device");
    check(hipSetDevice(device), "select device");
  }
  ~DeviceGuard() {
    (void)hipSetDevice(previous_);
  }

 private:
  int previous_{-1};
};
BusId busId(int device) {
  BusId bus{};
  check(
      hipDeviceGetPCIBusId(bus.data(), bus.size(), device), "get PCI identity");
  hipDeviceProp_t properties{};
  check(hipGetDeviceProperties(&properties, device), "get device properties");
  require(
      std::strncmp(properties.gcnArchName, "gfx950", 6) == 0,
      "workspace requires gfx950 devices");
  return bus;
}
void enablePair(int first, int second) {
  for (auto direction : {std::pair<int, int>{first, second}, {second, first}}) {
    int accessible = 0;
    check(
        hipDeviceCanAccessPeer(&accessible, direction.first, direction.second),
        "query peer access");
    require(accessible != 0, "required peer direction is inaccessible");
    DeviceGuard guard(direction.first);
    const auto status = hipDeviceEnablePeerAccess(direction.second, 0);
    if (status == hipErrorPeerAccessAlreadyEnabled) {
      (void)hipGetLastError();
    } else {
      check(status, "enable peer access");
    }
  }
}
IpcHandle exportHandle(void* base, bool producer = false) {
  hipIpcMemHandle_t handle{};
  const auto status = hipIpcGetMemHandle(&handle, base);
  if (producer &&
      (status == hipErrorInvalidHandle || status == hipErrorNotSupported ||
       status == hipErrorInvalidValue ||
       status == hipErrorInvalidDevicePointer)) {
    (void)hipGetLastError();
    throw UnsupportedAllocation(
        "allocation cannot be exported with legacy HIP IPC");
  }
  check(status, "export legacy IPC allocation");
  IpcHandle result{};
  std::memcpy(result.data(), &handle, sizeof(handle));
  return result;
}
struct ProbeWord {
  uint64_t lo{};
  uint64_t hi{};
  bool operator==(const ProbeWord& other) const {
    return lo == other.lo && hi == other.hi;
  }
};
ProbeWord canary(uint64_t generation, size_t domain, size_t site) {
  const uint64_t value = generation ^ (domain * 0x9e3779b97f4a7c15ULL) ^ site;
  return {value, ~value};
}
std::vector<size_t> offsets(size_t bytes) {
  std::vector<size_t> result;
  for (size_t offset = 0; offset + 16 <= bytes; offset += 2ULL << 20) {
    result.push_back(offset);
  }
  if (result.empty() || result.back() != bytes - 16) {
    result.push_back(bytes - 16);
  }
  return result;
}
void writeCanaries(
    void* base,
    size_t bytes,
    uint64_t generation,
    size_t domain) {
  size_t site = 0;
  for (auto offset : offsets(bytes)) {
    const auto word = canary(generation, domain, site++);
    check(
        hipMemcpy(
            static_cast<unsigned char*>(base) + offset,
            &word,
            16,
            hipMemcpyHostToDevice),
        "write workspace canary");
  }
}
struct CaptureInfo {
  unsigned long long identity{};
  hipGraph_t graph{};
  std::vector<hipGraphNode_t> dependencies;
};
CaptureInfo captureInfo(hipStream_t stream) {
  CaptureInfo result;
  hipStreamCaptureStatus status{};
  const hipGraphNode_t* dependencies = nullptr;
  size_t count = 0;
  check(
      hipStreamGetCaptureInfo_v2(
          stream,
          &status,
          &result.identity,
          &result.graph,
          &dependencies,
          &count),
      "get stream capture dependencies");
  require(
      status == hipStreamCaptureStatusActive && result.graph && result.identity,
      "captureAllReduce requires active CAP");
  if (count) {
    result.dependencies.assign(dependencies, dependencies + count);
  }
  return result;
}
hipGraphNode_t latestNode(
    hipStream_t stream,
    const CaptureInfo& before,
    hipGraphNodeType expected) {
  const auto after = captureInfo(stream);
  require(
      after.identity == before.identity && after.graph == before.graph &&
          after.dependencies.size() == 1,
      "capture changed during canonical enqueue");
  const auto node = after.dependencies.front();
  hipGraphNodeType type{};
  check(hipGraphNodeGetType(node, &type), "get latest captured node type");
  require(
      type == expected, "canonical enqueue did not produce the expected node");
  return node;
}
} // namespace

namespace detail {
struct Mapping {
  void* base{};
  uint64_t generation{};
  size_t bytes{};
  size_t refs{};
};
struct CaptureRecord {
  RegistrationId id{};
  uint64_t order{};
  Owner owner;
  const void* input{};
  void* output{};
  size_t bytes{};
  hipStream_t stream{};
  CaptureInfo capture;
  hipGraphNode_t copy{};
  hipGraphNode_t kernel{};
  CapturedDescriptor local{};
  CapturedDescriptor peer{};
  bool described{};
  bool bound{};
  bool finalized{};
};
struct WorkspaceState {
  explicit WorkspaceState(Config c) : config(std::move(c)) {}
  Config config;
  std::mutex mutex;
  WorkspaceDescriptor local{};
  WorkspaceDescriptor peer{};
  BusId peerBus{};
  int rank{};
  RegistrationRecords records;
  std::map<uint64_t, CaptureRecord> captures;
  uint64_t captureOrder{};
  std::map<IpcHandle, Mapping> imports;
  std::set<RegistrationId> boundPeerIds;
  std::vector<RegistrationId> retiring;
  Owner incomingOwner;
  std::vector<std::pair<int, void*>> allocations;
  std::vector<void*> rings;
  std::vector<void*> peerRings;
  std::array<hipEvent_t, 7> helperEvents{};
  std::array<const unsigned char**, 7> helperAddresses{};
  std::array<ProbeWord*, 7> helperResults{};
  void* input{};
  void* flagBase{};
  RegisteredRelayFlags* flags{};
  void* peerInput{};
  void* peerFlagBase{};
  RegisteredRelayFlags* peerFlags{};
  const unsigned char** probeAddresses{};
  ProbeWord* probeResults{};
  hipEvent_t completion{};
  hipStream_t stream{};
  enum class Phase { Ready, Poisoned, Closed };
  std::atomic<Phase> phase{Phase::Ready};
  bool connected{};
  bool probed{};
  bool allClosing{};
  bool importsClosed{};
  std::atomic<bool> retained{};

  void healthy() const {
    require(phase == Phase::Ready, "workspace is poisoned or closed");
  }
  void usable() const {
    healthy();
    require(
        probed && !allClosing && retiring.empty(),
        "workspace is not ready or is retiring");
  }
  void streamCheck(hipStream_t candidate, bool bind = false) {
    int current = -1;
    check(hipGetDevice(&current), "get execution device");
    require(
        current == config.device, "current device does not match workspace");
    hipDevice_t device = current;
    check(hipStreamGetDevice(candidate, &device), "get stream device");
    require(device == config.device, "stream belongs to another device");
    hipStreamCaptureStatus capture{};
    check(hipStreamIsCapturing(candidate, &capture), "query stream capture");
    require(
        capture == hipStreamCaptureStatusNone,
        "eager workspace call during capture");
    require(bind || candidate == stream, "execution stream cannot change");
    if (bind) {
      stream = candidate;
    }
  }
  void wait(hipEvent_t event) {
    const auto deadline = std::chrono::steady_clock::now() + kCompletionTimeout;
    for (;;) {
      healthy();
      const auto status = hipEventQuery(event);
      if (status == hipSuccess) {
        return;
      }
      if (status != hipErrorNotReady) {
        check(status, "query stream completion");
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error(
            "HIPIPC completion deadline exceeded; terminate workers");
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
  void drain() {
    if (probed) {
      streamCheck(stream);
      check(hipEventRecord(completion, stream), "record stream completion");
      wait(completion);
    }
  }
  void* allocate(int device, size_t bytes, bool uncached = true) {
    DeviceGuard guard(device);
    void* pointer = nullptr;
    check(
        uncached
            ? hipExtMallocWithFlags(&pointer, bytes, hipDeviceMallocUncached)
            : hipMalloc(&pointer, bytes),
        "allocate persistent workspace resource");
    allocations.emplace_back(device, pointer);
    check(
        hipMemset(pointer, 0, bytes),
        "initialize persistent workspace resource");
    return pointer;
  }
  void* open(const IpcHandle& handle, uint64_t generation, size_t bytes) {
    auto found = imports.find(handle);
    if (found != imports.end()) {
      require(
          found->second.base && found->second.generation == generation &&
              found->second.bytes == bytes,
          "live IPC allocation identity changed");
      ++found->second.refs;
      return found->second.base;
    }
    auto& mapping = imports[handle];
    mapping.generation = generation;
    mapping.bytes = bytes;
    mapping.refs = 1;
    hipIpcMemHandle_t ipc{};
    std::memcpy(&ipc, handle.data(), sizeof(ipc));
    check(
        hipIpcOpenMemHandle(&mapping.base, ipc, hipIpcMemLazyEnablePeerAccess),
        "open peer IPC allocation");
    return mapping.base;
  }
  void closeImport(const IpcHandle& handle) {
    auto found = imports.find(handle);
    require(
        found != imports.end() && found->second.refs,
        "missing peer IPC import");
    if (found->second.refs == 1) {
      check(
          hipIpcCloseMemHandle(found->second.base),
          "close peer IPC allocation");
      imports.erase(found);
    } else {
      --found->second.refs;
    }
  }
  RegisteredRelayArgs
  args(const void* mine, const void* theirs, void* out, size_t bytes) {
    return makeRegisteredRelayArgs(
        mine,
        theirs,
        out,
        bytes,
        rank,
        rings.size(),
        kPolicy,
        rings.data(),
        peerRings.data(),
        flags,
        peerFlags);
  }
  void payload(const void* in, void* out, size_t bytes) const {
    require(
        in && out && reinterpret_cast<uintptr_t>(out) % 16 == 0 &&
            bytes >= (2ULL << 20) && bytes <= config.capacity &&
            bytes % 16 == 0,
        "invalid bounded BF16 payload/output");
    require(
        !rangesOverlap(
            reinterpret_cast<uintptr_t>(in),
            bytes,
            reinterpret_cast<uintptr_t>(out),
            bytes),
        "output overlaps its input");
  }
  CaptureRecord& captured(RegistrationId id) {
    auto found = captures.find(id.identity);
    require(
        found != captures.end() && found->second.id == id,
        "stale captured op identity/generation");
    return found->second;
  }
  void controlCheck(RegistrationId id) {
    if (captures.count(id.identity)) {
      captured(id);
    } else {
      records.get(id);
    }
  }
  void captureEnded(const CaptureRecord& op) {
    streamCheck(stream);
    hipStreamCaptureStatus status{};
    check(hipStreamIsCapturing(op.stream, &status), "query CAP after capture");
    require(
        status == hipStreamCaptureStatusNone,
        "end CAP before importing/finalizing");
  }
  void cleanup() {
    require(imports.empty(), "close imports before freeing exported memory");
    DeviceGuard guard(config.device);
    if (completion) {
      check(hipEventDestroy(completion), "destroy completion event");
      completion = nullptr;
    }
    for (size_t h = 0; h < config.helperDevices.size(); ++h) {
      DeviceGuard helperGuard(config.helperDevices[h]);
      if (helperEvents[h]) {
        check(
            hipEventDestroy(helperEvents[h]), "destroy helper readiness event");
        helperEvents[h] = nullptr;
      }
    }
    for (auto& allocation : allocations) {
      if (allocation.second) {
        DeviceGuard allocationGuard(allocation.first);
        check(
            hipFree(allocation.second), "free acknowledged workspace resource");
        allocation.second = nullptr;
      }
    }
  }
};
} // namespace detail

namespace {
using detail::WorkspaceState;
void retainForever(const std::shared_ptr<WorkspaceState>& state) noexcept {
  auto phase = state->phase.load();
  do {
    if (phase == WorkspaceState::Phase::Closed) {
      return;
    }
  } while (!state->phase.compare_exchange_weak(
      phase, WorkspaceState::Phase::Poisoned));
  if (state->retained.exchange(true)) {
    return;
  }
  try {
    // Neither the vector nor its owners are destroyed during process shutdown.
    static auto* mutex = new std::mutex;
    static auto* retained = new std::vector<std::shared_ptr<WorkspaceState>>;
    std::lock_guard<std::mutex> lock(*mutex);
    retained->push_back(state);
  } catch (...) {
    std::terminate();
  }
}
template <class F>
decltype(auto) locked(
    const std::shared_ptr<WorkspaceState>& state,
    F&& function) {
  std::lock_guard<std::mutex> lock(state->mutex);
  try {
    return function(*state);
  } catch (const UnsupportedAllocation&) {
    throw;
  } catch (...) {
    retainForever(state);
    throw;
  }
}
void initialize(WorkspaceState& s) {
  const auto& c = s.config;
  require(
      c.device >= 0 && c.peerDevice >= 0 && c.device != c.peerDevice &&
          c.capacity >= (2ULL << 20) && c.capacity <= kMaxCapacityBytes &&
          c.capacity % 16 == 0 && !c.helperDevices.empty() &&
          c.helperDevices.size() <= 7,
      "invalid workspace config");
  s.allocations.reserve(4 + 3 * c.helperDevices.size());
  s.rings.reserve(c.helperDevices.size());
  s.peerRings.reserve(c.helperDevices.size());
  s.local.generation = nextIdentity();
  s.local.capacity = c.capacity;
  s.local.helperCount = c.helperDevices.size();
  s.local.activeBus = busId(c.device);
  s.peerBus = busId(c.peerDevice);
  require(s.local.activeBus != s.peerBus, "active devices alias");
  // Canonical combine needs complementary operand order, not control-group
  // rank.
  s.rank = s.local.activeBus < s.peerBus ? 0 : 1;
  for (size_t h = 0; h < c.helperDevices.size(); ++h) {
    const int helper = c.helperDevices[h];
    require(
        helper >= 0 && helper != c.device && helper != c.peerDevice &&
            std::find(
                c.helperDevices.begin(), c.helperDevices.begin() + h, helper) ==
                c.helperDevices.begin() + h,
        "invalid or duplicate helper device");
    s.local.helperBus[h] = busId(helper);
    enablePair(c.device, helper);
    enablePair(c.peerDevice, helper);
  }
  enablePair(c.device, c.peerDevice);
  s.input = s.allocate(c.device, c.capacity);
  s.flagBase = s.allocate(c.device, kFlagBytes);
  s.flags = reinterpret_cast<RegisteredRelayFlags*>(
      static_cast<unsigned char*>(s.flagBase) + kFlagPrefix);
  writeCanaries(s.input, c.capacity, s.local.generation, 1);
  // Live flags remain zero once initialized. Only the stable prefix is probed.
  writeCanaries(s.flagBase, kFlagPrefix, s.local.generation, 2);
  for (size_t h = 0; h < c.helperDevices.size(); ++h) {
    const int helper = c.helperDevices[h];
    void* ring = s.allocate(helper, kRingBytes);
    s.rings.push_back(ring);
    DeviceGuard guard(helper);
    writeCanaries(ring, kRingBytes, s.local.generation, h + 3);
    s.local.rings[h] = exportHandle(ring);
    check(
        hipEventCreateWithFlags(&s.helperEvents[h], hipEventDisableTiming),
        "create persistent helper readiness event");
    s.helperAddresses[h] = static_cast<const unsigned char**>(
        s.allocate(helper, 3 * sizeof(void*), false));
    s.helperResults[h] = static_cast<ProbeWord*>(
        s.allocate(helper, 3 * sizeof(ProbeWord), false));
    std::array<const unsigned char*, 3> addresses{};
    std::array<ProbeWord, 3> expected{};
    const auto sites = offsets(kRingBytes);
    for (size_t i = 0; i < sites.size(); ++i) {
      addresses[i] = static_cast<unsigned char*>(ring) + sites[i];
      expected[i] = canary(s.local.generation, h + 3, i);
    }
    check(
        hipMemcpy(
            s.helperAddresses[h],
            addresses.data(),
            sizeof(addresses),
            hipMemcpyHostToDevice),
        "upload helper shader probe");
    // Default stream: a dedicated stream would keep one more hardware queue
    // mapped on every helper device for the life of the workspace.
    check(
        hipipcRelayProbeRead(
            s.helperAddresses[h],
            sites.size(),
            reinterpret_cast<unsigned long long*>(s.helperResults[h]),
            nullptr),
        "warm helper shader resources");
    check(
        hipEventRecord(s.helperEvents[h], nullptr), "record helper completion");
    s.wait(s.helperEvents[h]);
    std::array<ProbeWord, 3> results{};
    check(
        hipMemcpy(
            results.data(),
            s.helperResults[h],
            sizeof(results),
            hipMemcpyDeviceToHost),
        "read helper shader probe");
    require(results == expected, "helper shader probe corrupted owned ring");
  }
  DeviceGuard guard(c.device);
  s.local.input = exportHandle(s.input);
  s.local.flags = exportHandle(s.flagBase);
  s.probeAddresses = static_cast<const unsigned char**>(
      s.allocate(c.device, kProbeSites * sizeof(void*), false));
  s.probeResults = static_cast<ProbeWord*>(
      s.allocate(c.device, kProbeSites * sizeof(ProbeWord), false));
  check(
      hipEventCreateWithFlags(&s.completion, hipEventDisableTiming),
      "create persistent completion event");
}
std::vector<RegistrationId> sortedIds(const std::vector<RegistrationId>& ids) {
  auto sorted = ids;
  std::sort(sorted.begin(), sorted.end());
  require(
      std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
      "duplicate retirement identity");
  return sorted;
}
void closeBatch(
    WorkspaceState& s,
    const std::vector<RegistrationId>& ids,
    bool all) {
  s.healthy();
  if (s.importsClosed) {
    require(
        s.allClosing == all && s.retiring == ids,
        "different retirement batch pending");
    return;
  }
  require(!s.allClosing && s.retiring.empty(), "retirement already pending");
  for (auto id : ids) {
    s.controlCheck(id);
  }
  s.retiring = ids;
  s.allClosing = all;
  DeviceGuard guard(s.config.device);
  s.drain();
  for (auto id : ids) {
    auto found = s.records.inputs.find(id.identity);
    if (found != s.records.inputs.end() && found->second.bound) {
      s.closeImport(found->second.peer.base);
    }
  }
  if (all) {
    if (s.connected) {
      s.closeImport(s.peer.input);
      s.closeImport(s.peer.flags);
      for (size_t h = 0; h < s.peer.helperCount; ++h) {
        s.closeImport(s.peer.rings[h]);
      }
    }
    require(s.imports.empty(), "unexpected surviving IPC import");
  }
  s.importsClosed = true;
}
RegistrationId registerInputRecord(
    WorkspaceState& s,
    const void* input,
    size_t bytes,
    const Owner& owner,
    RegistrationId id) {
  s.usable();
  s.incomingOwner = owner;
  require(
      owner && input && reinterpret_cast<uintptr_t>(input) % 16 == 0 &&
          bytes <= s.config.capacity,
      "registration needs an aligned owned range");
  DeviceGuard guard(s.config.device);
  s.streamCheck(s.stream);
  hipPointerAttribute_t attributes{};
  check(
      hipPointerGetAttributes(&attributes, input), "query producer allocation");
  if (attributes.type != hipMemoryTypeDevice || attributes.isManaged) {
    s.incomingOwner.reset();
    throw UnsupportedAllocation(
        "registration requires non-managed HIP device memory");
  }
  require(
      attributes.device == s.config.device, "producer is on another device");
  void* base = nullptr;
  size_t allocationBytes = 0;
  const auto rangeStatus =
      hipMemGetAddressRange(&base, &allocationBytes, const_cast<void*>(input));
  if (rangeStatus == hipErrorNotSupported ||
      rangeStatus == hipErrorInvalidValue ||
      rangeStatus == hipErrorInvalidDevicePointer) {
    s.incomingOwner.reset();
    (void)hipGetLastError();
    throw UnsupportedAllocation("allocation has no legacy HIP address range");
  }
  check(rangeStatus, "resolve producer allocation base");
  const auto address = reinterpret_cast<uintptr_t>(input);
  require(
      address >= reinterpret_cast<uintptr_t>(base),
      "invalid producer allocation base");
  const size_t offset = address - reinterpret_cast<uintptr_t>(base);
  validateRange(allocationBytes, offset, bytes);
  require(
      base != s.input && base != s.flagBase,
      "register producer, not workspace storage");
  auto found = s.records.bases.find(base);
  IpcHandle handle{};
  uint64_t allocationGeneration = nextIdentity();
  if (found != s.records.bases.end()) {
    require(
        found->second.bytes == allocationBytes,
        "live allocation extent changed");
    handle = found->second.handle;
    allocationGeneration = found->second.generation;
  } else {
    try {
      handle = exportHandle(base, true);
    } catch (const UnsupportedAllocation&) {
      s.incomingOwner.reset();
      throw;
    }
  }
  detail::InputRecord record;
  record.input = input;
  record.base = base;
  record.local.workspaceGeneration = s.local.generation;
  record.local.peerWorkspaceGeneration = s.peer.generation;
  record.local.registration = id;
  record.local.allocationGeneration = allocationGeneration;
  record.local.allocationBytes = allocationBytes;
  record.local.offset = offset;
  record.local.bytes = bytes;
  record.local.base = handle;
  s.records.add(record, owner);
  s.incomingOwner.reset();
  return id;
}
void bindInputRecord(
    WorkspaceState& s,
    RegistrationId id,
    const RegistrationDescriptor& peer,
    bool remember = true) {
  auto& r = s.records.get(id);
  require(!r.bound, "registration peer may only bind once");
  require(
      peer.workspaceGeneration == s.peer.generation &&
          peer.peerWorkspaceGeneration == s.local.generation &&
          peer.registration.identity && peer.registration.generation &&
          peer.allocationGeneration && peer.bytes == r.local.bytes,
      "peer registration identity/generation/size mismatch");
  validateRange(peer.allocationBytes, peer.offset, peer.bytes);
  // Keep seen IDs after retirement, rejecting replay of a retired descriptor.
  require(
      !remember || s.boundPeerIds.insert(peer.registration).second,
      "stale or already bound peer registration");
  DeviceGuard guard(s.config.device);
  s.open(peer.base, peer.allocationGeneration, peer.allocationBytes);
  r.peer = peer;
  r.bound = true;
}
} // namespace

Workspace::Workspace(std::shared_ptr<WorkspaceState> state)
    : state_(std::move(state)) {}
std::unique_ptr<Workspace> Workspace::create(const Config& config) {
  auto state = std::make_shared<WorkspaceState>(config);
  auto workspace = std::unique_ptr<Workspace>(new Workspace(state));
  locked(state, [](auto& s) { initialize(s); });
  return workspace;
}
Workspace::~Workspace() {
  retainForever(state_);
}
std::vector<uint8_t> Workspace::localDescriptor() const {
  return locked(state_, [](auto& s) {
    s.healthy();
    require(!s.allClosing, "workspace is closing");
    return encodeRecord(s.local);
  });
}
void Workspace::connect(const std::vector<uint8_t>& bytes) {
  locked(state_, [&](auto& s) {
    s.healthy();
    require(!s.connected, "connect may only run once");
    const auto peer = decodeRecord<WorkspaceDescriptor>(bytes);
    require(
        peer.generation && peer.generation != s.local.generation &&
            peer.capacity == s.local.capacity &&
            peer.helperCount == s.local.helperCount &&
            peer.helperBus == s.local.helperBus && peer.activeBus == s.peerBus,
        "peer workspace identity/topology/capacity mismatch");
    s.peer = peer;
    DeviceGuard guard(s.config.device);
    s.peerInput = s.open(peer.input, peer.generation, peer.capacity);
    s.peerFlagBase = s.open(peer.flags, peer.generation, kFlagBytes);
    s.peerFlags = reinterpret_cast<RegisteredRelayFlags*>(
        static_cast<unsigned char*>(s.peerFlagBase) + kFlagPrefix);
    for (size_t h = 0; h < peer.helperCount; ++h) {
      s.peerRings.push_back(s.open(peer.rings[h], peer.generation, kRingBytes));
    }
    s.connected = true;
  });
}
void Workspace::probe(void* stream) {
  locked(state_, [&](auto& s) {
    s.healthy();
    require(
        s.connected && !s.probed && !s.allClosing,
        "probe needs one complete connect");
    s.streamCheck(static_cast<hipStream_t>(stream), true);
    DeviceGuard guard(s.config.device);
    std::vector<const unsigned char*> addresses;
    std::vector<ProbeWord> expected;
    auto add = [&](const void* pointer, size_t bytes, size_t domain) {
      size_t site = 0;
      for (auto offset : offsets(bytes)) {
        addresses.push_back(
            static_cast<const unsigned char*>(pointer) + offset);
        expected.push_back(canary(s.peer.generation, domain, site++));
      }
    };
    add(s.peerInput, s.peer.capacity, 1);
    add(s.peerFlagBase, kFlagPrefix, 2);
    for (size_t h = 0; h < s.peerRings.size(); ++h) {
      add(s.peerRings[h], kRingBytes, h + 3);
    }
    require(
        addresses.size() <= kProbeSites, "probe exceeds persistent capacity");
    check(
        hipMemcpy(
            s.probeAddresses,
            addresses.data(),
            addresses.size() * sizeof(void*),
            hipMemcpyHostToDevice),
        "upload probe addresses");
    check(
        hipipcRelayProbeRead(
            s.probeAddresses,
            addresses.size(),
            reinterpret_cast<unsigned long long*>(s.probeResults),
            s.stream),
        "launch shader-visible workspace probe");
    check(hipEventRecord(s.completion, s.stream), "record probe completion");
    s.wait(s.completion);
    std::vector<ProbeWord> results(expected.size());
    check(
        hipMemcpy(
            results.data(),
            s.probeResults,
            results.size() * sizeof(ProbeWord),
            hipMemcpyDeviceToHost),
        "read probe results");
    require(
        results == expected, "workspace probe found stale or aliased mapping");
    // Zero vectors never access staging/rings. The prefix stays immutable while
    // the peer probes, even if this rank has already advanced the live flags.
    check(
        launchRegisteredRelayAllReduce(
            s.args(s.input, s.peerInput, s.input, 0), s.stream),
        "warm canonical relay runtime");
    check(
        hipEventRecord(s.completion, s.stream),
        "record canonical runtime readiness");
    s.wait(s.completion);
    s.probed = true;
  });
}
RegistrationId
Workspace::registerInput(const void* input, size_t bytes, Owner owner) {
  return locked(state_, [&](auto& s) {
    return registerInputRecord(
        s, input, bytes, owner, {nextIdentity(), nextIdentity()});
  });
}
std::vector<uint8_t> Workspace::describe(RegistrationId id) const {
  return locked(state_, [&](auto& s) {
    s.usable();
    return encodeRecord(s.records.get(id).local);
  });
}
void Workspace::bindPeer(RegistrationId id, const std::vector<uint8_t>& bytes) {
  locked(state_, [&](auto& s) {
    s.usable();
    bindInputRecord(s, id, decodeRecord<RegistrationDescriptor>(bytes));
  });
}
void Workspace::allReduceStaged(
    const void* input,
    void* output,
    size_t bytes,
    void* stream) {
  locked(state_, [&](auto& s) {
    s.usable();
    s.streamCheck(static_cast<hipStream_t>(stream));
    s.payload(input, output, bytes);
    DeviceGuard guard(s.config.device);
    check(
        hipMemcpyAsync(
            s.input, input, bytes, hipMemcpyDeviceToDevice, s.stream),
        "stage eager input");
    check(
        launchRegisteredRelayAllReduce(
            s.args(s.input, s.peerInput, output, bytes), s.stream),
        "launch canonical TP2 relay");
  });
}
void Workspace::allReduceDirect(
    RegistrationId id,
    void* output,
    size_t bytes,
    void* stream) {
  locked(state_, [&](auto& s) {
    s.usable();
    s.streamCheck(static_cast<hipStream_t>(stream));
    const auto& r = s.records.get(id);
    require(
        r.bound && bytes <= r.local.bytes,
        "direct execution needs a bound range");
    s.payload(r.input, output, bytes);
    const auto& mapping = s.imports.at(r.peer.base);
    const auto* peer =
        static_cast<unsigned char*>(mapping.base) + r.peer.offset;
    DeviceGuard guard(s.config.device);
    check(
        launchRegisteredRelayAllReduce(
            s.args(r.input, peer, output, bytes), s.stream),
        "launch canonical TP2 relay");
  });
}
CapturedOp Workspace::captureAllReduce(
    const void* input,
    void* output,
    size_t bytes,
    Owner owner,
    void* captureStream) {
  return locked(state_, [&](auto& s) {
    s.usable();
    s.incomingOwner = owner;
    const RegistrationId id{nextIdentity(), nextIdentity()};
    auto& op = s.captures[id.identity];
    op.id = id;
    op.order = ++s.captureOrder;
    op.owner = owner;
    op.input = input;
    op.output = output;
    op.bytes = bytes;
    op.stream = static_cast<hipStream_t>(captureStream);
    require(owner != nullptr, "capture requires input/output owner group");
    s.payload(input, output, bytes);
    require(op.stream != s.stream, "CAP must be separate from bound EXEC");
    int device = -1;
    check(hipGetDevice(&device), "get CAP device");
    require(device == s.config.device, "CAP current device mismatch");
    hipDevice_t streamDevice = device;
    check(
        hipStreamGetDevice(op.stream, &streamDevice), "get CAP stream device");
    require(streamDevice == device, "CAP stream device mismatch");
    op.capture = captureInfo(op.stream);
    check(
        hipMemcpyAsync(
            s.input, input, bytes, hipMemcpyDeviceToDevice, op.stream),
        "capture provisional staging copy");
    op.copy = latestNode(op.stream, op.capture, hipGraphNodeTypeMemcpy);
    check(
        launchRegisteredRelayAllReduce(
            s.args(s.input, s.peerInput, output, bytes), op.stream),
        "capture canonical TP2 relay");
    op.kernel = latestNode(op.stream, op.capture, hipGraphNodeTypeKernel);
    s.incomingOwner.reset();
    return CapturedOp{id};
  });
}
std::vector<uint8_t> Workspace::describe(CapturedOp token) {
  return locked(state_, [&](auto& s) {
    s.usable();
    auto& op = s.captured(token.id);
    s.captureEnded(op);
    if (!op.described) {
      auto& descriptor = op.local.input;
      descriptor.workspaceGeneration = s.local.generation;
      descriptor.peerWorkspaceGeneration = s.peer.generation;
      descriptor.registration = op.id;
      descriptor.bytes = op.bytes;
      if (reinterpret_cast<uintptr_t>(op.input) % 16 == 0) {
        try {
          registerInputRecord(s, op.input, op.bytes, op.owner, op.id);
          descriptor = s.records.get(op.id).local;
          op.local.role = CaptureRole::Direct;
        } catch (const UnsupportedAllocation&) {
          // The capture's owner remains rooted even for non-exportable storage.
        }
      }
      op.described = true;
    }
    return encodeRecord(op.local);
  });
}
void Workspace::bindPeer(CapturedOp token, const std::vector<uint8_t>& bytes) {
  locked(state_, [&](auto& s) {
    s.usable();
    auto& op = s.captured(token.id);
    s.captureEnded(op);
    require(
        op.described && !op.bound && !op.finalized,
        "describe then bind captured peer once");
    const auto peer = decodeRecord<CapturedDescriptor>(bytes);
    validateCapturedDescriptor(peer);
    require(
        peer.input.workspaceGeneration == s.peer.generation &&
            peer.input.peerWorkspaceGeneration == s.local.generation &&
            peer.input.bytes == op.bytes,
        "peer captured op workspace/size mismatch");
    require(
        s.boundPeerIds.insert(peer.input.registration).second,
        "stale or already bound peer captured op");
    if (op.local.role == CaptureRole::Direct &&
        peer.role == CaptureRole::Direct) {
      bindInputRecord(s, op.id, peer.input, false);
    }
    op.peer = peer;
    op.bound = true;
  });
}
void Workspace::finalizeCapture(
    void* rawGraph,
    const std::vector<CapturedOp>& ops) {
  require(
      rawGraph && !ops.empty(),
      "finalize requires exact raw graph and captured ops");
  locked(state_, [&](auto& s) {
    s.usable();
    const auto& first = s.captured(ops.front().id);
    s.captureEnded(first);
    std::vector<const detail::CaptureRecord*> expected;
    for (const auto& entry : s.captures) {
      const auto& op = entry.second;
      if (op.capture.graph == rawGraph &&
          op.capture.identity == first.capture.identity &&
          op.stream == first.stream) {
        expected.push_back(&op);
      }
    }
    std::sort(expected.begin(), expected.end(), [](auto* a, auto* b) {
      return a->order < b->order;
    });
    require(
        expected.size() == ops.size(), "incomplete captured operation list");
    for (size_t i = 0; i < ops.size(); ++i) {
      require(
          expected[i]->id == ops[i].id,
          "captured operations must follow capture order");
    }
    std::set<RegistrationId> seen;
    for (auto token : ops) {
      const auto& op = s.captured(token.id);
      require(
          op.capture.graph == rawGraph &&
              op.capture.identity == first.capture.identity &&
              op.stream == first.stream && op.described && op.bound &&
              !op.finalized && seen.insert(token.id).second,
          "wrong graph, stale, duplicate or unbound captured op");
    }
    for (auto token : ops) {
      auto& op = s.captured(token.id);
      const bool direct = op.local.role == CaptureRole::Direct &&
          op.peer.role == CaptureRole::Direct;
      const void* mine = s.input;
      const void* peer = s.peerInput;
      if (direct) {
        const auto& record = s.records.get(op.id);
        mine = op.input;
        peer =
            static_cast<unsigned char*>(s.imports.at(record.peer.base).base) +
            record.peer.offset;
      }
      auto arguments = s.args(mine, peer, op.output, op.bytes);
      hipKernelNodeParams params{};
      check(
          hipGraphKernelNodeGetParams(op.kernel, &params),
          "get captured canonical kernel params");
      void* argument = &arguments;
      params.extra = nullptr;
      params.kernelParams = &argument;
      check(
          hipGraphKernelNodeSetParams(op.kernel, &params),
          "patch canonical graph pointers before instantiate");
      if (direct) {
        // Only the copy and its recorded parents are rewired; no DAG traversal.
        for (auto parent : op.capture.dependencies) {
          check(
              hipGraphAddDependencies(op.capture.graph, &parent, &op.kernel, 1),
              "preserve direct kernel ordering without staging copy");
        }
        check(
            hipGraphDestroyNode(op.copy),
            "remove unnecessary direct staging copy");
      }
      op.finalized = true;
    }
  });
}
void Workspace::drain() {
  locked(state_, [](auto& s) {
    s.healthy();
    DeviceGuard guard(s.config.device);
    s.drain();
  });
}
void Workspace::closeImports(const std::vector<RegistrationId>& ids) {
  locked(state_, [&](auto& s) { closeBatch(s, sortedIds(ids), false); });
}
void Workspace::retire(const std::vector<RegistrationId>& ids) {
  std::vector<std::vector<Owner>> detached;
  std::vector<Owner> capturedOwners;
  locked(state_, [&](auto& s) {
    s.healthy();
    const auto batch = sortedIds(ids);
    require(
        !s.allClosing && s.importsClosed && s.retiring == batch,
        "retire needs the closed acknowledgment for this import batch");
    detached.reserve(batch.size());
    capturedOwners.reserve(batch.size());
    for (auto id : batch) {
      s.controlCheck(id);
    }
    for (auto id : batch) {
      if (s.records.inputs.count(id.identity)) {
        detached.push_back(s.records.erase(id));
      }
      auto found = s.captures.find(id.identity);
      if (found != s.captures.end()) {
        capturedOwners.push_back(std::move(found->second.owner));
        s.captures.erase(found);
      }
    }
    s.retiring.clear();
    s.importsClosed = false;
  });
  // Managed storage deleters may acquire locks or synchronize the device.
}
void Workspace::closeImports() {
  locked(state_, [](auto& s) {
    if (s.phase == WorkspaceState::Phase::Closed) {
      return;
    }
    std::vector<RegistrationId> ids;
    for (const auto& entry : s.records.inputs) {
      ids.push_back(entry.second.local.registration);
    }
    for (const auto& entry : s.captures) {
      if (!s.records.inputs.count(entry.first)) {
        ids.push_back(entry.second.id);
      }
    }
    closeBatch(s, sortedIds(ids), true);
  });
}
void Workspace::close() {
  detail::RegistrationRecords detached;
  std::map<uint64_t, detail::CaptureRecord> detachedCaptures;
  locked(state_, [&](auto& s) {
    if (s.phase == WorkspaceState::Phase::Closed) {
      return;
    }
    s.healthy();
    require(
        s.allClosing && s.importsClosed,
        "close needs the peer closed acknowledgment");
    // Finish every possibly failing HIP action before detaching any owners.
    s.cleanup();
    auto expected = WorkspaceState::Phase::Ready;
    require(
        s.phase.compare_exchange_strong(
            expected, WorkspaceState::Phase::Closed),
        "workspace poisoned during close");
    std::swap(detached, s.records);
    std::swap(detachedCaptures, s.captures);
  });
}
bool Workspace::poisoned() const {
  return state_->phase.load() == WorkspaceState::Phase::Poisoned;
}
void Workspace::poison() noexcept {
  retainForever(state_);
}
} // namespace rcclx::relay::hipipc
extern "C" uint32_t hipipc_relay_abi_version() {
  return rcclx::relay::hipipc::kAbiVersion;
}
