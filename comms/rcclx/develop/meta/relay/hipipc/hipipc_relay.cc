/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hipipc_relay.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "comms/rcclx/develop/meta/relay/registered_allreduce_relay_kernels.h"
#include "hipipc_probe.h"

namespace {

using rcclx::relay::launchRegisteredRelayAllReduce;
using rcclx::relay::makeRegisteredRelayArgs;
using rcclx::relay::RegisteredRelayArgs;
using rcclx::relay::RegisteredRelayFlags;
using rcclx::relay::RegisteredRelayLaunchPolicy;

constexpr uint64_t kDescriptorMagic = 0x49504352454c4159ULL; // "IPCRELAY"
constexpr uint32_t kAbi = 1;
constexpr size_t kAlignBytes = 16;
constexpr size_t kCapacityBytes = 144ULL << 20;
constexpr size_t kMinExecuteBytes = 2ULL << 20;
constexpr int kMaxHelpers = 7;
constexpr size_t kStagingBytesPerHelper = 4ULL << 20;
constexpr size_t kProbeStrideBytes = 2ULL << 20;
constexpr size_t kProbeBytes = 16;
constexpr size_t kMaxInputProbeSites = kCapacityBytes / kProbeStrideBytes + 1;
constexpr size_t kSliceBytes = 256ULL << 10;
constexpr int kDirectWeight = 1;
constexpr int kLanes = 8;
constexpr int kSlots = 2;
constexpr size_t kBusBytes = 32;

#if !defined(HIP_UNCACHED_MEMORY)
#error "Standalone gfx950 HIPIPC requires HIP_UNCACHED_MEMORY"
#endif
constexpr unsigned int kAllocFlags = hipDeviceMallocUncached;

thread_local std::string g_last_error;
std::atomic<uint64_t> g_nonce_counter{1};

void setError(const char* message) {
  g_last_error = message != nullptr ? message : "unknown HIPIPC relay error";
}

void setHipError(const char* operation, hipError_t status) {
  g_last_error = std::string(operation) + ": " + hipGetErrorString(status);
}

class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    status_ = hipGetDevice(&previous_);
    if (status_ == hipSuccess && previous_ != device) {
      status_ = hipSetDevice(device);
    }
  }

  ~DeviceGuard() {
    if (previous_ >= 0) {
      (void)hipSetDevice(previous_);
    }
  }

  hipError_t status() const {
    return status_;
  }

 private:
  int previous_{-1};
  hipError_t status_{hipSuccess};
};

struct ProbeWord {
  uint64_t lo;
  uint64_t hi;
};

struct Descriptor {
  uint64_t magic;
  uint32_t abi;
  uint32_t rank;
  uint64_t capacity;
  uint64_t nonce;
  uint64_t inputOffset;
  uint64_t inputAllocationBytes;
  uint32_t helperCount;
  uint32_t inputProbeCount;
  char activeBus[kBusBytes];
  char helperBus[kMaxHelpers][kBusBytes];
  hipIpcMemHandle_t inputBase;
  hipIpcMemHandle_t flags;
  hipIpcMemHandle_t staging[kMaxHelpers];
  ProbeWord inputOriginal[kMaxInputProbeSites];
};

struct ImportMapping {
  void* pointer;
};

struct RelayRequest {
  int rank{-1};
  int device{-1};
  int peerDevice{-1};
  std::vector<int> helpers{};
  std::array<std::array<char, kBusBytes>, kMaxHelpers> helperBus{};
  size_t capacity{0};
  void* input{nullptr};
  void* inputBase{nullptr};
  size_t inputOffset{0};
  size_t inputAllocationBytes{0};
  void* output{nullptr};
  RegisteredRelayFlags* flags{nullptr};
  std::vector<void*> staging{};
  std::vector<ImportMapping> imports{};
  void* peerInputBase{nullptr};
  void* peerInput{nullptr};
  RegisteredRelayFlags* peerFlags{nullptr};
  std::vector<void*> peerStaging{};
  std::array<ProbeWord, kMaxInputProbeSites> inputOriginal{};
  std::array<ProbeWord, kMaxInputProbeSites> peerInputOriginal{};
  uint32_t inputProbeCount{0};
  uint64_t nonce{0};
  uint64_t peerNonce{0};
  bool exported{false};
  bool importAttempted{false};
  bool imported{false};
  bool probed{false};
  bool closeCalled{false};
  bool streamBound{false};
  hipStream_t executionStream{nullptr};
  bool executionStarted{false};
  bool drained{true};
  bool quiescenceKnown{true};
  bool failStop{false};
};

uint64_t mix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

ProbeWord canary(uint64_t nonce, uint64_t domain, uint64_t site) {
  return {
      mix64(nonce ^ (domain << 32) ^ site),
      mix64(nonce ^ 0xd6e8feb86659fd93ULL ^ (domain << 32) ^ site)};
}

uint64_t newNonce(const void* address) {
  const uint64_t ticks = static_cast<uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const uint64_t serial =
      g_nonce_counter.fetch_add(1, std::memory_order_relaxed);
  return mix64(ticks ^ serial ^ reinterpret_cast<uintptr_t>(address));
}

std::vector<size_t> probeOffsets(size_t bytes) {
  std::vector<size_t> offsets;
  for (size_t offset = 0; offset + kProbeBytes <= bytes;
       offset += kProbeStrideBytes) {
    offsets.push_back(offset);
  }
  const size_t tail = bytes - kProbeBytes;
  if (offsets.empty() || offsets.back() != tail) {
    offsets.push_back(tail);
  }
  return offsets;
}

bool getBusId(int device, char* bus, size_t length) {
  const hipError_t status = hipDeviceGetPCIBusId(bus, length, device);
  if (status != hipSuccess) {
    setHipError("HIP PCI bus ID is unavailable", status);
    (void)hipGetLastError();
    return false;
  }
  return true;
}

int findDeviceByBus(const char* bus) {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) {
    (void)hipGetLastError();
    return -1;
  }
  for (int device = 0; device < count; ++device) {
    char candidate[kBusBytes] = {};
    if (hipDeviceGetPCIBusId(candidate, sizeof(candidate), device) ==
            hipSuccess &&
        std::strncmp(candidate, bus, kBusBytes) == 0) {
      return device;
    }
    (void)hipGetLastError();
  }
  return -1;
}

bool isGfx950(int device) {
  hipDeviceProp_t prop{};
  return hipGetDeviceProperties(&prop, device) == hipSuccess &&
      std::strncmp(prop.gcnArchName, "gfx950", 6) == 0;
}

bool enablePeerDirection(int from, int to) {
  int canAccess = 0;
  hipError_t status = hipDeviceCanAccessPeer(&canAccess, from, to);
  if (status != hipSuccess || canAccess == 0) {
    if (status != hipSuccess) {
      setHipError("HIP peer-access query failed", status);
      (void)hipGetLastError();
    } else {
      setError("required HIP peer-access direction is unavailable");
    }
    return false;
  }
  DeviceGuard guard(from);
  if (guard.status() != hipSuccess) {
    setHipError("cannot select a HIP device for peer access", guard.status());
    return false;
  }
  status = hipDeviceEnablePeerAccess(to, 0);
  if (status == hipSuccess || status == hipErrorPeerAccessAlreadyEnabled) {
    (void)hipGetLastError();
    return true;
  }
  setHipError("enabling HIP peer access failed", status);
  (void)hipGetLastError();
  return false;
}

bool enablePeerPair(int first, int second) {
  return enablePeerDirection(first, second) &&
      enablePeerDirection(second, first);
}

bool containsDevice(const std::vector<int>& devices, int device) {
  for (int candidate : devices) {
    if (candidate == device) {
      return true;
    }
  }
  return false;
}

bool writeCanarySites(
    void* pointer,
    size_t bytes,
    uint64_t nonce,
    uint64_t domain,
    std::array<ProbeWord, kMaxInputProbeSites>* originals,
    uint32_t* originalCount) {
  const std::vector<size_t> offsets = probeOffsets(bytes);
  if (originalCount != nullptr) {
    *originalCount = 0;
  }
  if (originals != nullptr && offsets.size() > originals->size()) {
    setError("input probe site count exceeds the descriptor capacity");
    return false;
  }
  for (size_t i = 0; i < offsets.size(); ++i) {
    auto* site = static_cast<unsigned char*>(pointer) + offsets[i];
    if (originals != nullptr) {
      hipError_t status =
          hipMemcpy(&(*originals)[i], site, kProbeBytes, hipMemcpyDeviceToHost);
      if (status != hipSuccess) {
        setHipError("saving an input probe site failed", status);
        (void)hipGetLastError();
        return false;
      }
      if (originalCount != nullptr) {
        *originalCount = static_cast<uint32_t>(i + 1);
      }
    }
    const ProbeWord value = canary(nonce, domain, i);
    const hipError_t status =
        hipMemcpy(site, &value, kProbeBytes, hipMemcpyHostToDevice);
    if (status != hipSuccess) {
      setHipError("writing a mapping probe canary failed", status);
      (void)hipGetLastError();
      return false;
    }
  }
  return true;
}

bool restoreInputCanaries(RelayRequest& state) {
  const std::vector<size_t> offsets = probeOffsets(state.capacity);
  for (size_t i = 0; i < state.inputProbeCount && i < offsets.size(); ++i) {
    const hipError_t status = hipMemcpy(
        static_cast<unsigned char*>(state.input) + offsets[i],
        &state.inputOriginal[i],
        kProbeBytes,
        hipMemcpyHostToDevice);
    if (status != hipSuccess) {
      setHipError("restoring a local input probe site failed", status);
      (void)hipGetLastError();
      return false;
    }
  }
  return true;
}

bool freeLocalAllocations(RelayRequest& state) {
  bool ok = true;
  for (size_t i = 0; i < state.staging.size();) {
    DeviceGuard helperGuard(state.helpers[i]);
    const hipError_t status = helperGuard.status() == hipSuccess
        ? hipFree(state.staging[i])
        : helperGuard.status();
    if (status == hipSuccess) {
      state.staging.erase(state.staging.begin() + i);
      state.helpers.erase(state.helpers.begin() + i);
    } else {
      setHipError("freeing helper staging failed", status);
      (void)hipGetLastError();
      ok = false;
      ++i;
    }
  }
  if (state.flags != nullptr) {
    DeviceGuard activeGuard(state.device);
    const hipError_t status = activeGuard.status() == hipSuccess
        ? hipFree(state.flags)
        : activeGuard.status();
    if (status == hipSuccess) {
      state.flags = nullptr;
    } else {
      setHipError("freeing relay flags failed", status);
      (void)hipGetLastError();
      ok = false;
    }
  }
  return ok;
}

bool openMapping(
    RelayRequest& state,
    const hipIpcMemHandle_t& handle,
    void** mapping,
    const char* failure) {
  void* opened = nullptr;
  const hipError_t status =
      hipIpcOpenMemHandle(&opened, handle, hipIpcMemLazyEnablePeerAccess);
  if (status != hipSuccess) {
    setHipError(failure, status);
    (void)hipGetLastError();
    return false;
  }
  state.imports.push_back({opened});
  *mapping = opened;
  return true;
}

bool appendProbeRegion(
    const void* base,
    size_t bytes,
    uint64_t nonce,
    uint64_t domain,
    std::vector<const unsigned char*>& addresses,
    std::vector<ProbeWord>& expected) {
  const std::vector<size_t> offsets = probeOffsets(bytes);
  for (size_t i = 0; i < offsets.size(); ++i) {
    addresses.push_back(static_cast<const unsigned char*>(base) + offsets[i]);
    expected.push_back(canary(nonce, domain, i));
  }
  return true;
}

} // namespace

struct HipIpcRelay {
  RelayRequest state{};
};

extern "C" {

uint32_t hipipc_relay_abi_version(void) {
  return kAbi;
}

int hipipc_relay_device_info(
    int device,
    char* bus,
    size_t buslen,
    char* arch,
    size_t archlen) {
  if (bus == nullptr || arch == nullptr || buslen == 0 || archlen == 0) {
    setError("device info needs output buffers");
    return -1;
  }
  DeviceGuard guard(device);
  if (guard.status() != hipSuccess) {
    setHipError("cannot select the requested HIP device", guard.status());
    return -1;
  }
  hipDeviceProp_t prop{};
  hipError_t status = hipGetDeviceProperties(&prop, device);
  if (status != hipSuccess) {
    setHipError("HIP device properties are unavailable", status);
    (void)hipGetLastError();
    return -1;
  }
  char busId[kBusBytes] = {};
  if (!getBusId(device, busId, sizeof(busId))) {
    return -1;
  }
  std::snprintf(bus, buslen, "%s", busId);
  std::snprintf(arch, archlen, "%s", prop.gcnArchName);
  return 0;
}

size_t hipipc_relay_descriptor_size(void) {
  return sizeof(Descriptor);
}

int hipipc_relay_create(
    int device,
    int rank,
    const int* helpers,
    int helper_count,
    void* input,
    void* output,
    size_t capacity,
    void** request) {
  if (request == nullptr) {
    setError("create needs a request output");
    return -1;
  }
  *request = nullptr;
  if (rank != 0 && rank != 1) {
    setError("HIPIPC relay supports exactly ranks 0 and 1");
    return -1;
  }
  if (device < 0 || helpers == nullptr || helper_count <= 0 ||
      helper_count > kMaxHelpers) {
    setError("HIPIPC relay needs 1-7 explicitly assigned helper devices");
    return -1;
  }
  if (input == nullptr || output == nullptr || input == output ||
      reinterpret_cast<uintptr_t>(input) % kAlignBytes != 0 ||
      reinterpret_cast<uintptr_t>(output) % kAlignBytes != 0) {
    setError(
        "HIPIPC relay needs distinct 16-byte-aligned input/output pointers");
    return -1;
  }
  if (capacity != kCapacityBytes || capacity % kAlignBytes != 0) {
    setError(
        "HIPIPC relay capacity must be exactly 144 MiB and 16-byte aligned");
    return -1;
  }

  DeviceGuard entryGuard(device);
  if (entryGuard.status() != hipSuccess) {
    setHipError("cannot select the active HIP device", entryGuard.status());
    return -1;
  }
  if (!isGfx950(device)) {
    setError("HIPIPC relay requires gfx950");
    return -1;
  }

  auto* relay = new (std::nothrow) HipIpcRelay();
  if (relay == nullptr) {
    setError("HIPIPC relay request allocation failed");
    return -1;
  }
  RelayRequest& state = relay->state;
  state.rank = rank;
  state.device = device;
  state.capacity = capacity;
  state.input = input;
  state.output = output;
  state.nonce = newNonce(relay);

  hipError_t status = hipMemGetAddressRange(
      &state.inputBase, &state.inputAllocationBytes, input);
  if (status != hipSuccess) {
    setHipError("resolving the input allocation base failed", status);
    (void)hipGetLastError();
    delete relay;
    return -1;
  }
  state.inputOffset = static_cast<size_t>(
      static_cast<unsigned char*>(input) -
      static_cast<unsigned char*>(state.inputBase));
  if (state.inputOffset > state.inputAllocationBytes ||
      capacity > state.inputAllocationBytes - state.inputOffset) {
    setError("input capacity exceeds its owning HIP allocation");
    delete relay;
    return -1;
  }
  void* outputBase = nullptr;
  size_t outputAllocationBytes = 0;
  status = hipMemGetAddressRange(&outputBase, &outputAllocationBytes, output);
  if (status != hipSuccess) {
    setHipError("resolving the output allocation base failed", status);
    (void)hipGetLastError();
    delete relay;
    return -1;
  }
  const size_t outputOffset = static_cast<size_t>(
      static_cast<unsigned char*>(output) -
      static_cast<unsigned char*>(outputBase));
  if (outputOffset > outputAllocationBytes ||
      capacity > outputAllocationBytes - outputOffset) {
    setError("output capacity exceeds its owning HIP allocation");
    delete relay;
    return -1;
  }

  for (int i = 0; i < helper_count; ++i) {
    if (helpers[i] < 0 || helpers[i] == device ||
        containsDevice(state.helpers, helpers[i])) {
      setError("helper devices must be distinct and exclude the active GPU");
      delete relay;
      return -1;
    }
    if (!isGfx950(helpers[i]) || !enablePeerPair(device, helpers[i])) {
      delete relay;
      return -1;
    }
    state.helpers.push_back(helpers[i]);
    if (!getBusId(
            helpers[i], state.helperBus[i].data(), state.helperBus[i].size())) {
      delete relay;
      return -1;
    }
  }

  status = hipExtMallocWithFlags(
      reinterpret_cast<void**>(&state.flags),
      sizeof(RegisteredRelayFlags),
      kAllocFlags);
  if (status != hipSuccess) {
    setHipError("allocating relay flags failed", status);
    (void)hipGetLastError();
    delete relay;
    return -1;
  }

  bool ok = writeCanarySites(
      state.input,
      state.capacity,
      state.nonce,
      1,
      &state.inputOriginal,
      &state.inputProbeCount);
  if (ok) {
    ok = writeCanarySites(
        state.flags,
        sizeof(RegisteredRelayFlags),
        state.nonce,
        2,
        nullptr,
        nullptr);
  }
  for (size_t h = 0; ok && h < state.helpers.size(); ++h) {
    DeviceGuard helperGuard(state.helpers[h]);
    if (helperGuard.status() != hipSuccess) {
      setHipError("cannot select a helper HIP device", helperGuard.status());
      ok = false;
      break;
    }
    void* staging = nullptr;
    status =
        hipExtMallocWithFlags(&staging, kStagingBytesPerHelper, kAllocFlags);
    if (status != hipSuccess) {
      setHipError("allocating helper staging failed", status);
      (void)hipGetLastError();
      ok = false;
      break;
    }
    state.staging.push_back(staging);
    ok = writeCanarySites(
        staging, kStagingBytesPerHelper, state.nonce, 3 + h, nullptr, nullptr);
  }
  if (!ok) {
    DeviceGuard activeGuard(device);
    (void)restoreInputCanaries(state);
    if (!freeLocalAllocations(state)) {
      *request = relay;
    } else {
      delete relay;
    }
    return -1;
  }

  *request = relay;
  return 0;
}

int hipipc_relay_export(void* request, void* descriptor, size_t length) {
  if (request == nullptr || descriptor == nullptr) {
    setError("export needs a request and a descriptor buffer");
    return -1;
  }
  if (length < sizeof(Descriptor)) {
    setError("descriptor buffer is too small");
    return -1;
  }
  auto* relay = static_cast<HipIpcRelay*>(request);
  RelayRequest& state = relay->state;
  DeviceGuard guard(state.device);
  if (guard.status() != hipSuccess) {
    setHipError("cannot select the active HIP device", guard.status());
    return -1;
  }

  Descriptor desc{};
  desc.magic = kDescriptorMagic;
  desc.abi = kAbi;
  desc.rank = static_cast<uint32_t>(state.rank);
  desc.capacity = state.capacity;
  desc.nonce = state.nonce;
  desc.inputOffset = state.inputOffset;
  desc.inputAllocationBytes = state.inputAllocationBytes;
  desc.helperCount = static_cast<uint32_t>(state.helpers.size());
  desc.inputProbeCount = state.inputProbeCount;
  std::copy(
      state.inputOriginal.begin(),
      state.inputOriginal.end(),
      std::begin(desc.inputOriginal));
  if (!getBusId(state.device, desc.activeBus, sizeof(desc.activeBus))) {
    return -1;
  }
  for (size_t h = 0; h < state.helpers.size(); ++h) {
    std::snprintf(
        desc.helperBus[h],
        sizeof(desc.helperBus[h]),
        "%s",
        state.helperBus[h].data());
  }

  hipError_t status = hipIpcGetMemHandle(&desc.inputBase, state.inputBase);
  if (status == hipSuccess) {
    status = hipIpcGetMemHandle(&desc.flags, state.flags);
  }
  if (status != hipSuccess) {
    setHipError("exporting active-device IPC allocations failed", status);
    (void)hipGetLastError();
    return -1;
  }
  for (size_t h = 0; h < state.staging.size(); ++h) {
    DeviceGuard helperGuard(state.helpers[h]);
    status = helperGuard.status() == hipSuccess
        ? hipIpcGetMemHandle(&desc.staging[h], state.staging[h])
        : helperGuard.status();
    if (status != hipSuccess) {
      setHipError("exporting helper staging IPC failed", status);
      (void)hipGetLastError();
      return -1;
    }
  }
  std::memcpy(descriptor, &desc, sizeof(desc));
  state.exported = true;
  return 0;
}

int hipipc_relay_import(
    void* request,
    const void* peer_descriptor,
    size_t length) {
  if (request == nullptr || peer_descriptor == nullptr) {
    setError("import needs a request and a peer descriptor");
    return -1;
  }
  if (length < sizeof(Descriptor)) {
    setError("peer descriptor is too small");
    return -1;
  }
  auto* relay = static_cast<HipIpcRelay*>(request);
  RelayRequest& state = relay->state;
  DeviceGuard guard(state.device);
  if (guard.status() != hipSuccess) {
    setHipError("cannot select the active HIP device", guard.status());
    return -1;
  }
  if (state.importAttempted) {
    setError("peer descriptor import may be attempted only once");
    return -1;
  }
  state.importAttempted = true;

  Descriptor desc{};
  std::memcpy(&desc, peer_descriptor, sizeof(desc));
  if (desc.magic != kDescriptorMagic || desc.abi != kAbi ||
      desc.rank == static_cast<uint32_t>(state.rank)) {
    setError("peer descriptor identity or ABI mismatch");
    return -1;
  }
  if (desc.capacity != state.capacity ||
      desc.helperCount != state.helpers.size() ||
      desc.inputOffset > desc.inputAllocationBytes ||
      desc.capacity > desc.inputAllocationBytes - desc.inputOffset ||
      desc.inputProbeCount != probeOffsets(desc.capacity).size()) {
    setError("peer descriptor capacity or route contract mismatch");
    return -1;
  }
  for (size_t h = 0; h < state.helpers.size(); ++h) {
    if (std::strncmp(desc.helperBus[h], state.helperBus[h].data(), kBusBytes) !=
        0) {
      setError("peer helper PCI identity/order mismatch");
      return -1;
    }
  }
  state.peerDevice = findDeviceByBus(desc.activeBus);
  if (state.peerDevice < 0 || state.peerDevice == state.device ||
      containsDevice(state.helpers, state.peerDevice) ||
      !enablePeerPair(state.device, state.peerDevice)) {
    setError("peer active GPU PCI identity is unavailable or inaccessible");
    return -1;
  }

  if (!openMapping(
          state,
          desc.inputBase,
          &state.peerInputBase,
          "opening the peer input allocation failed")) {
    return -1;
  }
  state.peerInput =
      static_cast<unsigned char*>(state.peerInputBase) + desc.inputOffset;
  void* peerFlags = nullptr;
  if (!openMapping(
          state, desc.flags, &peerFlags, "opening the peer flags failed")) {
    return -1;
  }
  state.peerFlags = static_cast<RegisteredRelayFlags*>(peerFlags);
  for (size_t h = 0; h < state.helpers.size(); ++h) {
    void* ring = nullptr;
    if (!openMapping(
            state,
            desc.staging[h],
            &ring,
            "opening peer helper staging failed")) {
      return -1;
    }
    state.peerStaging.push_back(ring);
  }
  std::copy(
      std::begin(desc.inputOriginal),
      std::end(desc.inputOriginal),
      state.peerInputOriginal.begin());
  state.peerNonce = desc.nonce;
  state.imported = true;
  return 0;
}

int hipipc_relay_probe(void* request, void* stream) {
  if (request == nullptr) {
    setError("probe needs a request");
    return -1;
  }
  auto* relay = static_cast<HipIpcRelay*>(request);
  RelayRequest& state = relay->state;
  DeviceGuard guard(state.device);
  if (guard.status() != hipSuccess) {
    setHipError("cannot select the active HIP device", guard.status());
    return -1;
  }
  if (!state.imported || state.probed || state.failStop) {
    setError("probe needs one complete, unprobed peer import");
    return -1;
  }

  std::vector<const unsigned char*> addresses;
  std::vector<ProbeWord> expected;
  appendProbeRegion(
      state.peerInput, state.capacity, state.peerNonce, 1, addresses, expected);
  appendProbeRegion(
      state.peerFlags,
      sizeof(RegisteredRelayFlags),
      state.peerNonce,
      2,
      addresses,
      expected);
  for (size_t h = 0; h < state.peerStaging.size(); ++h) {
    appendProbeRegion(
        state.peerStaging[h],
        kStagingBytesPerHelper,
        state.peerNonce,
        3 + h,
        addresses,
        expected);
  }

  const hipStream_t probeStream = static_cast<hipStream_t>(stream);
  const size_t pointerBytes = addresses.size() * sizeof(addresses[0]);
  const size_t resultBytes = expected.size() * sizeof(ProbeWord);
  const unsigned char** deviceAddresses = nullptr;
  ProbeWord* deviceResults = nullptr;
  std::vector<ProbeWord> results(expected.size());
  hipError_t status =
      hipMalloc(reinterpret_cast<void**>(&deviceAddresses), pointerBytes);
  if (status == hipSuccess) {
    status = hipMalloc(reinterpret_cast<void**>(&deviceResults), resultBytes);
  }
  if (status == hipSuccess) {
    status = hipMemcpyAsync(
        deviceAddresses,
        addresses.data(),
        pointerBytes,
        hipMemcpyHostToDevice,
        probeStream);
  }
  if (status == hipSuccess) {
    status = hipipcRelayProbeRead(
        deviceAddresses,
        static_cast<int>(addresses.size()),
        reinterpret_cast<unsigned long long*>(deviceResults),
        probeStream);
  }
  if (status == hipSuccess) {
    status = hipMemcpyAsync(
        results.data(),
        deviceResults,
        resultBytes,
        hipMemcpyDeviceToHost,
        probeStream);
  }
  if (status == hipSuccess) {
    status = hipStreamSynchronize(probeStream);
  }
  if (status != hipSuccess) {
    setHipError("shader-visible peer mapping probe failed", status);
    (void)hipGetLastError();
    (void)hipFree(deviceResults);
    (void)hipFree(deviceAddresses);
    state.failStop = true;
    return -1;
  }
  if (!std::equal(
          results.begin(), results.end(), expected.begin(), [](auto a, auto b) {
            return a.lo == b.lo && a.hi == b.hi;
          })) {
    setError("peer mapping probe returned a stale or aliased canary");
    (void)hipFree(deviceResults);
    (void)hipFree(deviceAddresses);
    state.failStop = true;
    return -1;
  }

  const std::vector<size_t> inputOffsets = probeOffsets(state.capacity);
  for (size_t i = 0; i < inputOffsets.size() && status == hipSuccess; ++i) {
    status = hipMemcpyAsync(
        static_cast<unsigned char*>(state.peerInput) + inputOffsets[i],
        &state.peerInputOriginal[i],
        kProbeBytes,
        hipMemcpyHostToDevice,
        probeStream);
  }
  if (status == hipSuccess) {
    status = hipMemsetAsync(
        state.peerFlags, 0, sizeof(RegisteredRelayFlags), probeStream);
  }
  if (status == hipSuccess) {
    status = hipStreamSynchronize(probeStream);
  }
  (void)hipFree(deviceResults);
  (void)hipFree(deviceAddresses);
  if (status != hipSuccess) {
    setHipError("restoring peer probe state failed", status);
    (void)hipGetLastError();
    state.failStop = true;
    return -1;
  }
  state.probed = true;
  return 0;
}

int hipipc_relay_execute(void* request, size_t bytes, void* stream) {
  if (request == nullptr) {
    setError("execute needs a request");
    return -1;
  }
  auto* relay = static_cast<HipIpcRelay*>(request);
  RelayRequest& state = relay->state;
  DeviceGuard guard(state.device);
  if (guard.status() != hipSuccess) {
    setHipError("cannot select the active HIP device", guard.status());
    return -1;
  }
  if (!state.probed || state.failStop) {
    setError("execute needs a successful probe and a healthy request");
    return -1;
  }
  if (bytes < kMinExecuteBytes || bytes > state.capacity ||
      bytes % kAlignBytes != 0) {
    setError("execute payload must be 2-144 MiB and 16-byte aligned");
    return -1;
  }
  const hipStream_t executionStream = static_cast<hipStream_t>(stream);
  if (state.streamBound && state.executionStream != executionStream) {
    setError("execution stream cannot change after the first execute");
    return -1;
  }
  if (!state.streamBound) {
    state.streamBound = true;
    state.executionStream = executionStream;
  }

  const RegisteredRelayLaunchPolicy policy{
      kSliceBytes, kSliceBytes, kDirectWeight, kLanes, kLanes, kSlots};
  const RegisteredRelayArgs args = makeRegisteredRelayArgs(
      state.input,
      state.peerInput,
      state.output,
      bytes,
      state.rank,
      static_cast<int>(state.helpers.size()),
      policy,
      state.staging.data(),
      state.peerStaging.data(),
      state.flags,
      state.peerFlags);

  const hipError_t status =
      launchRegisteredRelayAllReduce(args, executionStream);
  if (status != hipSuccess) {
    setHipError("canonical TP2 BF16 relay launch failed", status);
    (void)hipGetLastError();
    state.failStop = true;
    state.quiescenceKnown = false;
    return -1;
  }
  state.executionStarted = true;
  state.drained = false;
  return 0;
}

int hipipc_relay_drain(void* request, void* stream) {
  if (request == nullptr) {
    setError("drain needs a request");
    return -1;
  }
  auto* relay = static_cast<HipIpcRelay*>(request);
  RelayRequest& state = relay->state;
  DeviceGuard guard(state.device);
  if (guard.status() != hipSuccess) {
    setHipError("cannot select the active HIP device", guard.status());
    return -1;
  }
  const hipStream_t drainStream = static_cast<hipStream_t>(stream);
  if (state.streamBound && drainStream != state.executionStream) {
    setError("drain must use the bound execution stream");
    return -1;
  }
  if (!state.quiescenceKnown) {
    setError("request quiescence is unknown after a launch failure");
    return -1;
  }
  const hipError_t status = state.streamBound
      ? hipStreamSynchronize(state.executionStream)
      : hipDeviceSynchronize();
  if (status != hipSuccess) {
    setHipError("execution stream drain failed", status);
    (void)hipGetLastError();
    state.quiescenceKnown = false;
    state.failStop = true;
    return -1;
  }
  state.drained = true;
  return 0;
}

int hipipc_relay_close_imports(void* request) {
  if (request == nullptr) {
    return 0;
  }
  auto* relay = static_cast<HipIpcRelay*>(request);
  RelayRequest& state = relay->state;
  DeviceGuard guard(state.device);
  if (guard.status() != hipSuccess) {
    setHipError("cannot select the active HIP device", guard.status());
    return -1;
  }
  if (!state.quiescenceKnown || (state.executionStarted && !state.drained)) {
    setError("cannot close imports before known execution quiescence");
    return -1;
  }
  bool ok = true;
  for (size_t i = 0; i < state.imports.size();) {
    const hipError_t status = hipIpcCloseMemHandle(state.imports[i].pointer);
    if (status == hipSuccess) {
      state.imports.erase(state.imports.begin() + i);
    } else {
      setHipError("closing a peer IPC mapping failed", status);
      (void)hipGetLastError();
      ok = false;
      ++i;
    }
  }
  if (state.imports.empty()) {
    state.peerInputBase = nullptr;
    state.peerInput = nullptr;
    state.peerFlags = nullptr;
    state.peerStaging.clear();
    state.imported = false;
    state.closeCalled = true;
  }
  return ok ? 0 : -1;
}

int hipipc_relay_destroy(void* request) {
  if (request == nullptr) {
    return 0;
  }
  auto* relay = static_cast<HipIpcRelay*>(request);
  RelayRequest& state = relay->state;
  DeviceGuard guard(state.device);
  if (guard.status() != hipSuccess) {
    setHipError("cannot select the active HIP device", guard.status());
    return -1;
  }
  if (!state.imports.empty()) {
    setError("destroy refuses outstanding peer imports");
    return -1;
  }
  if (!state.quiescenceKnown || (state.executionStarted && !state.drained)) {
    setError("destroy refuses unknown or undrained execution quiescence");
    return -1;
  }
  if (state.exported && !state.closeCalled) {
    setError("destroy requires the coordinated close-imports phase");
    return -1;
  }
  if (!freeLocalAllocations(state)) {
    return -1;
  }
  delete relay;
  return 0;
}

const char* hipipc_relay_last_error(void) {
  return g_last_error.empty() ? "" : g_last_error.c_str();
}

} // extern "C"
