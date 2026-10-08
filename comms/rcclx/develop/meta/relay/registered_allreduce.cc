/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "meta/relay/registered_allreduce.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#include "archinfo.h"
#include "bootstrap.h"
#include "comm.h"
#include "debug.h"
#include "meta/relay/registered_allreduce_kernels.h"

namespace rcclx::relay {

namespace {

#if defined(HIP_UNCACHED_MEMORY)
constexpr unsigned int kStateAllocFlags = hipDeviceMallocUncached;
#else
constexpr unsigned int kStateAllocFlags = hipDeviceMallocFinegrained;
#endif

constexpr size_t kSentinelBytes = sizeof(uint64_t);
std::atomic<size_t> gLiveRequests{0};
std::atomic<size_t> gLivePeerMappings{0};
std::atomic<size_t> gLiveStateAllocations{0};
std::atomic<size_t> gPooledStateImports{0};
std::atomic<int> gFailIpcOpenRankForTest{-1};
std::atomic<int> gStaleReaderForTest{-1};
std::atomic<int> gStaleOwnerForTest{-1};
constexpr size_t kProbePage = size_t{2} << 20;

uint64_t sentinelFor(uint64_t commHash, int rank) {
  return (commHash ^ 0x9E3779B97F4A7C15ull) * 0x100000001B3ull +
      static_cast<uint64_t>(rank) + 1ull;
}

bool isAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15u) == 0;
}

struct ByteRange {
  const void* begin;
  size_t bytes;
};

bool overlaps(const ByteRange& first, const ByteRange& second) {
  const auto a = reinterpret_cast<uintptr_t>(first.begin);
  const auto b = reinterpret_cast<uintptr_t>(second.begin);
  return a < b + second.bytes && b < a + first.bytes;
}

bool validCapacity(size_t capacityBytes) {
  return capacityBytes > 0 && capacityBytes % 16 == 0;
}

bool supportedComm(ncclComm_t comm) {
  return comm != nullptr && isRegisteredAllReduceRankCount(comm->nRanks);
}

bool validEpsilon(float epsilon) {
  return std::isfinite(epsilon) && epsilon >= 0.0f;
}

// Validates the optional gated-residual-norm epilogue. Every written buffer
// (output, residualOut, routerOut) must not overlap any other operand, except
// that residualOut may alias residualIn exactly.
ncclResult_t toGatedResidualNormArgs(
    const void* input,
    const void* output,
    size_t count,
    const ncclRegisteredAllReduceGatedResidualNorm& norm,
    RegisteredAllReduceGatedResidualNormArgs& args) {
  constexpr size_t kNormCount = static_cast<size_t>(kRegisteredAllReduceRows) *
      kRegisteredAllReduceNormHidden;
  if (count != kNormCount ||
      norm.hiddenSize != kRegisteredAllReduceNormHidden) {
    return ncclInvalidArgument;
  }
  if (norm.residualIn == nullptr || norm.residualOut == nullptr ||
      norm.postNormWeight == nullptr || norm.preNormWeight == nullptr ||
      norm.gateAlpha == nullptr || norm.gateBeta == nullptr) {
    return ncclInvalidArgument;
  }
  if (!isAligned16(norm.residualIn) || !isAligned16(norm.residualOut) ||
      !isAligned16(norm.routerOut) || !isAligned16(norm.postNormWeight) ||
      !isAligned16(norm.preNormWeight) || !isAligned16(norm.gateAlpha) ||
      !isAligned16(norm.gateBeta)) {
    return ncclInvalidArgument;
  }
  if (!validEpsilon(norm.postNormEpsilon) ||
      !validEpsilon(norm.preNormEpsilon)) {
    return ncclInvalidArgument;
  }
  const size_t bf16Bytes = count * sizeof(__nv_bfloat16);
  const size_t floatBytes = count * sizeof(float);
  const size_t hidden = norm.hiddenSize;
  const ByteRange outputRange{output, bf16Bytes};
  const ByteRange residualOutRange{norm.residualOut, floatBytes};
  const ByteRange routerRange{norm.routerOut, floatBytes};
  const ByteRange reads[] = {
      {input, bf16Bytes},
      {norm.residualIn, floatBytes},
      {norm.postNormWeight, hidden * sizeof(__nv_bfloat16)},
      {norm.preNormWeight, hidden * sizeof(__nv_bfloat16)},
      {norm.gateAlpha, hidden * sizeof(float)},
      {norm.gateBeta, hidden * sizeof(float)},
  };
  const bool residualInPlace = norm.residualOut == norm.residualIn;
  for (size_t i = 0; i < std::size(reads); ++i) {
    const bool isResidualIn = i == 1;
    if (overlaps(outputRange, reads[i]) ||
        (norm.routerOut != nullptr && overlaps(routerRange, reads[i])) ||
        (!(isResidualIn && residualInPlace) &&
         overlaps(residualOutRange, reads[i]))) {
      return ncclInvalidArgument;
    }
  }
  if (overlaps(outputRange, residualOutRange) ||
      (norm.routerOut != nullptr &&
       (overlaps(routerRange, outputRange) ||
        overlaps(routerRange, residualOutRange)))) {
    return ncclInvalidArgument;
  }
  args.residualIn = norm.residualIn;
  args.residualOut = norm.residualOut;
  args.routerOut = norm.routerOut;
  args.postNormWeight = static_cast<const __nv_bfloat16*>(norm.postNormWeight);
  args.preNormWeight = static_cast<const __nv_bfloat16*>(norm.preNormWeight);
  args.gateAlpha = norm.gateAlpha;
  args.gateBeta = norm.gateBeta;
  args.postNormEpsilon = norm.postNormEpsilon;
  args.preNormEpsilon = norm.preNormEpsilon;
  return ncclSuccess;
}

ncclResult_t hipToNccl(hipError_t result, const char* what) {
  if (result == hipSuccess) {
    return ncclSuccess;
  }
  WARN("Registered all-reduce: %s failed: %s", what, hipGetErrorString(result));
  return ncclUnhandledCudaError;
}

ncclResult_t closeHandle(void*& handle) {
  if (handle == nullptr) {
    return ncclSuccess;
  }
  const hipError_t result = hipIpcCloseMemHandle(handle);
  if (result != hipSuccess) {
    return hipToNccl(result, "hipIpcCloseMemHandle");
  }
  handle = nullptr;
  gLivePeerMappings.fetch_sub(1, std::memory_order_relaxed);
  return ncclSuccess;
}

struct InitRecord {
  hipIpcMemHandle_t inputHandle;
  hipIpcMemHandle_t stateHandle;
  // Owner-assigned identity of the pooled state region; peers import each
  // identity once per communicator.
  uint64_t stateId{0};
  // IPC handles name the whole allocation and open at its base, so the
  // registered input's offset inside that allocation travels separately.
  size_t inputOffset{0};
  size_t capacityBytes{0};
  // Registration-unique value mixed into the input mapping probes.
  uint64_t probeNonce{0};
  uint32_t result{static_cast<uint32_t>(ncclInternalError)};
  uint32_t reserved{0};
};

struct InitVote {
  uint8_t ok{0};
};

} // namespace

// State regions belong to the communicator, not the request. Peers write
// handshake flags into them over IPC, and freeing one while the process keeps
// running was observed to corrupt a 4 KiB page of whatever allocation later
// reused its address (lost writes and wrong reads at the offset of its flag
// page). Regions are therefore reused by later requests, imported once by each
// peer, and freed only at ncclCommDestroy.
struct StatePool {
  struct Slot {
    RegisteredAllReduceStateRegion* state;
    hipIpcMemHandle_t handle;
    uint64_t id;
    bool busy;
  };
  struct Import {
    int peer;
    uint64_t id;
    void* mapping;
  };
  ncclComm_t comm{nullptr};
  uint64_t commHash{0};
  std::vector<Slot> slots;
  std::vector<Import> imports;
  uint64_t nextId{0};
};

struct RegisteredAllReduce {
  ncclComm_t comm{nullptr};
  uint64_t commHash{0};
  int rank{-1};
  int nRanks{0};
  const void* inputBase{nullptr};
  void* outputBase{nullptr};
  size_t capacityBytes{0};
  size_t peerMinCapacityBytes{0};
  RegisteredAllReduceStateRegion* localState{nullptr};
  StatePool* pool{nullptr};
  int slot{-1};
  void* mappedInput[kRegisteredAllReduceRanks]{};
  RegisteredAllReduceInputTable inputTable{};
  RegisteredAllReduceStateTable stateTable{};
  bool mappingsMayExist{false};
  bool initialized{false};
};

namespace {

std::mutex& statePoolMutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<std::unique_ptr<StatePool>>& statePools() {
  static std::vector<std::unique_ptr<StatePool>> pools;
  return pools;
}

// Pools are keyed on the comm pointer and its hash (comm pointers are
// recycled by the allocator).
StatePool* statePoolFor(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(statePoolMutex());
  for (const auto& pool : statePools()) {
    if (pool->comm == comm && pool->commHash == comm->commHash) {
      return pool.get();
    }
  }
  auto pool = std::make_unique<StatePool>();
  pool->comm = comm;
  pool->commHash = comm->commHash;
  pool->nextId = (static_cast<uint64_t>(comm->rank) << 48) + 1;
  statePools().push_back(std::move(pool));
  return statePools().back().get();
}

struct RegisteredAllReducePublicRequest {
  RegisteredAllReduce* request{nullptr};
  ncclComm_t comm{nullptr};
  uint64_t commHash{0};
  bool finalizing{false};
  RegisteredAllReducePublicRequest* next{nullptr};
};

std::mutex& publicRegistryMutex() {
  static std::mutex mutex;
  return mutex;
}

RegisteredAllReducePublicRequest*& publicRegistryHead() {
  static RegisteredAllReducePublicRequest* head = nullptr;
  return head;
}

RegisteredAllReducePublicRequest* findPublicRequestLocked(void* handle) {
  auto* const want = static_cast<RegisteredAllReducePublicRequest*>(handle);
  for (RegisteredAllReducePublicRequest* cur = publicRegistryHead();
       cur != nullptr;
       cur = cur->next) {
    if (cur == want) {
      return cur;
    }
  }
  return nullptr;
}

void registerPublicRequest(RegisteredAllReducePublicRequest* entry) {
  std::lock_guard<std::mutex> lock(publicRegistryMutex());
  entry->next = publicRegistryHead();
  publicRegistryHead() = entry;
}

bool unregisterPublicRequestLocked(RegisteredAllReducePublicRequest* entry) {
  for (RegisteredAllReducePublicRequest** cur = &publicRegistryHead();
       *cur != nullptr;
       cur = &(*cur)->next) {
    if (*cur == entry) {
      *cur = entry->next;
      entry->next = nullptr;
      return true;
    }
  }
  return false;
}

ncclResult_t rejectActiveCapture(hipStream_t stream) {
#if ROCM_VERSION >= 60100
  if (stream == nullptr) {
    return ncclSuccess;
  }
  hipStreamCaptureStatus status;
  unsigned long long graphId = 0;
  hipGraph_t graph = nullptr;
  const hipError_t hipResult = hipStreamGetCaptureInfo_v2(
      stream, &status, &graphId, &graph, nullptr, nullptr);
  if (hipResult != hipSuccess) {
    return hipToNccl(hipResult, "hipStreamGetCaptureInfo_v2(finalize)");
  }
  if (status == hipStreamCaptureStatusActive) {
    WARN("Registered all-reduce: finalize is not valid during stream capture");
    return ncclInvalidUsage;
  }
#endif
  return ncclSuccess;
}

ncclResult_t collectPublicResult(ncclComm_t comm, ncclResult_t localResult) {
  std::vector<uint32_t> results(comm->nRanks, 0);
  results[comm->rank] = static_cast<uint32_t>(localResult);
  const ncclResult_t gatherResult =
      bootstrapAllGather(comm->bootstrap, results.data(), sizeof(uint32_t));
  if (gatherResult != ncclSuccess) {
    return gatherResult;
  }
  for (uint32_t result : results) {
    if (result != static_cast<uint32_t>(ncclSuccess)) {
      return static_cast<ncclResult_t>(result);
    }
  }
  return ncclSuccess;
}

ncclResult_t firstInitResult(const std::vector<InitRecord>& records) {
  for (const InitRecord& record : records) {
    if (record.result == static_cast<uint32_t>(ncclInvalidArgument)) {
      return ncclInvalidArgument;
    }
  }
  for (const InitRecord& record : records) {
    if (record.result != static_cast<uint32_t>(ncclSuccess)) {
      return static_cast<ncclResult_t>(record.result);
    }
  }
  return ncclSuccess;
}

// Closes this request's imports of peers' inputs. Pooled state imports stay
// open for the communicator's lifetime.
ncclResult_t closePeerMappings(RegisteredAllReduce& request) {
  ncclResult_t result = ncclSuccess;
  for (int peer = 0; peer < request.nRanks; ++peer) {
    if (peer == request.rank) {
      continue;
    }
    const ncclResult_t inputResult = closeHandle(request.mappedInput[peer]);
    if (result == ncclSuccess && inputResult != ncclSuccess) {
      result = inputResult;
    }
  }
  return result;
}

ncclResult_t releaseLocalState(RegisteredAllReduce& request) {
  if (request.localState != nullptr) {
    request.pool->slots[request.slot].busy = false;
    request.localState = nullptr;
    request.slot = -1;
    gLiveStateAllocations.fetch_sub(1, std::memory_order_relaxed);
  }
  request.inputBase = nullptr;
  request.outputBase = nullptr;
  request.capacityBytes = 0;
  request.peerMinCapacityBytes = 0;
  request.inputTable = RegisteredAllReduceInputTable{};
  request.stateTable = RegisteredAllReduceStateTable{};
  request.mappingsMayExist = false;
  request.initialized = false;
  return ncclSuccess;
}

ncclResult_t closeMappingsCollectively(RegisteredAllReduce& request) {
  const ncclResult_t localResult = closePeerMappings(request);
  std::vector<uint32_t> results(request.nRanks, 0);
  results[request.rank] = static_cast<uint32_t>(localResult);
  const ncclResult_t gatherResult = bootstrapAllGather(
      request.comm->bootstrap, results.data(), sizeof(uint32_t));
  if (gatherResult != ncclSuccess) {
    return gatherResult;
  }
  for (uint32_t result : results) {
    if (result != static_cast<uint32_t>(ncclSuccess)) {
      return static_cast<ncclResult_t>(result);
    }
  }
  request.mappingsMayExist = false;
  request.initialized = false;
  return releaseLocalState(request);
}

ncclResult_t validateTopology(const RegisteredAllReduce& request) {
  ncclComm_t comm = request.comm;
  if (!supportedComm(comm) || comm->nNodes != 1 ||
      comm->localRanks != comm->nRanks || !comm->peerInfoValid ||
      comm->archName == nullptr || !IsArchMatch(comm->archName, "gfx950") ||
      !comm->isAllDirectP2p) {
    WARN(
        "Registered all-reduce requires two or four local gfx950 ranks on one node with direct all-pairs P2P; got nRanks=%d nNodes=%d localRanks=%d arch=%s directP2p=%d peerInfoValid=%d",
        comm == nullptr ? -1 : comm->nRanks,
        comm == nullptr ? -1 : comm->nNodes,
        comm == nullptr ? -1 : comm->localRanks,
        comm == nullptr || comm->archName == nullptr ? "<null>"
                                                     : comm->archName,
        comm == nullptr ? 0 : static_cast<int>(comm->isAllDirectP2p),
        comm == nullptr ? 0 : static_cast<int>(comm->peerInfoValid));
    return ncclInvalidArgument;
  }

  // isAllDirectP2p is RCCLX's communicator-wide runtime topology check. The
  // later IPC opens and sentinel reads additionally prove that each registered
  // allocation is mapped and readable from every peer.
  return ncclSuccess;
}

ncclResult_t validateInitArgs(
    const RegisteredAllReduce& request,
    const void* inputBase,
    void* outputBase,
    size_t capacityBytes) {
  if (request.initialized || inputBase == nullptr || outputBase == nullptr ||
      inputBase == outputBase) {
    return ncclInvalidArgument;
  }
  if (!isAligned16(inputBase) || !isAligned16(outputBase)) {
    return ncclInvalidArgument;
  }
  if (!validCapacity(capacityBytes)) {
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

ncclResult_t exportInput(
    const RegisteredAllReduce& request,
    InitRecord& record) {
  void* allocationBase = nullptr;
  size_t allocationBytes = 0;
  hipError_t hipResult = hipMemGetAddressRange(
      &allocationBase, &allocationBytes, const_cast<void*>(request.inputBase));
  if (hipResult != hipSuccess) {
    return hipToNccl(hipResult, "hipMemGetAddressRange(input)");
  }
  const size_t offset = static_cast<size_t>(
      static_cast<const char*>(request.inputBase) -
      static_cast<const char*>(allocationBase));
  if (offset > allocationBytes ||
      request.capacityBytes > allocationBytes - offset) {
    WARN(
        "Registered all-reduce: input capacity %zu at offset %zu exceeds its %zu-byte allocation",
        request.capacityBytes,
        offset,
        allocationBytes);
    return ncclInvalidArgument;
  }
  record.inputOffset = offset;

  hipResult = hipIpcGetMemHandle(&record.inputHandle, allocationBase);
  if (hipResult != hipSuccess) {
    return hipToNccl(hipResult, "hipIpcGetMemHandle(input)");
  }
  return ncclSuccess;
}

ncclResult_t openPeerInput(
    const InitRecord& record,
    void*& mapping,
    const __nv_bfloat16*& input) {
  const hipError_t hipResult = hipIpcOpenMemHandle(
      &mapping, record.inputHandle, hipIpcMemLazyEnablePeerAccess);
  if (hipResult != hipSuccess) {
    mapping = nullptr;
    return hipToNccl(hipResult, "hipIpcOpenMemHandle(input)");
  }
  gLivePeerMappings.fetch_add(1, std::memory_order_relaxed);
  input = reinterpret_cast<const __nv_bfloat16*>(
      static_cast<const char*>(mapping) + record.inputOffset);
  return ncclSuccess;
}

// Binds a free pooled state region to the request, allocating one if every
// region is in use. Finalize drained every kernel of a region's previous
// request on all ranks, and peers write to it only after Init completes, so
// resetting it here is safe.
ncclResult_t allocateLocalState(
    RegisteredAllReduce& request,
    InitRecord& record) {
  StatePool& pool = *request.pool;
  int slot = -1;
  for (size_t k = 0; k < pool.slots.size(); ++k) {
    if (!pool.slots[k].busy) {
      slot = static_cast<int>(k);
      break;
    }
  }
  hipError_t hipResult = hipSuccess;
  if (slot < 0) {
    StatePool::Slot fresh{};
    hipResult = hipExtMallocWithFlags(
        reinterpret_cast<void**>(&fresh.state),
        sizeof(RegisteredAllReduceStateRegion),
        kStateAllocFlags);
    if (hipResult != hipSuccess) {
      return hipToNccl(hipResult, "hipExtMallocWithFlags(state)");
    }
    hipResult = hipIpcGetMemHandle(&fresh.handle, fresh.state);
    if (hipResult != hipSuccess) {
      (void)hipFree(fresh.state);
      return hipToNccl(hipResult, "hipIpcGetMemHandle(state)");
    }
    fresh.id = pool.nextId++;
    pool.slots.push_back(fresh);
    slot = static_cast<int>(pool.slots.size()) - 1;
  }
  pool.slots[slot].busy = true;
  request.slot = slot;
  request.localState = pool.slots[slot].state;
  gLiveStateAllocations.fetch_add(1, std::memory_order_relaxed);
  record.stateHandle = pool.slots[slot].handle;
  record.stateId = pool.slots[slot].id;

  hipResult =
      hipMemset(request.localState, 0, sizeof(RegisteredAllReduceStateRegion));
  if (hipResult != hipSuccess) {
    return hipToNccl(hipResult, "hipMemset(state)");
  }

  const uint64_t sentinel = sentinelFor(request.commHash, request.rank);
  hipResult = hipMemcpy(
      &request.localState->sentinel,
      &sentinel,
      kSentinelBytes,
      hipMemcpyHostToDevice);
  if (hipResult != hipSuccess) {
    return hipToNccl(hipResult, "hipMemcpy(state sentinel)");
  }

  return exportInput(request, record);
}

// The communicator-lifetime import of `peer`'s pooled state region `id`.
ncclResult_t importPeerState(
    StatePool& pool,
    int peer,
    const InitRecord& record,
    void** mapping) {
  for (const StatePool::Import& entry : pool.imports) {
    if (entry.peer == peer && entry.id == record.stateId) {
      *mapping = entry.mapping;
      return ncclSuccess;
    }
  }
  void* opened = nullptr;
  const hipError_t hipResult = hipIpcOpenMemHandle(
      &opened, record.stateHandle, hipIpcMemLazyEnablePeerAccess);
  if (hipResult != hipSuccess) {
    return hipToNccl(hipResult, "hipIpcOpenMemHandle(state)");
  }
  pool.imports.push_back({peer, record.stateId, opened});
  gPooledStateImports.fetch_add(1, std::memory_order_relaxed);
  *mapping = opened;
  return ncclSuccess;
}

ncclResult_t mapPeers(
    RegisteredAllReduce& request,
    const std::vector<InitRecord>& records) {
  request.inputTable.input[request.rank] =
      reinterpret_cast<const __nv_bfloat16*>(request.inputBase);
  request.stateTable.state[request.rank] = request.localState;

  for (int peer = 0; peer < request.nRanks; ++peer) {
    if (peer == request.rank) {
      continue;
    }
    if (gFailIpcOpenRankForTest.load(std::memory_order_relaxed) ==
            request.rank &&
        peer == 2) {
      return ncclInternalError;
    }

    const ncclResult_t inputResult = openPeerInput(
        records[peer],
        request.mappedInput[peer],
        request.inputTable.input[peer]);
    if (inputResult != ncclSuccess) {
      return inputResult;
    }

    void* peerState = nullptr;
    const ncclResult_t stateResult =
        importPeerState(*request.pool, peer, records[peer], &peerState);
    if (stateResult != ncclSuccess) {
      return stateResult;
    }
    request.stateTable.state[peer] =
        static_cast<RegisteredAllReduceStateRegion*>(peerState);

    uint64_t peerSentinel = 0;
    hipError_t hipResult = hipMemcpy(
        &peerSentinel,
        &request.stateTable.state[peer]->sentinel,
        kSentinelBytes,
        hipMemcpyDeviceToHost);
    if (hipResult != hipSuccess) {
      return hipToNccl(hipResult, "hipMemcpy(peer sentinel)");
    }
    const uint64_t wantSentinel = sentinelFor(request.commHash, peer);
    if (peerSentinel != wantSentinel) {
      WARN(
          "Registered all-reduce: mapping for peer %d has sentinel %llx, expected %llx",
          peer,
          static_cast<unsigned long long>(peerSentinel),
          static_cast<unsigned long long>(wantSentinel));
      return ncclInternalError;
    }
  }

  return ncclSuccess;
}

ncclResult_t allRanksResult(ncclComm_t comm, ncclResult_t local) {
  std::vector<uint32_t> results(comm->nRanks, 0);
  results[comm->rank] = static_cast<uint32_t>(local);
  const ncclResult_t gathered =
      bootstrapAllGather(comm->bootstrap, results.data(), sizeof(uint32_t));
  if (gathered != ncclSuccess) {
    return gathered;
  }
  for (const uint32_t result : results) {
    if (result != static_cast<uint32_t>(ncclSuccess)) {
      return static_cast<ncclResult_t>(result);
    }
  }
  return ncclSuccess;
}

uint64_t probeValue(uint64_t nonce, int rank, uint64_t tag) {
  uint64_t x = nonce ^ (uint64_t(rank + 1) * 0x9e3779b97f4a7c15ull) ^ tag;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x ^= x >> 29;
  return x;
}

// One 16-byte probe of `owner`'s input: its byte offset, where this rank
// reads it, and the pattern the owner wrote there.
struct ProbeSite {
  int owner;
  size_t offset;
  unsigned char* address;
  uint64_t expected[2];
};

// Every 2 MiB page of `owner`'s registered input (the granularity at which
// stale imports were observed) and its last 16 bytes, addressed from this rank.
std::vector<ProbeSite> probeSitesOf(
    const RegisteredAllReduce& request,
    const std::vector<InitRecord>& records,
    int owner) {
  const size_t bytes = records[owner].capacityBytes;
  unsigned char* base = reinterpret_cast<unsigned char*>(
      const_cast<__nv_bfloat16*>(request.inputTable.input[owner]));
  std::vector<ProbeSite> sites;
  auto add = [&](size_t offset) {
    const uint64_t nonce = records[owner].probeNonce;
    sites.push_back(
        {owner,
         offset,
         base + offset,
         {probeValue(nonce, owner, offset),
          probeValue(nonce, owner, offset ^ 0x5555555555555555ull)}});
  };
  for (size_t offset = 0; offset + 16 <= bytes; offset += kProbePage) {
    add(offset);
  }
  if (bytes >= 16 && (sites.empty() || sites.back().offset != bytes - 16)) {
    add(bytes - 16);
  }
  return sites;
}

// Reads every site through a GPU kernel that loads the way the all-reduce
// kernels do (the copy engine and the shader can translate a mapping
// differently). Returns the 16 bytes read per site.
ncclResult_t readSitesOnDevice(
    const std::vector<ProbeSite>& sites,
    std::vector<uint64_t>& values) {
  values.assign(2 * sites.size(), 0);
  if (sites.empty()) {
    return ncclSuccess;
  }
  std::vector<const unsigned char*> addresses;
  addresses.reserve(sites.size());
  for (const ProbeSite& site : sites) {
    addresses.push_back(site.address);
  }
  void* deviceAddresses = nullptr;
  void* deviceValues = nullptr;
  hipStream_t stream = nullptr;
  hipError_t result =
      hipMalloc(&deviceAddresses, addresses.size() * sizeof(addresses[0]));
  if (result == hipSuccess) {
    result = hipMalloc(&deviceValues, values.size() * sizeof(uint64_t));
  }
  if (result == hipSuccess) {
    result = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  }
  if (result == hipSuccess) {
    result = hipMemcpyAsync(
        deviceAddresses,
        addresses.data(),
        addresses.size() * sizeof(addresses[0]),
        hipMemcpyHostToDevice,
        stream);
  }
  if (result == hipSuccess) {
    result = launchRegisteredAllReduceProbe(
        static_cast<const unsigned char* const*>(deviceAddresses),
        static_cast<int>(addresses.size()),
        static_cast<uint64_t*>(deviceValues),
        stream);
  }
  if (result == hipSuccess) {
    result = hipMemcpyAsync(
        values.data(),
        deviceValues,
        values.size() * sizeof(uint64_t),
        hipMemcpyDeviceToHost,
        stream);
  }
  if (result == hipSuccess) {
    result = hipStreamSynchronize(stream);
  }
  if (stream != nullptr) {
    (void)hipStreamDestroy(stream);
  }
  (void)hipFree(deviceValues);
  (void)hipFree(deviceAddresses);
  return hipToNccl(result, "registered all-reduce mapping probe");
}

// Proves that every peer-input mapping reaches the owner's current memory: a
// fresh IPC import of a re-used address was observed to resolve part of it to
// an earlier allocation's pages, from byte 2 MiB on. Owners write
// registration-unique patterns at every 2 MiB page of their input (restored
// afterwards); every rank reads its peers' patterns through its mappings on
// the GPU. Every device stream is drained first, since a producer of the input
// may still be in flight on a non-blocking stream. Collective; returns the
// agreed result.
ncclResult_t verifyPeerInputs(
    RegisteredAllReduce& request,
    const std::vector<InitRecord>& records) {
  ncclComm_t comm = request.comm;
  const std::vector<ProbeSite> own =
      probeSitesOf(request, records, request.rank);
  std::vector<uint64_t> saved(2 * own.size(), 0);
  ncclResult_t local =
      hipToNccl(hipDeviceSynchronize(), "hipDeviceSynchronize(probe)");
  for (size_t k = 0; k < own.size() && local == ncclSuccess; ++k) {
    local = hipToNccl(
        hipMemcpy(&saved[2 * k], own[k].address, 16, hipMemcpyDefault),
        "hipMemcpy(save probe)");
    if (local == ncclSuccess) {
      local = hipToNccl(
          hipMemcpy(own[k].address, own[k].expected, 16, hipMemcpyDefault),
          "hipMemcpy(write probe)");
    }
  }
  if (local == ncclSuccess) {
    local = hipToNccl(hipDeviceSynchronize(), "hipDeviceSynchronize(probe)");
  }
  ncclResult_t result = allRanksResult(comm, local);

  if (result == ncclSuccess) {
    std::vector<ProbeSite> theirs;
    for (int peer = 0; peer < request.nRanks; ++peer) {
      if (peer != request.rank) {
        const std::vector<ProbeSite> sites =
            probeSitesOf(request, records, peer);
        theirs.insert(theirs.end(), sites.begin(), sites.end());
      }
    }
    std::vector<uint64_t> values;
    local = readSitesOnDevice(theirs, values);
    for (size_t k = 0; k < theirs.size() && local == ncclSuccess; ++k) {
      const ProbeSite& site = theirs[k];
      if (site.offset >= kProbePage &&
          request.rank == gStaleReaderForTest.load() &&
          site.owner == gStaleOwnerForTest.load()) {
        values[2 * k] = ~values[2 * k];
      }
      if (values[2 * k] != site.expected[0] ||
          values[2 * k + 1] != site.expected[1]) {
        WARN(
            "Registered all-reduce: rank %d's mapping of rank %d's input returns stale data at byte %zu; a fresh import of a re-used address resolved to earlier pages. Register input buffers that stay allocated for the process lifetime.",
            request.rank,
            site.owner,
            site.offset);
        local = ncclSystemError;
      }
    }
    result = allRanksResult(comm, local);
  }

  // Every peer has finished reading; restore the owner's input bytes.
  ncclResult_t restored = ncclSuccess;
  for (size_t k = 0; k < own.size(); ++k) {
    const ncclResult_t e = hipToNccl(
        hipMemcpy(own[k].address, &saved[2 * k], 16, hipMemcpyDefault),
        "hipMemcpy(restore probe)");
    restored = restored == ncclSuccess ? e : restored;
  }
  if (restored == ncclSuccess) {
    restored =
        hipToNccl(hipDeviceSynchronize(), "hipDeviceSynchronize(restore)");
  }
  const ncclResult_t restoredAll = allRanksResult(comm, restored);
  return result != ncclSuccess ? result : restoredAll;
}

ncclResult_t verifyGlobalQuiescence(
    RegisteredAllReduce& request,
    hipStream_t stream) {
  const ncclResult_t local =
      hipToNccl(hipStreamSynchronize(stream), "hipStreamSynchronize(finalize)");

  std::vector<uint32_t> states(request.nRanks, 0);
  states[request.rank] = static_cast<uint32_t>(local);
  ncclResult_t result = bootstrapAllGather(
      request.comm->bootstrap, states.data(), sizeof(uint32_t));
  if (result != ncclSuccess) {
    return result;
  }
  for (uint32_t state : states) {
    if (state != static_cast<uint32_t>(ncclSuccess)) {
      return static_cast<ncclResult_t>(state);
    }
  }

  return ncclSuccess;
}

} // namespace

ncclResult_t registeredAllReducePrepare(
    ncclComm_t comm,
    RegisteredAllReduce** request) {
  if (request == nullptr || !supportedComm(comm)) {
    return ncclInvalidArgument;
  }
  *request = new (std::nothrow) RegisteredAllReduce;
  if (*request == nullptr) {
    return ncclInternalError;
  }
  (*request)->comm = comm;
  (*request)->commHash = comm->commHash;
  (*request)->rank = comm->rank;
  (*request)->nRanks = comm->nRanks;
  gLiveRequests.fetch_add(1, std::memory_order_relaxed);
  return ncclSuccess;
}

ncclResult_t registeredAllReduceInit(
    RegisteredAllReduce* request,
    const void* inputBase,
    void* outputBase,
    size_t capacityBytes) {
  if (request == nullptr || !supportedComm(request->comm) ||
      request->comm->nRanks != request->nRanks) {
    return ncclInvalidArgument;
  }
  if (request->initialized || request->mappingsMayExist ||
      request->localState != nullptr) {
    return ncclInvalidUsage;
  }

  request->inputBase = inputBase;
  request->outputBase = outputBase;
  request->capacityBytes = capacityBytes;
  request->pool = statePoolFor(request->comm);

  InitRecord mine{};
  ncclResult_t localResult =
      validateInitArgs(*request, inputBase, outputBase, capacityBytes);
  if (localResult == ncclSuccess) {
    localResult = validateTopology(*request);
  }
  if (localResult == ncclSuccess) {
    localResult = allocateLocalState(*request, mine);
  }
  mine.capacityBytes = capacityBytes;
  static std::atomic<uint64_t> registrations{0};
  mine.probeNonce =
      request->commHash ^
      (registrations.fetch_add(1, std::memory_order_relaxed) + 1) *
          0xd1b54a32d192ed03ull ^
      static_cast<uint64_t>(
          std::chrono::steady_clock::now().time_since_epoch().count());
  mine.result = static_cast<uint32_t>(localResult);

  std::vector<InitRecord> records(request->nRanks);
  records[request->rank] = mine;
  ncclResult_t result = bootstrapAllGather(
      request->comm->bootstrap, records.data(), sizeof(InitRecord));
  if (result != ncclSuccess) {
    const ncclResult_t cleanupResult = releaseLocalState(*request);
    return cleanupResult == ncclSuccess ? result : cleanupResult;
  }

  result = firstInitResult(records);
  size_t peerMinCapacityBytes = records[0].capacityBytes;
  for (const InitRecord& record : records) {
    peerMinCapacityBytes = std::min(peerMinCapacityBytes, record.capacityBytes);
    if (!validCapacity(record.capacityBytes)) {
      result = ncclInvalidArgument;
    }
  }
  if (result != ncclSuccess) {
    const ncclResult_t cleanupResult = releaseLocalState(*request);
    return cleanupResult == ncclSuccess ? result : cleanupResult;
  }

  request->mappingsMayExist = true;
  result = mapPeers(*request, records);
  std::vector<InitVote> votes(request->nRanks);
  votes[request->rank].ok = result == ncclSuccess ? 1 : 0;
  const ncclResult_t voteResult = bootstrapAllGather(
      request->comm->bootstrap, votes.data(), sizeof(InitVote));
  if (voteResult != ncclSuccess) {
    const ncclResult_t closeResult = closePeerMappings(*request);
    return closeResult == ncclSuccess ? voteResult : closeResult;
  }
  for (const InitVote& vote : votes) {
    if (vote.ok == 0) {
      const ncclResult_t cleanupResult = closeMappingsCollectively(*request);
      return cleanupResult == ncclSuccess ? ncclInternalError : cleanupResult;
    }
  }
  result = verifyPeerInputs(*request, records);
  if (result != ncclSuccess) {
    const ncclResult_t cleanupResult = closeMappingsCollectively(*request);
    return cleanupResult == ncclSuccess ? result : cleanupResult;
  }

  request->peerMinCapacityBytes = peerMinCapacityBytes;
  request->initialized = true;
  return ncclSuccess;
}

ncclResult_t registeredAllReduceExecute(
    RegisteredAllReduce* request,
    const void* input,
    void* output,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    const ncclRegisteredAllReduceGatedResidualNorm* norm,
    hipStream_t stream) {
  if (request == nullptr || !request->initialized || input == nullptr ||
      output == nullptr) {
    return ncclInvalidArgument;
  }
  if (request->comm == nullptr ||
      request->commHash != request->comm->commHash ||
      request->comm->nRanks != request->nRanks ||
      request->rank != request->comm->rank) {
    return ncclInvalidArgument;
  }
  if (input != request->inputBase || output != request->outputBase ||
      input == output) {
    return ncclInvalidArgument;
  }
  if (datatype != ncclBfloat16 || op != ncclSum) {
    return ncclInvalidArgument;
  }
  if (count == 0 || count > SIZE_MAX / sizeof(__nv_bfloat16) ||
      !validCapacity(count * sizeof(__nv_bfloat16))) {
    return ncclInvalidArgument;
  }
  if (norm != nullptr && request->nRanks != kRegisteredAllReduceRanks) {
    return ncclInvalidArgument;
  }
  const size_t bytes = count * sizeof(__nv_bfloat16);
  if (bytes > request->capacityBytes || bytes > request->peerMinCapacityBytes) {
    return ncclInvalidArgument;
  }
  RegisteredAllReduceGatedResidualNormArgs normArgs{};
  if (norm != nullptr) {
    const ncclResult_t normResult =
        toGatedResidualNormArgs(input, output, count, *norm, normArgs);
    if (normResult != ncclSuccess) {
      return normResult;
    }
  }

  return hipToNccl(
      launchRegisteredAllReduceKernel(
          output,
          request->inputTable,
          request->stateTable,
          request->rank,
          count,
          norm == nullptr ? nullptr : &normArgs,
          stream,
          request->nRanks),
      "registered all-reduce kernel launch");
}

ncclResult_t registeredAllReduceFinalize(
    RegisteredAllReduce* request,
    hipStream_t stream,
    bool graphsTeardownComplete) {
  if (request == nullptr) {
    return ncclInvalidArgument;
  }
  if (!graphsTeardownComplete) {
    return ncclInvalidUsage;
  }

  ncclResult_t localResult = ncclSuccess;
  if (request->initialized) {
    const ncclResult_t quiesced = verifyGlobalQuiescence(*request, stream);
    if (quiesced != ncclSuccess) {
      return quiesced;
    }
    localResult = closeMappingsCollectively(*request);
  } else if (request->mappingsMayExist) {
    localResult = closeMappingsCollectively(*request);
  } else {
    localResult = releaseLocalState(*request);
  }

  const ncclResult_t result = collectPublicResult(request->comm, localResult);
  if (result != ncclSuccess) {
    return result;
  }

  delete request;
  gLiveRequests.fetch_sub(1, std::memory_order_relaxed);
  return ncclSuccess;
}

void registeredAllReduceReleaseComm(ncclComm_t comm) {
  std::unique_ptr<StatePool> pool;
  {
    std::lock_guard<std::mutex> lock(statePoolMutex());
    auto& pools = statePools();
    for (auto it = pools.begin(); it != pools.end(); ++it) {
      if ((*it)->comm == comm && (*it)->commHash == comm->commHash) {
        pool = std::move(*it);
        pools.erase(it);
        break;
      }
    }
  }
  if (pool == nullptr) {
    return;
  }
  for (const StatePool::Import& entry : pool->imports) {
    (void)hipIpcCloseMemHandle(entry.mapping);
    gPooledStateImports.fetch_sub(1, std::memory_order_relaxed);
  }
  for (const StatePool::Slot& slot : pool->slots) {
    (void)hipFree(slot.state);
  }
}

ncclResult_t registeredAllReducePublicInit(
    const void* sendbuff,
    void* recvbuff,
    size_t capacityBytes,
    ncclComm_t comm,
    void** request) {
  if (request != nullptr) {
    *request = nullptr;
  }
  if (!supportedComm(comm)) {
    return ncclInvalidArgument;
  }

  auto* entry = request == nullptr ? nullptr
                                   : new (std::nothrow)
                                         RegisteredAllReducePublicRequest;
  RegisteredAllReduce* internalRequest = nullptr;
  ncclResult_t localResult = request == nullptr ? ncclInvalidArgument
      : entry == nullptr                        ? ncclInternalError
                         : registeredAllReducePrepare(comm, &internalRequest);
  ncclResult_t result = collectPublicResult(comm, localResult);
  if (result != ncclSuccess) {
    if (internalRequest != nullptr) {
      delete internalRequest;
      gLiveRequests.fetch_sub(1, std::memory_order_relaxed);
    }
    delete entry;
    return result;
  }

  localResult = registeredAllReduceInit(
      internalRequest, sendbuff, recvbuff, capacityBytes);
  result = collectPublicResult(comm, localResult);
  if (result != ncclSuccess) {
    const ncclResult_t cleanupResult =
        registeredAllReduceFinalize(internalRequest, nullptr, true);
    if (cleanupResult == ncclSuccess) {
      delete entry;
      return result;
    }
    entry->request = internalRequest;
    entry->comm = comm;
    entry->commHash = comm->commHash;
    registerPublicRequest(entry);
    *request = entry;
    return cleanupResult;
  }

  entry->request = internalRequest;
  entry->comm = comm;
  entry->commHash = comm->commHash;
  registerPublicRequest(entry);
  *request = entry;
  return ncclSuccess;
}

ncclResult_t registeredAllReducePublicExec(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    const ncclRegisteredAllReduceGatedResidualNorm* norm,
    hipStream_t stream,
    void* request) {
  std::lock_guard<std::mutex> lock(publicRegistryMutex());
  RegisteredAllReducePublicRequest* entry = findPublicRequestLocked(request);
  if (entry == nullptr || entry->request == nullptr || entry->comm == nullptr ||
      entry->finalizing || entry->commHash != entry->comm->commHash) {
    return ncclInvalidArgument;
  }
  return registeredAllReduceExecute(
      entry->request, sendbuff, recvbuff, count, datatype, op, norm, stream);
}

ncclResult_t registeredAllReducePublicFinalize(
    void* request,
    hipStream_t stream) {
  RegisteredAllReducePublicRequest* entry = nullptr;
  RegisteredAllReduce* internalRequest = nullptr;
  {
    std::lock_guard<std::mutex> lock(publicRegistryMutex());
    entry = findPublicRequestLocked(request);
    if (entry == nullptr || entry->request == nullptr ||
        entry->comm == nullptr || entry->finalizing ||
        entry->commHash != entry->comm->commHash) {
      return ncclInvalidArgument;
    }
    entry->finalizing = true;
    internalRequest = entry->request;
  }

  ncclResult_t result =
      collectPublicResult(entry->comm, rejectActiveCapture(stream));
  if (result == ncclSuccess) {
    result = registeredAllReduceFinalize(internalRequest, stream, true);
  }
  if (result != ncclSuccess) {
    std::lock_guard<std::mutex> lock(publicRegistryMutex());
    entry->finalizing = false;
    return result;
  }

  {
    std::lock_guard<std::mutex> lock(publicRegistryMutex());
    if (!unregisterPublicRequestLocked(entry)) {
      return ncclInvalidUsage;
    }
  }
  delete entry;
  return ncclSuccess;
}

bool registeredAllReduceCommHasLiveRequests(ncclComm_t comm) {
  if (comm == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(publicRegistryMutex());
  for (RegisteredAllReducePublicRequest* entry = publicRegistryHead();
       entry != nullptr;
       entry = entry->next) {
    if (entry->comm == comm && entry->commHash == comm->commHash) {
      return true;
    }
  }
  return false;
}

void registeredAllReduceAbandonComm(ncclComm_t comm) {
  if (comm == nullptr) {
    return;
  }

  {
    // Device work may still reference the pool's regions; leak them.
    std::lock_guard<std::mutex> poolLock(statePoolMutex());
    auto& pools = statePools();
    for (auto it = pools.begin(); it != pools.end(); ++it) {
      if ((*it)->comm == comm && (*it)->commHash == comm->commHash) {
        (void)it->release();
        pools.erase(it);
        break;
      }
    }
  }
  size_t abandoned = 0;
  std::lock_guard<std::mutex> lock(publicRegistryMutex());
  for (RegisteredAllReducePublicRequest** cur = &publicRegistryHead();
       *cur != nullptr;) {
    RegisteredAllReducePublicRequest* entry = *cur;
    if (entry->comm == comm && entry->commHash == comm->commHash) {
      *cur = entry->next;
      entry->next = nullptr;
      ++abandoned;
      continue;
    }
    cur = &entry->next;
  }
  if (abandoned != 0) {
    WARN(
        "Registered all-reduce: abandoning %zu live request(s) during communicator abort; device resources remain allocated until process exit",
        abandoned);
  }
}

size_t registeredAllReduceLiveRequestsForTest() {
  return gLiveRequests.load(std::memory_order_relaxed);
}

size_t registeredAllReduceLivePeerMappingsForTest() {
  return gLivePeerMappings.load(std::memory_order_relaxed);
}

size_t registeredAllReducePooledStateImportsForTest() {
  return gPooledStateImports.load(std::memory_order_relaxed);
}

size_t registeredAllReduceLiveStateAllocationsForTest() {
  return gLiveStateAllocations.load(std::memory_order_relaxed);
}

void registeredAllReduceSetStaleMappingForTest(int reader, int owner) {
  gStaleReaderForTest.store(reader);
  gStaleOwnerForTest.store(owner);
}

void registeredAllReduceSetIpcOpenFailureRankForTest(int rank) {
  gFailIpcOpenRankForTest.store(rank, std::memory_order_relaxed);
}

} // namespace rcclx::relay
