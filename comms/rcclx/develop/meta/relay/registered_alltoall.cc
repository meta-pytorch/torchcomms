/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "meta/relay/registered_alltoall.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#include "archinfo.h"
#include "bootstrap.h"
#include "comm.h"
#include "debug.h"
#include "meta/relay/registered_alltoall_kernels.h"
#include "meta/relay/relay_helper_devices.h"
#include "param.h"

namespace rcclx::relay {

namespace {

constexpr int kActive = kRegisteredAllToAllActiveRanks;
constexpr int kDefaultChunkRows = 16;
constexpr int kDefaultDirectCtasPerPeer = 16;
constexpr int kDefaultRelayCtasPerPath = 4;
constexpr size_t kAlignment = 16;
#if defined(HIP_UNCACHED_MEMORY)
constexpr unsigned int kFlagsAllocFlags = hipDeviceMallocUncached;
#else
constexpr unsigned int kFlagsAllocFlags = hipDeviceMallocFinegrained;
#endif

constexpr int kSlots = 16;
// Relay staging is allocated in steps of this size, so small growth reuses it.
constexpr size_t kStagingGranule = size_t{2} << 20;
constexpr size_t kPage = size_t{2} << 20;

std::atomic<size_t> gLivePeerMappings{0};
std::atomic<size_t> gPooledMappings{0};
std::atomic<int> gStaleReaderForTest{-1};
std::atomic<int> gStaleOwnerForTest{-1};

bool isAligned(size_t value) {
  return value % kAlignment == 0;
}

bool isAligned(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % kAlignment == 0;
}

ncclResult_t hipToNccl(hipError_t result, const char* what) {
  if (result == hipSuccess) {
    return ncclSuccess;
  }
  WARN("Registered all-to-all: %s failed: %s", what, hipGetErrorString(result));
  return ncclUnhandledCudaError;
}

// Bytes a pair-strided buffer must span from its base.
size_t
spanBytes(size_t rows, size_t rowBytes, size_t rowStride, size_t peerStride) {
  return (kActive - 1) * peerStride + (rows - 1) * rowStride + rowBytes;
}

// Everything every rank must agree on, exchanged at init.
struct InitRecord {
  hipIpcMemHandle_t dataHandle;
  hipIpcMemHandle_t flagsHandle;
  hipIpcMemHandle_t stagingHandles[kRegisteredAllToAllMaxHelpers];
  // Owner-assigned identities of the pooled flags and staging allocations;
  // peers import each identity once per communicator.
  uint64_t flagsId{0};
  uint64_t stagingIds[kRegisteredAllToAllMaxHelpers]{};
  int slot{-1};
  size_t dataOffset{0};
  ncclRegisteredAllToAllLayout layout{};
  float relayFraction{0.0f};
  int chunkRows{0};
  int directCtasPerPeer{0};
  int relayCtasPerPath{0};
  int helpers{0};
  uint32_t result{static_cast<uint32_t>(ncclInternalError)};
  // Registration-unique value mixed into the mapping probes.
  uint64_t probeNonce{0};
};

ncclResult_t firstFailure(const std::vector<InitRecord>& records) {
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

bool sameContract(const InitRecord& a, const InitRecord& b) {
  return std::memcmp(&a.layout, &b.layout, sizeof(a.layout)) == 0 &&
      a.relayFraction == b.relayFraction && a.chunkRows == b.chunkRows &&
      a.directCtasPerPeer == b.directCtasPerPeer &&
      a.relayCtasPerPath == b.relayCtasPerPath && a.helpers == b.helpers &&
      a.slot == b.slot;
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

} // namespace

// Memory the registered all-to-all owns, kept for the communicator's
// lifetime and reused by every request on it. Flags and relay staging are
// never freed or re-exported while the communicator lives, and peers import
// each allocation once, so no request depends on a fresh import of RCCLX
// memory. (Fresh imports of re-used addresses have been observed to resolve to
// stale pages.) Only the caller's send buffers are imported per request.
struct CommPool {
  struct Staging {
    int slot;
    int device;
    void* ptr;
    size_t bytes;
    uint64_t id;
    hipIpcMemHandle_t handle;
  };
  struct Import {
    int peer;
    uint64_t id;
    void* mapping;
  };

  ncclComm_t comm{nullptr};
  uint64_t commHash{0};
  RegisteredAllToAllFlags* flags{nullptr}; // kSlots slots, one per request
  uint64_t flagsId{0};
  hipIpcMemHandle_t flagsHandle{};
  bool busy[kSlots]{};
  std::vector<Staging> staging; // current, per (slot, helper device)
  std::vector<Staging> retired; // outgrown; freed with the communicator
  std::vector<Import> imports;
  uint64_t nextId{0};
};

struct RegisteredAllToAll {
  ncclComm_t comm{nullptr};
  uint64_t commHash{0};
  int rank{-1};
  int helpers{0};
  const void* sendBase{nullptr};
  void* recvBase{nullptr};
  CommPool* pool{nullptr};
  int slot{-1};
  std::vector<int> helperDevices;
  std::vector<void*> mappedSend; // per-request imports of peers' send buffers
  RegisteredAllToAllArgs args{};
  bool finalizing{false};
  RegisteredAllToAll* next{nullptr};
};

namespace {

std::mutex& registryMutex() {
  static std::mutex mutex;
  return mutex;
}

RegisteredAllToAll*& registryHead() {
  static RegisteredAllToAll* head = nullptr;
  return head;
}

RegisteredAllToAll* findLocked(void* handle) {
  for (RegisteredAllToAll* cur = registryHead(); cur != nullptr;
       cur = cur->next) {
    if (cur == handle) {
      return cur;
    }
  }
  return nullptr;
}

std::vector<std::unique_ptr<CommPool>>& pools() {
  static std::vector<std::unique_ptr<CommPool>> all;
  return all;
}

// Pools are keyed on the comm pointer and its hash (the allocator recycles
// comm pointers). Callers hold registryMutex().
CommPool* poolLocked(ncclComm_t comm, bool create) {
  for (const auto& pool : pools()) {
    if (pool->comm == comm && pool->commHash == comm->commHash) {
      return pool.get();
    }
  }
  if (!create) {
    return nullptr;
  }
  auto pool = std::make_unique<CommPool>();
  pool->comm = comm;
  pool->commHash = comm->commHash;
  // Identities are unique per owning rank; peers key imports on (rank, id).
  pool->nextId = (static_cast<uint64_t>(comm->rank) << 48) + 1;
  pools().push_back(std::move(pool));
  return pools().back().get();
}

void unregisterLocked(RegisteredAllToAll* request) {
  for (RegisteredAllToAll** cur = &registryHead(); *cur != nullptr;
       cur = &(*cur)->next) {
    if (*cur == request) {
      *cur = request->next;
      request->next = nullptr;
      return;
    }
  }
}

// NCCL_RELAY_MAX_HELPERS caps the relay GPUs (0 = direct only).
int relayHelperCap() {
  const char* env = ncclGetEnv("NCCL_RELAY_MAX_HELPERS");
  const long value = env == nullptr || env[0] == '\0'
      ? kRegisteredAllToAllMaxHelpers
      : std::strtol(env, nullptr, 10);
  return static_cast<int>(
      std::clamp<long>(value, 0, kRegisteredAllToAllMaxHelpers));
}

ncclResult_t validateTopology(ncclComm_t comm) {
  if (comm->nRanks != kActive || comm->nNodes != 1 ||
      comm->localRanks != comm->nRanks || comm->archName == nullptr ||
      !IsArchMatch(comm->archName, "gfx950") || !comm->isAllDirectP2p) {
    WARN(
        "Registered all-to-all requires exactly 4 local gfx950 ranks on one node with direct all-pairs P2P; got nRanks=%d nNodes=%d localRanks=%d arch=%s directP2p=%d",
        comm->nRanks,
        comm->nNodes,
        comm->localRanks,
        comm->archName == nullptr ? "<null>" : comm->archName,
        static_cast<int>(comm->isAllDirectP2p));
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

ncclResult_t buildGeometry(
    const ncclRegisteredAllToAllLayout& layout,
    const InitRecord& record,
    int helpers,
    RegisteredAllToAllGeometry& geometry) {
  if (layout.rows == 0 || layout.rows > INT32_MAX || layout.rowBytes == 0 ||
      !isAligned(layout.rowBytes) || !isAligned(layout.sendRowStride) ||
      !isAligned(layout.sendPeerStride) || !isAligned(layout.recvRowStride) ||
      !isAligned(layout.recvPeerStride) ||
      layout.sendRowStride < layout.rowBytes ||
      layout.recvRowStride < layout.rowBytes) {
    return ncclInvalidArgument;
  }
  if (!(record.relayFraction >= 0.0f && record.relayFraction < 1.0f) ||
      record.chunkRows <= 0 || record.directCtasPerPeer <= 0 ||
      record.relayCtasPerPath <= 0) {
    return ncclInvalidArgument;
  }
  const int rows = static_cast<int>(layout.rows);
  const int relayRows = helpers == 0
      ? 0
      : static_cast<int>(std::floor(rows * record.relayFraction / helpers));
  geometry.rowBytes = layout.rowBytes;
  geometry.sendRowStride = layout.sendRowStride;
  geometry.sendPeerStride = layout.sendPeerStride;
  geometry.recvRowStride = layout.recvRowStride;
  geometry.recvPeerStride = layout.recvPeerStride;
  geometry.rows = rows;
  geometry.relayRows = relayRows;
  geometry.helpers = relayRows == 0 ? 0 : helpers;
  geometry.directRows = rows - relayRows * geometry.helpers;
  geometry.chunkRows = record.chunkRows;
  geometry.directCtasPerPeer = record.directCtasPerPeer;
  geometry.relayCtasPerGroup = record.relayCtasPerPath;
  if ((relayRows + record.chunkRows - 1) / record.chunkRows >
      kRegisteredAllToAllMaxChunks) {
    WARN(
        "Registered all-to-all: %d relayed rows per helper need more than %d chunks of %d rows",
        relayRows,
        kRegisteredAllToAllMaxChunks,
        record.chunkRows);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

// Exports the rank's data buffer (active: send buffer, helper: staging) with
// its offset inside the allocation, after checking the layout fits in it.
ncclResult_t exportData(const void* data, size_t bytes, InitRecord& record) {
  void* base = nullptr;
  size_t allocationBytes = 0;
  hipError_t result =
      hipMemGetAddressRange(&base, &allocationBytes, const_cast<void*>(data));
  if (result != hipSuccess) {
    return hipToNccl(result, "hipMemGetAddressRange(data)");
  }
  const size_t offset =
      static_cast<const char*>(data) - static_cast<char*>(base);
  if (offset > allocationBytes || bytes > allocationBytes - offset) {
    WARN(
        "Registered all-to-all: layout needs %zu bytes at offset %zu of a %zu-byte allocation",
        bytes,
        offset,
        allocationBytes);
    return ncclInvalidArgument;
  }
  record.dataOffset = offset;
  return hipToNccl(
      hipIpcGetMemHandle(&record.dataHandle, base), "hipIpcGetMemHandle(data)");
}

ncclResult_t ensureFlags(CommPool& pool) {
  if (pool.flags != nullptr) {
    return ncclSuccess;
  }
  void* flags = nullptr;
  hipError_t result = hipExtMallocWithFlags(
      &flags, kSlots * sizeof(RegisteredAllToAllFlags), kFlagsAllocFlags);
  if (result != hipSuccess) {
    return hipToNccl(result, "hipExtMallocWithFlags(flags)");
  }
  result = hipMemset(flags, 0, kSlots * sizeof(RegisteredAllToAllFlags));
  if (result == hipSuccess) {
    result = hipIpcGetMemHandle(&pool.flagsHandle, flags);
  }
  if (result != hipSuccess) {
    (void)hipFree(flags);
    return hipToNccl(result, "flags setup");
  }
  pool.flags = static_cast<RegisteredAllToAllFlags*>(flags);
  pool.flagsId = pool.nextId++;
  return ncclSuccess;
}

// This request's staging on `device`, grown if the pooled one is too small.
// An outgrown buffer is retired, not freed, so its address is never reused
// while peers may still hold an import of it.
ncclResult_t ensureStaging(
    CommPool& pool,
    int slot,
    int device,
    size_t bytes,
    const CommPool::Staging** out) {
  CommPool::Staging* current = nullptr;
  for (CommPool::Staging& entry : pool.staging) {
    if (entry.slot == slot && entry.device == device) {
      current = &entry;
    }
  }
  if (current != nullptr && current->bytes >= bytes) {
    *out = current;
    return ncclSuccess;
  }
  const size_t rounded =
      (bytes + kStagingGranule - 1) / kStagingGranule * kStagingGranule;
  CommPool::Staging fresh{slot, device, nullptr, rounded, 0, {}};
  hipError_t result = hipSuccess;
  {
    RelayDeviceGuard onHelper(device);
    result = hipExtMallocWithFlags(&fresh.ptr, rounded, kFlagsAllocFlags);
  }
  if (result != hipSuccess) {
    return hipToNccl(result, "hipExtMallocWithFlags(helper staging)");
  }
  enableRelayPeerAccess(device);
  result = hipIpcGetMemHandle(&fresh.handle, fresh.ptr);
  if (result != hipSuccess) {
    RelayDeviceGuard onHelper(device);
    (void)hipFree(fresh.ptr);
    return hipToNccl(result, "hipIpcGetMemHandle(helper staging)");
  }
  fresh.id = pool.nextId++;
  if (current != nullptr) {
    pool.retired.push_back(*current);
    *current = fresh;
  } else {
    pool.staging.push_back(fresh);
    current = &pool.staging.back();
  }
  *out = current;
  return ncclSuccess;
}

ncclResult_t prepareLocal(
    RegisteredAllToAll& request,
    const ncclRegisteredAllToAllLayout& layout,
    InitRecord& record) {
  ncclResult_t result =
      buildGeometry(layout, record, request.helpers, request.args.geometry);
  if (result != ncclSuccess) {
    return result;
  }
  if (request.sendBase == nullptr || request.recvBase == nullptr ||
      request.sendBase == request.recvBase || !isAligned(request.sendBase) ||
      !isAligned(request.recvBase)) {
    return ncclInvalidArgument;
  }
  CommPool& pool = *request.pool;
  result = ensureFlags(pool);
  if (result != ncclSuccess) {
    return result;
  }
  for (int slot = 0; slot < kSlots; ++slot) {
    if (!pool.busy[slot]) {
      request.slot = slot;
      break;
    }
  }
  if (request.slot < 0) {
    WARN(
        "Registered all-to-all: more than %d live requests on one communicator",
        kSlots);
    return ncclInvalidUsage;
  }
  pool.busy[request.slot] = true;
  // Finalize drained every kernel of the slot's previous request on all
  // ranks, and peers write here only after this registration completes.
  const hipError_t cleared =
      hipMemset(pool.flags + request.slot, 0, sizeof(RegisteredAllToAllFlags));
  if (cleared != hipSuccess) {
    return hipToNccl(cleared, "hipMemset(flags slot)");
  }
  record.slot = request.slot;
  record.flagsHandle = pool.flagsHandle;
  record.flagsId = pool.flagsId;
  const int helpers = request.args.geometry.helpers;
  const size_t stagingBytes = std::max<size_t>(
      registeredAllToAllStagingBytes(request.args.geometry), kAlignment);
  for (int h = 0; h < helpers; ++h) {
    const CommPool::Staging* staging = nullptr;
    result = ensureStaging(
        pool, request.slot, request.helperDevices[h], stagingBytes, &staging);
    if (result != ncclSuccess) {
      return result;
    }
    request.args.staging[request.rank][h] =
        static_cast<unsigned char*>(staging->ptr);
    record.stagingHandles[h] = staging->handle;
    record.stagingIds[h] = staging->id;
  }
  return exportData(
      request.sendBase,
      spanBytes(
          layout.rows,
          layout.rowBytes,
          layout.sendRowStride,
          layout.sendPeerStride),
      record);
}

ncclResult_t closeMappings(RegisteredAllToAll& request) {
  ncclResult_t result = ncclSuccess;
  for (void*& mapping : request.mappedSend) {
    if (mapping == nullptr) {
      continue;
    }
    const hipError_t closed = hipIpcCloseMemHandle(mapping);
    if (closed != hipSuccess && result == ncclSuccess) {
      result = hipToNccl(closed, "hipIpcCloseMemHandle");
    }
    mapping = nullptr;
    gLivePeerMappings.fetch_sub(1, std::memory_order_relaxed);
  }
  return result;
}

void releaseSlot(RegisteredAllToAll& request) {
  if (request.pool != nullptr && request.slot >= 0) {
    request.pool->busy[request.slot] = false;
  }
  request.slot = -1;
}

ncclResult_t openMapping(const hipIpcMemHandle_t& handle, void*& mapping) {
  const hipError_t result =
      hipIpcOpenMemHandle(&mapping, handle, hipIpcMemLazyEnablePeerAccess);
  if (result != hipSuccess) {
    mapping = nullptr;
    return hipToNccl(result, "hipIpcOpenMemHandle");
  }
  return ncclSuccess;
}

// The communicator-lifetime import of `peer`'s pooled allocation `id`.
ncclResult_t importPooled(
    CommPool& pool,
    int peer,
    uint64_t id,
    const hipIpcMemHandle_t& handle,
    void** mapping) {
  for (const CommPool::Import& entry : pool.imports) {
    if (entry.peer == peer && entry.id == id) {
      *mapping = entry.mapping;
      return ncclSuccess;
    }
  }
  void* opened = nullptr;
  const ncclResult_t result = openMapping(handle, opened);
  if (result != ncclSuccess) {
    return result;
  }
  pool.imports.push_back({peer, id, opened});
  gPooledMappings.fetch_add(1, std::memory_order_relaxed);
  *mapping = opened;
  return ncclSuccess;
}

// Offsets probed in a region: every 2 MiB page (the granularity at which
// stale imports were observed) and the last 16 bytes.
std::vector<size_t> probeOffsets(size_t bytes) {
  std::vector<size_t> offsets;
  for (size_t off = 0; off + 16 <= bytes; off += kPage) {
    offsets.push_back(off);
  }
  if (bytes >= 16 && (offsets.empty() || offsets.back() != bytes - 16)) {
    offsets.push_back(bytes - 16);
  }
  return offsets;
}

uint64_t probeValue(uint64_t nonce, int rank, uint64_t tag) {
  uint64_t x = nonce ^ (uint64_t(rank + 1) * 0x9e3779b97f4a7c15ull) ^ tag;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x ^= x >> 29;
  return x;
}

enum class ProbeKind { Send, Flags, Staging };

const char* probeKindName(ProbeKind kind) {
  switch (kind) {
    case ProbeKind::Send:
      return "send buffer";
    case ProbeKind::Flags:
      return "protocol flags";
    case ProbeKind::Staging:
      return "relay staging";
  }
  return "?";
}

// One 16-byte probe of `owner`'s memory: where this rank reads it, and the
// pattern the owner wrote there.
struct ProbeSite {
  int owner;
  ProbeKind kind;
  int helper;
  size_t offset;
  unsigned char* address;
  uint64_t expected[2];
};

uint64_t probeTag(ProbeKind kind, int helper, size_t offset) {
  return (static_cast<uint64_t>(kind) + 1) << 56 ^
      static_cast<uint64_t>(helper) << 48 ^ offset;
}

// Every 16-byte probe site of `owner`'s memory, addressed from this rank.
std::vector<ProbeSite> probeSitesOf(
    const RegisteredAllToAll& request,
    const std::vector<InitRecord>& records,
    int owner) {
  const RegisteredAllToAllGeometry& g = request.args.geometry;
  const RegisteredAllToAllArgs& args = request.args;
  const uint64_t nonce = records[owner].probeNonce;
  std::vector<ProbeSite> sites;
  auto add = [&](ProbeKind kind, int helper, size_t offset, unsigned char* p) {
    const uint64_t tag = probeTag(kind, helper, offset);
    sites.push_back(
        {owner,
         kind,
         helper,
         offset,
         p,
         {probeValue(nonce, owner, tag),
          probeValue(nonce, owner, tag ^ 0x5555555555555555ull)}});
  };
  unsigned char* send = const_cast<unsigned char*>(args.send[owner]);
  for (size_t off : probeOffsets(
           spanBytes(g.rows, g.rowBytes, g.sendRowStride, g.sendPeerStride))) {
    add(ProbeKind::Send, 0, off, send + off);
  }
  add(ProbeKind::Flags,
      0,
      0,
      reinterpret_cast<unsigned char*>(args.activeFlags[owner]->probe));
  const size_t stagingBytes =
      std::max<size_t>(registeredAllToAllStagingBytes(g), kAlignment);
  for (int h = 0; h < g.helpers; ++h) {
    for (size_t off : probeOffsets(stagingBytes)) {
      add(ProbeKind::Staging, h, off, args.staging[owner][h] + off);
    }
  }
  return sites;
}

// Reads every site through a GPU kernel that loads the way the exchange does
// (the copy engine and the shader can translate a mapping differently, and a
// hipMemcpy-based check was seen to pass a mapping the exchange then read
// stale). Returns the 16 bytes read per site.
ncclResult_t readSitesOnDevice(
    const std::vector<ProbeSite>& sites,
    std::vector<uint64_t>& values) {
  values.assign(2 * sites.size(), 0);
  if (sites.empty()) {
    return ncclSuccess;
  }
  std::vector<RegisteredAllToAllProbe> probes;
  probes.reserve(sites.size());
  for (const ProbeSite& site : sites) {
    probes.push_back({site.address, site.kind == ProbeKind::Flags ? 1 : 0});
  }
  void* deviceProbes = nullptr;
  void* deviceValues = nullptr;
  hipStream_t stream = nullptr;
  hipError_t result =
      hipMalloc(&deviceProbes, probes.size() * sizeof(probes[0]));
  if (result == hipSuccess) {
    result = hipMalloc(&deviceValues, values.size() * sizeof(uint64_t));
  }
  if (result == hipSuccess) {
    result = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  }
  if (result == hipSuccess) {
    result = hipMemcpyAsync(
        deviceProbes,
        probes.data(),
        probes.size() * sizeof(probes[0]),
        hipMemcpyHostToDevice,
        stream);
  }
  if (result == hipSuccess) {
    result = launchRegisteredAllToAllProbe(
        static_cast<const RegisteredAllToAllProbe*>(deviceProbes),
        static_cast<int>(probes.size()),
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
  (void)hipFree(deviceProbes);
  return hipToNccl(result, "registered all-to-all mapping probe");
}

// Proves that every mapping this request uses reaches the owner's current
// memory, reading it the way the exchange will. Owners write
// registration-unique patterns at every 2 MiB page of their send buffer
// (restored afterwards) and relay staging, and into their flags slot; every
// rank reads its peers' patterns through its mappings on the GPU. Collective;
// returns the agreed result.
ncclResult_t verifyPeerMappings(
    RegisteredAllToAll& request,
    const std::vector<InitRecord>& records) {
  ncclComm_t comm = request.comm;
  const std::vector<ProbeSite> own =
      probeSitesOf(request, records, request.rank);
  std::vector<uint64_t> saved(2 * own.size(), 0);
  // The probes overwrite the caller's send buffer with null-stream copies,
  // which do not order against non-blocking streams: drain every stream first
  // so no producer of the buffer (or exchange of another request) is still in
  // flight.
  ncclResult_t local =
      hipToNccl(hipDeviceSynchronize(), "hipDeviceSynchronize(probe)");
  for (size_t k = 0; k < own.size() && local == ncclSuccess; ++k) {
    if (own[k].kind == ProbeKind::Send) {
      local = hipToNccl(
          hipMemcpy(&saved[2 * k], own[k].address, 16, hipMemcpyDefault),
          "hipMemcpy(save probe)");
    }
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
    for (int peer = 0; peer < kActive; ++peer) {
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
      if (site.kind == ProbeKind::Send && site.offset >= kPage &&
          request.rank == gStaleReaderForTest.load() &&
          site.owner == gStaleOwnerForTest.load()) {
        values[2 * k] = ~values[2 * k];
      }
      if (values[2 * k] != site.expected[0] ||
          values[2 * k + 1] != site.expected[1]) {
        WARN(
            "Registered all-to-all: rank %d's mapping of rank %d's %s returns stale data at byte %zu; a fresh import of a re-used address resolved to earlier pages. Register send buffers that stay allocated for the process lifetime.",
            request.rank,
            site.owner,
            probeKindName(site.kind),
            site.offset);
        local = ncclSystemError;
      }
    }
    result = allRanksResult(comm, local);
  }

  // Every peer has finished reading; restore the owner's send buffer bytes.
  ncclResult_t restored = ncclSuccess;
  for (size_t k = 0; k < own.size(); ++k) {
    if (own[k].kind != ProbeKind::Send) {
      continue;
    }
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

// Maps every peer's send buffer (per request) and pooled flags and staging
// (once per communicator), and fills the kernel tables.
ncclResult_t mapPeers(
    RegisteredAllToAll& request,
    const std::vector<InitRecord>& records) {
  CommPool& pool = *request.pool;
  const int helpers = request.args.geometry.helpers;
  request.mappedSend.assign(kActive, nullptr);
  RegisteredAllToAllArgs& args = request.args;
  for (int peer = 0; peer < kActive; ++peer) {
    if (peer == request.rank) {
      args.send[peer] =
          static_cast<unsigned char*>(const_cast<void*>(request.sendBase));
      args.activeFlags[peer] = pool.flags + request.slot;
      continue;
    }
    ncclResult_t result =
        openMapping(records[peer].dataHandle, request.mappedSend[peer]);
    if (result != ncclSuccess) {
      return result;
    }
    gLivePeerMappings.fetch_add(1, std::memory_order_relaxed);
    args.send[peer] = static_cast<unsigned char*>(request.mappedSend[peer]) +
        records[peer].dataOffset;
    void* flags = nullptr;
    result = importPooled(
        pool, peer, records[peer].flagsId, records[peer].flagsHandle, &flags);
    if (result != ncclSuccess) {
      return result;
    }
    args.activeFlags[peer] =
        static_cast<RegisteredAllToAllFlags*>(flags) + request.slot;
    for (int h = 0; h < helpers; ++h) {
      void* staging = nullptr;
      result = importPooled(
          pool,
          peer,
          records[peer].stagingIds[h],
          records[peer].stagingHandles[h],
          &staging);
      if (result != ncclSuccess) {
        return result;
      }
      args.staging[peer][h] = static_cast<unsigned char*>(staging);
    }
  }
  args.recv = static_cast<unsigned char*>(request.recvBase);
  args.index = request.rank;
  return ncclSuccess;
}

ncclResult_t teardown(RegisteredAllToAll& request) {
  const ncclResult_t closed = closeMappings(request);
  const ncclResult_t agreed = allRanksResult(request.comm, closed);
  releaseSlot(request);
  return agreed;
}

ncclResult_t initRequest(
    RegisteredAllToAll& request,
    const ncclRegisteredAllToAllLayout* layout,
    const ncclRegisteredAllToAllConfig* config) {
  InitRecord mine{};
  if (layout != nullptr) {
    mine.layout = *layout;
  }
  mine.relayFraction = config == nullptr ? 0.0f : config->relayFraction;
  mine.chunkRows = config == nullptr || config->chunkRows == 0
      ? kDefaultChunkRows
      : config->chunkRows;
  mine.directCtasPerPeer = config == nullptr || config->directCtasPerPeer == 0
      ? kDefaultDirectCtasPerPeer
      : config->directCtasPerPeer;
  mine.relayCtasPerPath = config == nullptr || config->relayCtasPerPath == 0
      ? kDefaultRelayCtasPerPath
      : config->relayCtasPerPath;
  static std::atomic<uint64_t> registrations{0};
  mine.probeNonce =
      request.commHash ^
      (registrations.fetch_add(1, std::memory_order_relaxed) + 1) *
          0xd1b54a32d192ed03ull ^
      static_cast<uint64_t>(
          std::chrono::steady_clock::now().time_since_epoch().count());
  ncclResult_t local = validateTopology(request.comm);
  if (local == ncclSuccess) {
    // Helper GPUs are needed only when relaying. Agreeing on them is
    // collective, and relayFraction is part of the contract every rank must
    // match, so every rank takes the same branch.
    local = agreeRelayHelperDevices(
        request.comm,
        mine.relayFraction > 0.0f ? relayHelperCap() : 0,
        &request.helperDevices);
  }
  if (local == ncclSuccess) {
    request.helpers = static_cast<int>(request.helperDevices.size());
    mine.helpers = request.helpers;
    local = layout == nullptr ? ncclInvalidArgument
                              : prepareLocal(request, *layout, mine);
  }
  mine.result = static_cast<uint32_t>(local);

  std::vector<InitRecord> records(request.comm->nRanks);
  records[request.rank] = mine;
  ncclResult_t result = bootstrapAllGather(
      request.comm->bootstrap, records.data(), sizeof(InitRecord));
  if (result == ncclSuccess) {
    result = firstFailure(records);
  }
  if (result == ncclSuccess) {
    for (const InitRecord& record : records) {
      if (!sameContract(record, records[0])) {
        WARN(
            "Registered all-to-all: ranks passed different layouts or configs");
        result = ncclInvalidArgument;
      }
    }
  }
  if (result != ncclSuccess) {
    releaseSlot(request);
    return result;
  }
  result = allRanksResult(request.comm, mapPeers(request, records));
  if (result == ncclSuccess) {
    result = verifyPeerMappings(request, records);
  }
  if (result != ncclSuccess) {
    teardown(request);
  }
  return result;
}

} // namespace

ncclResult_t registeredAllToAllPublicInit(
    const void* sendbuff,
    void* recvbuff,
    const ncclRegisteredAllToAllLayout* layout,
    const ncclRegisteredAllToAllConfig* config,
    ncclComm_t comm,
    void** request) {
  if (request == nullptr || comm == nullptr) {
    return ncclInvalidArgument;
  }
  *request = nullptr;
  auto* entry = new (std::nothrow) RegisteredAllToAll;
  if (allRanksResult(comm, entry == nullptr ? ncclSystemError : ncclSuccess) !=
      ncclSuccess) {
    delete entry;
    return ncclSystemError;
  }
  entry->comm = comm;
  entry->commHash = comm->commHash;
  entry->rank = comm->rank;
  entry->sendBase = sendbuff;
  entry->recvBase = recvbuff;
  {
    std::lock_guard<std::mutex> lock(registryMutex());
    entry->pool = poolLocked(comm, /*create=*/true);
  }
  const ncclResult_t result = initRequest(*entry, layout, config);
  if (result != ncclSuccess) {
    delete entry;
    return result;
  }
  std::lock_guard<std::mutex> lock(registryMutex());
  entry->next = registryHead();
  registryHead() = entry;
  *request = entry;
  return ncclSuccess;
}

ncclResult_t registeredAllToAllPublicExec(
    const void* sendbuff,
    void* recvbuff,
    hipStream_t stream,
    void* request) {
  std::lock_guard<std::mutex> lock(registryMutex());
  RegisteredAllToAll* entry = findLocked(request);
  if (entry == nullptr || entry->finalizing ||
      entry->commHash != entry->comm->commHash) {
    return ncclInvalidArgument;
  }
  if (sendbuff != entry->sendBase || recvbuff != entry->recvBase) {
    return ncclInvalidArgument;
  }
  return hipToNccl(
      launchRegisteredAllToAllActive(entry->args, stream),
      "registered all-to-all launch");
}

ncclResult_t registeredAllToAllPublicFinalize(
    void* request,
    hipStream_t stream) {
  RegisteredAllToAll* entry = nullptr;
  {
    std::lock_guard<std::mutex> lock(registryMutex());
    entry = findLocked(request);
    if (entry == nullptr || entry->finalizing ||
        entry->commHash != entry->comm->commHash) {
      return ncclInvalidArgument;
    }
    entry->finalizing = true;
  }
  ncclResult_t result = allRanksResult(
      entry->comm,
      hipToNccl(
          hipStreamSynchronize(stream), "hipStreamSynchronize(finalize)"));
  if (result == ncclSuccess) {
    result = teardown(*entry);
  }
  if (result != ncclSuccess) {
    std::lock_guard<std::mutex> lock(registryMutex());
    entry->finalizing = false;
    return result;
  }
  {
    std::lock_guard<std::mutex> lock(registryMutex());
    unregisterLocked(entry);
  }
  delete entry;
  return ncclSuccess;
}

bool registeredAllToAllCommHasLiveRequests(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(registryMutex());
  for (RegisteredAllToAll* cur = registryHead(); cur != nullptr;
       cur = cur->next) {
    if (cur->comm == comm && cur->commHash == comm->commHash) {
      return true;
    }
  }
  return false;
}

void registeredAllToAllReleaseComm(ncclComm_t comm) {
  std::unique_ptr<CommPool> pool;
  {
    std::lock_guard<std::mutex> lock(registryMutex());
    auto& all = pools();
    for (auto it = all.begin(); it != all.end(); ++it) {
      if ((*it)->comm == comm && (*it)->commHash == comm->commHash) {
        pool = std::move(*it);
        all.erase(it);
        break;
      }
    }
  }
  if (pool == nullptr) {
    return;
  }
  for (const CommPool::Import& entry : pool->imports) {
    (void)hipIpcCloseMemHandle(entry.mapping);
    gPooledMappings.fetch_sub(1, std::memory_order_relaxed);
  }
  for (const auto* list : {&pool->staging, &pool->retired}) {
    for (const CommPool::Staging& entry : *list) {
      RelayDeviceGuard onHelper(entry.device);
      (void)hipFree(entry.ptr);
    }
  }
  if (pool->flags != nullptr) {
    (void)hipFree(pool->flags);
  }
}

void registeredAllToAllAbandonComm(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(registryMutex());
  // Device work may still reference the pool; leak it with the requests.
  auto& all = pools();
  for (auto it = all.begin(); it != all.end(); ++it) {
    if ((*it)->comm == comm && (*it)->commHash == comm->commHash) {
      (void)it->release();
      all.erase(it);
      break;
    }
  }
  size_t abandoned = 0;
  for (RegisteredAllToAll** cur = &registryHead(); *cur != nullptr;) {
    RegisteredAllToAll* entry = *cur;
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
        "Registered all-to-all: abandoning %zu live request(s) during communicator abort; device resources remain allocated until process exit",
        abandoned);
  }
}

void registeredAllToAllSetStaleMappingForTest(int reader, int owner) {
  gStaleReaderForTest.store(reader);
  gStaleOwnerForTest.store(owner);
}

size_t registeredAllToAllLivePeerMappingsForTest() {
  return gLivePeerMappings.load(std::memory_order_relaxed);
}

size_t registeredAllToAllPooledMappingsForTest() {
  return gPooledMappings.load(std::memory_order_relaxed);
}

} // namespace rcclx::relay
