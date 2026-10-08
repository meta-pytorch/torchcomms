/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "meta/relay/registered_allreduce_relay.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#include "bootstrap.h"
#include "comm.h"
#include "debug.h"
#include "meta/relay/registered_allreduce_relay_kernels.h"
#include "meta/relay/relay_helper_devices.h"
#include "param.h"

namespace rcclx::relay {

// Must match on both ranks; registration verifies it.
NCCL_PARAM(RegisteredArRelayMinBytes, "REGISTERED_AR_RELAY_MIN_BYTES", 2 << 20);
NCCL_PARAM(RegisteredArRelayLanes, "REGISTERED_AR_RELAY_LANES", 8);
NCCL_PARAM(RegisteredArRelaySlots, "REGISTERED_AR_RELAY_SLOTS", 2);
NCCL_PARAM(RegisteredArRelaySliceKb, "REGISTERED_AR_RELAY_SLICE_KB", 256);
NCCL_PARAM(RegisteredArRelayMinSliceKb, "REGISTERED_AR_RELAY_MIN_SLICE_KB", 16);
NCCL_PARAM(
    RegisteredArRelayDirectWeight,
    "REGISTERED_AR_RELAY_DIRECT_WEIGHT",
    1);
NCCL_PARAM(RelayMaxHelpers, "RELAY_MAX_HELPERS", kRegisteredRelayMaxHelpers);

namespace {

// Staging and flags cross GPUs, so they bypass both L2s.
#if defined(HIP_UNCACHED_MEMORY)
constexpr unsigned int kAllocFlags = hipDeviceMallocUncached;
#else
constexpr unsigned int kAllocFlags = hipDeviceMallocFinegrained;
#endif

constexpr size_t kVectorBytes = 16;

struct Config {
  int32_t lanes;
  int32_t slots;
  int32_t directWeight;
  int32_t maxHelpers;
  uint64_t sliceBytes;
  uint64_t minSliceBytes;
  uint64_t minBytes;
  uint64_t capacityBytes;

  bool sameContract(const Config& o) const {
    return lanes == o.lanes && slots == o.slots &&
        directWeight == o.directWeight && maxHelpers == o.maxHelpers &&
        sliceBytes == o.sliceBytes && minSliceBytes == o.minSliceBytes &&
        minBytes == o.minBytes;
  }
};

struct ExportRecord {
  int32_t ok;
  uint32_t stagingOk;
  int32_t slot;
  hipIpcMemHandle_t flags;
  hipIpcMemHandle_t staging[kRegisteredRelayMaxHelpers];
  // Owner-assigned identities of the pooled allocations; the peer imports
  // each identity once per communicator.
  uint64_t flagsId;
  uint64_t stagingId[kRegisteredRelayMaxHelpers];
};

// Relay flags and staging belong to the communicator, not the request: the
// peer writes into them over IPC, and freeing such memory while the process
// keeps running was observed to corrupt the page of a later allocation that
// reused the address. A slot is reset and reused by later requests, imported
// once by the peer, and freed only at ncclCommDestroy.
struct RelayPool {
  struct Staging {
    int device;
    void* ptr;
    size_t bytes;
    uint64_t id;
    hipIpcMemHandle_t handle;
  };
  struct Slot {
    RegisteredRelayFlags* flags;
    uint64_t flagsId;
    hipIpcMemHandle_t flagsHandle;
    std::vector<Staging> staging;
    bool busy;
  };
  struct Import {
    uint64_t id;
    void* mapping;
  };
  ncclComm_t comm;
  uint64_t commHash;
  int device;
  std::vector<Slot> slots;
  std::vector<Import> imports;
  uint64_t nextId;
};

std::mutex& relayPoolMutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<std::unique_ptr<RelayPool>>& relayPools() {
  static std::vector<std::unique_ptr<RelayPool>> pools;
  return pools;
}

RelayPool* relayPoolFor(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(relayPoolMutex());
  for (const auto& pool : relayPools()) {
    if (pool->comm == comm && pool->commHash == comm->commHash) {
      return pool.get();
    }
  }
  auto pool = std::make_unique<RelayPool>();
  pool->comm = comm;
  pool->commHash = comm->commHash;
  pool->device = comm->cudaDev;
  pool->nextId = (static_cast<uint64_t>(comm->rank) << 48) + 1;
  relayPools().push_back(std::move(pool));
  return relayPools().back().get();
}

size_t powerOfTwoFloor(size_t v) {
  size_t p = 1;
  while (p <= v / 2) {
    p *= 2;
  }
  return p;
}

Config readConfig(size_t capacityBytes) {
  Config c{};
  c.lanes = static_cast<int32_t>(std::clamp<int64_t>(
      ncclParamRegisteredArRelayLanes(), 1, kRegisteredRelayMaxLanes));
  // A power of two, so the slot index stays continuous when the 32-bit
  // sequence counters wrap.
  c.slots = static_cast<int32_t>(powerOfTwoFloor(
      static_cast<size_t>(std::clamp<int64_t>(
          ncclParamRegisteredArRelaySlots(), 2, kRegisteredRelayMaxSlots))));
  c.directWeight = static_cast<int32_t>(
      std::clamp<int64_t>(ncclParamRegisteredArRelayDirectWeight(), 1, 16));
  c.maxHelpers = static_cast<int32_t>(std::clamp<int64_t>(
      ncclParamRelayMaxHelpers(), 0, kRegisteredRelayMaxHelpers));
  c.sliceBytes = powerOfTwoFloor(
                     static_cast<size_t>(std::clamp<int64_t>(
                         ncclParamRegisteredArRelaySliceKb(), 16, 16384))) *
      1024;
  c.minSliceBytes = std::min<uint64_t>(
      powerOfTwoFloor(
          static_cast<size_t>(std::clamp<int64_t>(
              ncclParamRegisteredArRelayMinSliceKb(), 16, 16384))) *
          1024,
      c.sliceBytes);
  c.minBytes = static_cast<uint64_t>(
      std::max<int64_t>(ncclParamRegisteredArRelayMinBytes(), 16));
  c.capacityBytes = capacityBytes;
  return c;
}

std::atomic<uint64_t>& launches() {
  static std::atomic<uint64_t> count{0};
  return count;
}

} // namespace

struct RegisteredRelay {
  int device{-1};
  int rank{-1};
  Config config{};
  size_t ringBytes{0};
  int helpers{0};
  RelayPool* pool{nullptr};
  int slot{-1};
  std::array<void*, kRegisteredRelayMaxHelpers> staging{};
  std::array<void*, kRegisteredRelayMaxHelpers> peerStaging{};
  RegisteredRelayFlags* flags{nullptr};
  RegisteredRelayFlags* peerFlags{nullptr};
};

namespace {

// Returns the request's slot to the pool; nothing is freed or closed.
void releaseSlot(RegisteredRelay& r) {
  if (r.pool != nullptr && r.slot >= 0) {
    r.pool->slots[r.slot].busy = false;
  }
  r.slot = -1;
}

// This slot's staging on `device`, allocated (and exported) on first use.
// Returns null if it cannot be created; outgrown staging is kept, not freed.
const RelayPool::Staging*
stagingFor(RelayPool& pool, RelayPool::Slot& slot, int device, size_t bytes) {
  for (RelayPool::Staging& entry : slot.staging) {
    if (entry.device == device && entry.bytes >= bytes) {
      return &entry;
    }
  }
  RelayPool::Staging fresh{device, nullptr, bytes, 0, {}};
  bool ok = false;
  {
    RelayDeviceGuard onHelper(device);
    ok = hipExtMallocWithFlags(&fresh.ptr, bytes, kAllocFlags) == hipSuccess;
  }
  if (ok) {
    enableRelayPeerAccess(device);
    ok = hipIpcGetMemHandle(&fresh.handle, fresh.ptr) == hipSuccess;
    if (!ok) {
      RelayDeviceGuard onHelper(device);
      (void)hipFree(fresh.ptr);
    }
  }
  if (!ok) {
    (void)hipGetLastError();
    return nullptr;
  }
  fresh.id = pool.nextId++;
  slot.staging.push_back(fresh);
  return &slot.staging.back();
}

// Binds a free slot (allocating one if needed), resets its flags, and fills
// the export record. Returns false if the flags cannot be created; a helper
// whose staging fails is left out of stagingOk.
bool allocateAndExport(
    RegisteredRelay& r,
    const std::vector<int>& helpers,
    ExportRecord& record) {
  RelayDeviceGuard guard(r.device);
  RelayPool& pool = *r.pool;
  int slot = -1;
  for (size_t k = 0; k < pool.slots.size(); ++k) {
    if (!pool.slots[k].busy) {
      slot = static_cast<int>(k);
      break;
    }
  }
  if (slot < 0) {
    RelayPool::Slot fresh{};
    if (hipExtMallocWithFlags(
            reinterpret_cast<void**>(&fresh.flags),
            sizeof(RegisteredRelayFlags),
            kAllocFlags) != hipSuccess) {
      (void)hipGetLastError();
      return false;
    }
    if (hipIpcGetMemHandle(&fresh.flagsHandle, fresh.flags) != hipSuccess) {
      (void)hipFree(fresh.flags);
      (void)hipGetLastError();
      return false;
    }
    fresh.flagsId = pool.nextId++;
    pool.slots.push_back(std::move(fresh));
    slot = static_cast<int>(pool.slots.size()) - 1;
  }
  RelayPool::Slot& s = pool.slots[slot];
  s.busy = true;
  r.slot = slot;
  r.flags = s.flags;
  record.slot = slot;
  record.flags = s.flagsHandle;
  record.flagsId = s.flagsId;
  // Finalize drained every relay kernel of the slot's previous request on both
  // ranks, and the peer writes here only after this registration completes.
  if (hipMemset(s.flags, 0, sizeof(RegisteredRelayFlags)) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  for (size_t h = 0; h < helpers.size(); ++h) {
    const RelayPool::Staging* staging =
        stagingFor(pool, s, helpers[h], r.ringBytes);
    if (staging != nullptr) {
      r.staging[h] = staging->ptr;
      record.staging[h] = staging->handle;
      record.stagingId[h] = staging->id;
      record.stagingOk |= 1u << h;
    }
  }
  return hipDeviceSynchronize() == hipSuccess;
}

// The communicator-lifetime import of the peer's pooled allocation `id`.
bool importPooled(
    RelayPool& pool,
    uint64_t id,
    const hipIpcMemHandle_t& handle,
    void** mapping) {
  for (const RelayPool::Import& entry : pool.imports) {
    if (entry.id == id) {
      *mapping = entry.mapping;
      return true;
    }
  }
  void* opened = nullptr;
  if (hipIpcOpenMemHandle(&opened, handle, hipIpcMemLazyEnablePeerAccess) !=
      hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  pool.imports.push_back({id, opened});
  *mapping = opened;
  return true;
}

} // namespace

ncclResult_t registeredRelaySetup(
    ncclComm_t comm,
    size_t capacityBytes,
    RegisteredRelay** relay) {
  *relay = nullptr;
  if (comm->nRanks != 2) {
    return ncclSuccess;
  }
  Config configs[2]{};
  configs[comm->rank] = readConfig(capacityBytes);
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, configs, sizeof(Config)));
  if (!configs[0].sameContract(configs[1])) {
    WARN(
        "Registered all-reduce: NCCL_REGISTERED_AR_RELAY_* / NCCL_RELAY_MAX_HELPERS differ between the two ranks");
    return ncclInvalidUsage;
  }
  const Config config = configs[0];
  if (std::min(configs[0].capacityBytes, configs[1].capacityBytes) <
      config.minBytes) {
    return ncclSuccess;
  }
  std::vector<int> devices;
  NCCLCHECK(agreeRelayHelperDevices(comm, config.maxHelpers, &devices));
  if (devices.empty()) {
    return ncclSuccess;
  }

  auto* r = new (std::nothrow) RegisteredRelay;
  ExportRecord exports[2]{};
  if (r != nullptr) {
    r->device = comm->cudaDev;
    r->rank = comm->rank;
    r->config = config;
    r->pool = relayPoolFor(comm);
    r->ringBytes =
        static_cast<size_t>(config.lanes) * config.slots * config.sliceBytes;
    exports[comm->rank].ok =
        allocateAndExport(*r, devices, exports[comm->rank]);
  }
  const ncclResult_t gathered =
      bootstrapAllGather(comm->bootstrap, exports, sizeof(ExportRecord));
  if (gathered != ncclSuccess || !exports[0].ok || !exports[1].ok ||
      exports[0].slot != exports[1].slot) {
    if (r != nullptr) {
      releaseSlot(*r);
      delete r;
    }
    WARN("Registered all-reduce: relay state allocation failed on a rank");
    return gathered != ncclSuccess ? gathered : ncclSystemError;
  }

  // Keep the helpers both ranks staged on, compacting them in the same order.
  const int peer = 1 - comm->rank;
  const uint32_t agreed = exports[0].stagingOk & exports[1].stagingOk;
  bool importOk = true;
  int kept = 0;
  {
    RelayDeviceGuard guard(r->device);
    for (size_t h = 0; h < devices.size(); ++h) {
      if ((agreed & (1u << h)) == 0) {
        r->staging[h] = nullptr;
        continue;
      }
      r->staging[kept] = r->staging[h];
      if (kept != static_cast<int>(h)) {
        r->staging[h] = nullptr;
      }
      importOk = importOk &&
          importPooled(
                     *r->pool,
                     exports[peer].stagingId[h],
                     exports[peer].staging[h],
                     &r->peerStaging[kept]);
      ++kept;
    }
    r->helpers = kept;
    void* peerFlags = nullptr;
    importOk =
        importOk &&
        importPooled(
            *r->pool, exports[peer].flagsId, exports[peer].flags, &peerFlags);
    r->peerFlags = static_cast<RegisteredRelayFlags*>(peerFlags);
  }
  int32_t votes[2]{};
  votes[comm->rank] = importOk ? 1 : 0;
  const ncclResult_t voted =
      bootstrapAllGather(comm->bootstrap, votes, sizeof(int32_t));
  if (voted != ncclSuccess || !votes[0] || !votes[1]) {
    releaseSlot(*r);
    delete r;
    WARN("Registered all-reduce: opening the peer's relay mappings failed");
    return voted != ncclSuccess ? voted : ncclSystemError;
  }
  if (r->helpers == 0) {
    releaseSlot(*r);
    delete r;
    return ncclSuccess;
  }
  INFO(
      NCCL_INIT,
      "Registered all-reduce relay route: %d helper GPUs, %zu KiB staging per helper, from %zu bytes",
      r->helpers,
      r->ringBytes >> 10,
      static_cast<size_t>(config.minBytes));
  *relay = r;
  return ncclSuccess;
}

bool registeredRelayServes(const RegisteredRelay* relay, size_t bytes) {
  return relay != nullptr && bytes >= relay->config.minBytes;
}

hipError_t registeredRelayLaunch(
    RegisteredRelay* relay,
    const void* mine,
    const void* peer,
    void* output,
    size_t bytes,
    hipStream_t stream) {
  const Config& c = relay->config;
  const int paths = relay->helpers + 1;
  const size_t sliceBytes = std::clamp<size_t>(
      powerOfTwoFloor(std::max<size_t>(bytes / (paths * c.lanes), 1)),
      c.minSliceBytes,
      c.sliceBytes);
  RegisteredRelayArgs args{};
  args.mine = mine;
  args.peer = peer;
  args.output = output;
  args.vectors = bytes / kVectorBytes;
  args.sliceVectors = sliceBytes / kVectorBytes;
  args.slotVectors = c.sliceBytes / kVectorBytes;
  args.directWeight = c.directWeight;
  args.helpers = relay->helpers;
  args.directLanes = c.lanes;
  args.lanes = c.lanes;
  args.slots = c.slots;
  args.rank = relay->rank;
  for (int h = 0; h < relay->helpers; ++h) {
    args.push[h] = relay->staging[h];
    args.pull[h] = relay->peerStaging[h];
  }
  args.myFlags = relay->flags;
  args.peerFlags = relay->peerFlags;
  launches().fetch_add(1, std::memory_order_relaxed);
  return launchRegisteredRelayAllReduce(args, stream);
}

void registeredRelayRelease(RegisteredRelay* relay) {
  if (relay == nullptr) {
    return;
  }
  releaseSlot(*relay);
  delete relay;
}

void registeredRelayReleaseComm(ncclComm_t comm) {
  std::unique_ptr<RelayPool> pool;
  {
    std::lock_guard<std::mutex> lock(relayPoolMutex());
    auto& pools = relayPools();
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
  RelayDeviceGuard guard(pool->device);
  for (const RelayPool::Import& entry : pool->imports) {
    (void)hipIpcCloseMemHandle(entry.mapping);
  }
  for (const RelayPool::Slot& slot : pool->slots) {
    for (const RelayPool::Staging& staging : slot.staging) {
      RelayDeviceGuard onHelper(staging.device);
      (void)hipFree(staging.ptr);
    }
    (void)hipFree(slot.flags);
  }
  (void)hipGetLastError();
}

void registeredRelayAbandonComm(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(relayPoolMutex());
  auto& pools = relayPools();
  for (auto it = pools.begin(); it != pools.end(); ++it) {
    if ((*it)->comm == comm && (*it)->commHash == comm->commHash) {
      (void)it->release();
      pools.erase(it);
      break;
    }
  }
}

int registeredRelayHelperCount(const RegisteredRelay* relay) {
  return relay == nullptr ? 0 : relay->helpers;
}

ncclResult_t registeredRelaySetSequenceForTest(
    RegisteredRelay* relay,
    uint32_t sequence) {
  if (relay == nullptr) {
    return ncclInvalidArgument;
  }
  RelayDeviceGuard guard(relay->device);
  RegisteredRelayFlags flags{};
  if (hipMemcpy(
          &flags,
          relay->flags,
          sizeof(RegisteredRelayFlags),
          hipMemcpyDeviceToHost) != hipSuccess) {
    return ncclUnhandledCudaError;
  }
  for (int h = 0; h < kRegisteredRelayMaxHelpers; ++h) {
    for (int lane = 0; lane < kRegisteredRelayMaxLanes; ++lane) {
      flags.pushSeq[h][lane] = sequence;
      flags.reduceSeq[h][lane] = sequence;
      for (int slot = 0; slot < kRegisteredRelayMaxSlots; ++slot) {
        flags.full[h][lane][slot] = sequence;
        flags.freed[h][lane][slot] = sequence;
      }
    }
  }
  return hipMemcpy(
             relay->flags,
             &flags,
             sizeof(RegisteredRelayFlags),
             hipMemcpyHostToDevice) == hipSuccess &&
          hipDeviceSynchronize() == hipSuccess
      ? ncclSuccess
      : ncclUnhandledCudaError;
}

uint64_t registeredRelayLaunchesForTest() {
  return launches().load(std::memory_order_relaxed);
}

} // namespace rcclx::relay
