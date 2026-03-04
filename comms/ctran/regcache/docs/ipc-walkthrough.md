# IPC Registration Walkthrough: Two-Rank Example

This document traces the complete lifecycle of an IPC-registered buffer
across two ranks, two communicators, and both cleanup paths. Each step
is verified against the code.

> **Note:** This walkthrough uses the `RegCache::globalRegister` /
> `globalDeregister` APIs, which are the intended registration path.
> There is also a legacy path through `CtranMapper::regMem` /
> `deregMem` / `freeSegment` that wraps the same underlying operations;
> it is being deprecated in favor of the direct `RegCache` APIs.

## Setup

- **Rank0**: Source rank, allocates and registers memory
- **Rank1**: Destination rank, imports remote memory via IPC
- **Comm1 / Comm2**: Two different communicators (mapper1 / mapper2)
- Both comms share the same `RegCache` singleton on each rank
- Both comms share the same `IpcRegCache` singleton on each rank

---

## Step 1: Allocation and Registration (Rank0 only)

User allocates memory and calls `globalRegister`.

```
  Rank0                                          Rank1
  ─────────────────────────────────────────      ─────────────────────────────
  User Code                                      (idle)
    │
    ├─ cudaMalloc(&buf, size)
    │
    └─ globalRegister(buf, size)
         │
         ▼
  RegCache (singleton)
    │
    ├─ cacheSegment(buf, size, cudaDev)
    │    ├─ pinRange(buf, cudaDev, size)
    │    │   → discovers SegmentRange{base, len}
    │    ├─ AVL search(base, len) → MISS
    │    ├─ create Segment{range, cudaDev}
    │    └─ AVL insert → segment cached
    │
    segmentsAvl_: [Segment(buf, size)]
    regElemsMaps_: (empty, lazy mode)

  Mapper1 (comm1): idle
  Mapper2 (comm2): idle
```

**State after Step 1:**
- Rank0 RegCache: 1 Segment in AVL tree, 0 RegElems (lazy mode)
- Rank0 IpcRegCache: empty
- Rank1: nothing happened yet

---

## Step 2: First Collective (Comm1)

GPE thread runs a collective on comm1. Rank0 sends data to Rank1.

```
  Rank0                                          Rank1
  ─────────────────────────────────────────      ─────────────────────────────
  GPE Thread (collective on comm1)               (idle, waiting for ctrl msg)
    │
    └─ mapper1->searchRegHandle(buf, size)
         │
         ▼
  RegCache
    │
    ├─ regRange(buf, size, enableBackends)
    │    ├─ FAST PATH: rlock regElemsMaps_
    │    │   searchRegElem(buf, size) → MISS
    │    │
    │    ├─ SLOW PATH: wlock segmentsAvl_
    │    │   pinRange → rediscover segments
    │    │   AVL lookup → segments found
    │    │
    │    ├─ registerSegmentsTogether(segments)
    │    │   → doRegister(IB, IPC, TCP)
    │    │     → IpcRegCache::regMem(buf, size)
    │    │       → creates IpcRegElem{hdl, uid=1}
    │    │     → CtranIb::regMem (if IB enabled)
    │    │
    │    └─ creates RegElem{buf, size, ibHdl, ipcHdl}
    │       regHdlToElemMap[regElem] = regElem
    │       segToRegElemsMap[seg] = [regElem]
    │
    regElemsMaps_: 1 RegElem (REGISTERED)
         │
         │ regHdl (RegElem*)
         ▼
  Mapper1 (comm1)                                Mapper1 (comm1)
    │                                              │
    ├─ exportMem(regHdl, rank=1)                   │
    │   → IpcRegCache::exportMem(ipcRegElem)       │
    │     → fills IpcDesc{handle, offset, uid=1}   │
    │                                              │
    ├─ exportRegCache_.record(regElem, peerRank=1)  │
    │                                              │
    └─ send ctrl msg ─────────────────────────►    ├─ recv ctrl msg
                                                   │
                                                   └─ importMem(IpcDesc)
                                                        │
                                                        ▼
                                                   IpcRegCache (singleton)
                                                     │
                                                     ├─ importRemMemImpl(
                                                     │    peerId=Rank0,
                                                     │    IpcDesc{hdl, uid=1})
                                                     │
                                                     ├─ cache key = {Rank0,
                                                     │    {base, uid=1}}
                                                     ├─ lookup → MISS
                                                     │
                                                     ├─ cuIpcOpenMemHandle
                                                     │   → maps Rank0's memory
                                                     │
                                                     └─ create IpcRemRegElem{
                                                          base, len, refCount=1
                                                        }
```

**State after Step 2:**
- Rank0 RegCache: 1 Segment, 1 RegElem (REGISTERED)
- Rank0 IpcRegCache: 1 IpcRegElem(uid=1)
- Rank0 Mapper1 exportRegCache_: {regElem → [Rank1]}
- Rank1 IpcRegCache: 1 IpcRemRegElem(refCount=1), CUDA IPC memory mapped

---

## Step 3: Second Collective (Comm2, same buffer)

A different communicator (comm2) uses the same buffer. Rank0 sends to Rank1 again.

```
  Rank0                                          Rank1
  ─────────────────────────────────────────      ─────────────────────────────
  GPE Thread (collective on comm2)               (idle, waiting for ctrl msg)
    │
    └─ mapper2->searchRegHandle(buf, size)
         │
         ▼
  RegCache
    │
    ├─ regRange(buf, size, enableBackends)
    │    ├─ FAST PATH: rlock regElemsMaps_
    │    │   searchRegElem(buf, size) → HIT!
    │    │   (finds existing RegElem from Step 2)
    │    │
    │    └─ return existing RegElem*
    │       (no new registration needed)
         │
         │ same regHdl as Step 2
         ▼
  Mapper2 (comm2)                                Mapper2 (comm2)
    │                                              │
    ├─ exportMem(regHdl, rank=1)                   │
    │   → same ipcRegElem as Step 2                │
    │   → IpcDesc{handle, offset, uid=1}           │
    │     (SAME uid -- same backing memory)        │
    │                                              │
    ├─ exportRegCache_.record(regElem, peerRank=1)  │
    │   (mapper2 now also tracks this export)       │
    │                                              │
    └─ send ctrl msg ─────────────────────────►    ├─ recv ctrl msg
                                                   │
                                                   └─ importMem(IpcDesc)
                                                        │
                                                        ▼
                                                   IpcRegCache (singleton)
                                                     │
                                                     ├─ importRemMemImpl(
                                                     │    peerId=Rank0,
                                                     │    IpcDesc{hdl, uid=1})
                                                     │
                                                     ├─ cache key = {Rank0,
                                                     │    {base, uid=1}}
                                                     ├─ lookup → HIT!
                                                     │
                                                     ├─ refCount.fetch_add(1)
                                                     │   refCount: 1 → 2
                                                     │
                                                     └─ return existing mapped
                                                        pointer (no new open)
```

**State after Step 3:**
- Rank0 RegCache: 1 Segment, 1 RegElem (unchanged)
- Rank0 Mapper1 exportRegCache_: {regElem → [Rank1]}
- Rank0 Mapper2 exportRegCache_: {regElem → [Rank1]}
- Rank1 IpcRegCache: 1 IpcRemRegElem(refCount=**2**)

---

## Step 4a: Deregistration via Communicator Destruction

Both comms are destroyed before the memory is freed. The mapper destructor
iterates `exportRegCache_` and calls `remReleaseMem` for each entry before
calling `deregisterExportClient`.

```
  Rank0                                          Rank1
  ─────────────────────────────────────────      ─────────────────────────────
  ~CtranMapper() [mapper1]
    │
    ├─ iterate exportRegCache_.dump()
    │    └─ remReleaseMem(regElem)
    │         └─ notifyRemote(rank=1, uid=1)
    │              │
    │              │ kRelease{uid=1}
    │              └──────────────────────────►   releaseRemReg(Rank0, uid=1)
    │                                              refCount: 2 → 1
    │                                              (keep entry)
    ├─ deregisterExportClient(this)
    │   (removes mapper1 from client list)
    │
    └─ mapper1 destroyed

  State:                                         IpcRegCache:
    Mapper1: destroyed                             {Rank0, {base, uid=1}} →
    Mapper2: still active                            IpcRemRegElem(refCount=1)
    RegCache: unchanged
```

```
  Rank0                                          Rank1
  ─────────────────────────────────────────      ─────────────────────────────
  ~CtranMapper() [mapper2]
    │
    ├─ iterate exportRegCache_.dump()
    │    └─ remReleaseMem(regElem)
    │         └─ notifyRemote(rank=1, uid=1)
    │              │
    │              │ kRelease{uid=1}
    │              └──────────────────────────►   releaseRemReg(Rank0, uid=1)
    │                                              refCount: 1 → 0
    │                                              cuIpcCloseMemHandle
    │                                              erase entry
    ├─ deregisterExportClient(this)
    │
    └─ mapper2 destroyed

  State:                                         IpcRegCache:
    Both mappers: destroyed                        ipcRemRegMap_: (empty)
    RegCache: unchanged                            Rank0's memory unmapped
    (segment still cached)
```

After both mappers are destroyed, the user calls `globalDeregister` + `cudaFree`:

```
  Rank0                                          Rank1
  ─────────────────────────────────────────      ─────────────────────────────
  User Code                                      (idle, already cleaned up)
    │
    ├─ globalDeregister(buf, size)
    │    │
    │    ▼
    │  RegCache
    │    ├─ lookupSegments → [segment]
    │    ├─ releaseFromAllClients(regElem)
    │    │   → NO active mappers (both
    │    │     deregistered) → nothing to do
    │    └─ freeSegment
    │         ├─ remove from AVL
    │         ├─ deregister (IB, IPC)
    │         └─ delete Segment
    │
    └─ cudaFree(buf)

  State:                                         State:
    RegCache: empty                                IpcRegCache: empty
    Memory: freed                                  No dangling references
```

---

## Step 4b: Deregistration via Memory Free (globalDeregister)

Alternative path: memory is freed while both communicators are still
active. `globalDeregister` forces all mappers to release remote
registrations before freeing segments.

```
  Rank0                                          Rank1
  ─────────────────────────────────────────      ─────────────────────────────
  User Code                                      (idle)
    │
    └─ globalDeregister(buf, size)
         │
         ▼
  RegCache
    │
    ├─ lookupSegmentsForBuffer → [segment]
    │
    ├─ for each regElem of segment:
    │
    │  releaseFromAllClients(regElem)
    │    │
    │    ├─ iterate active IpcExportClients:
    │    │  [mapper1, mapper2]
    │    │
    │    ├─ mapper1->remReleaseMem(regElem)
    │    │    └─ notifyRemote(rank=1, uid=1)
    │    │         │
    │    │         │ kRelease{uid=1}
    │    │         └─────────────────────────►   releaseRemReg(Rank0, uid=1)
    │    │                                        refCount: 2 → 1
    │    │                                        (keep entry)
    │    │
    │    └─ mapper2->remReleaseMem(regElem)
    │         └─ notifyRemote(rank=1, uid=1)
    │              │
    │              │ kRelease{uid=1}
    │              └─────────────────────────►   releaseRemReg(Rank0, uid=1)
    │                                              refCount: 1 → 0
    │                                              cuIpcCloseMemHandle
    │                                              erase entry
    │
    ├─ freeSegment
    │    ├─ remove from AVL
    │    ├─ deregister (IB, IPC)
    │    └─ delete Segment
    │
    └─ done

  User Code                                      IpcRegCache:
    └─ cudaFree(buf)                               ipcRemRegMap_: (empty)
                                                   Rank0's memory unmapped
  State:
    RegCache: empty
    Both mappers: still active
    (exportRegCache_ entries cleaned up)
```

**Key difference from Step 4a:** The mappers survive. When the comms are
later destroyed, their destructors find no exports for this regElem in
`exportRegCache_` (already cleaned up by `releaseFromAllClients`), so
no duplicate release messages are sent.

---

## Summary: Two Cleanup Paths

| Path | Trigger | Who sends kRelease | Rank1 cleanup | Segments freed |
|------|---------|-------------------|------------|----------------|
| **4a** (comm dies first) | `~CtranMapper()` | Mapper destructor iterates `exportRegCache_` | Correct (refCount → 0) | `globalDeregister` later (no active mappers to notify) |
| **4b** (memory freed first) | `globalDeregister` | `releaseFromAllClients` iterates active mappers | Correct (refCount → 0) | Immediately after release |

Both paths work correctly. When either completes, both ranks reach the
clean end state:
- Rank0: Segment removed from AVL, RegElem deregistered, memory freed
- Rank1: IpcRemRegElem refCount reaches 0, CUDA IPC handle closed, entry erased

---

## Key Code References

| Operation | File | Function |
|-----------|------|----------|
| globalRegister | `RegCache.cc` | `RegCache::globalRegister()` |
| cacheSegment | `RegCache.cc` | `RegCache::cacheSegment()` |
| regRange (lazy register) | `RegCache.cc` | `RegCache::regRange()` |
| searchRegHandle | `CtranMapper.cc` | `CtranMapper::searchRegHandle()` |
| exportMem | `CtranMapper.cc` | `CtranMapper::exportMem()` |
| importMem | `CtranMapper.cc` | `CtranMapper::importMem()` |
| importRemMemImpl (refCount++) | `IpcRegCache.cc` | `IpcRegCache::importRemMemImpl()` |
| remReleaseMem (send kRelease) | `CtranMapper.cc` | `CtranMapper::remReleaseMem()` |
| releaseRemReg (refCount--) | `IpcRegCache.cc` | `IpcRegCache::releaseRemReg()` |
| releaseFromAllClients | `IpcRegCache.cc` | `IpcRegCache::releaseFromAllClients()` |
| globalDeregister | `RegCache.cc` | `RegCache::globalDeregister()` |
| freeSegment | `RegCache.cc` | `RegCache::freeSegment()` |
| mapper destructor | `CtranMapper.cc` | `CtranMapper::~CtranMapper()` |
