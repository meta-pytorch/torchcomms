# RegCache Design

## Overview

RegCache is a **global singleton** that manages GPU memory registration for
RDMA (InfiniBand), IPC (NVLink), and TCP device memory backends in ctran.
It decouples **caching** (tracking known memory segments) from **registering**
(binding segments to backend hardware), enabling four registration modes:
none, eager, lazy, and async.

The cache is shared across all communicators in a process. Individual
communicators interact with it through `CtranMapper`, which handles
per-communicator concerns like remote peer notification.

## Architecture

```
 +-------------------------------+
 |      PyTorch / User Code      |
 +----+------+---------+----+----+
      |      |         |    |
      |  ncclMemAlloc  | ncclMemFree
      |      |         |    |
      |      v         |    v
      |  globalRegister|  globalDeregister
      |      |         |    |
      |      |         |    |    (notifies all mappers
      |      |         |    |     to release remote regs,
      |      |         |    |     then frees segments)
      |      |         |    |
      v      v         v    v
 +----+------+---------+----+----+     +--------------------+
 |      RegCache (Singleton)     |     | IpcRegCache        |
 |                               |     | (Singleton)        |
 |  +-------------------------+  |     |                    |
 |  | segmentsAvl_            |  |     | ipcRemRegMap_      |
 |  | (AVL Tree)              |  |     | (imported remote   |
 |  |  [Segment]--[Range]     |  |     |  registrations)    |
 |  +-------------------------+  |     |                    |
 |                               |     | asyncServerSocket  |
 |  +-------------------------+  |     | (IPC release msgs) |
 |  | regElemsMaps_           |  |     +--------+-----------+
 |  |  regHdlToElemMap        |  |              |
 |  |  segToRegElemsMap       |  |              |
 |  +-------------------------+  |              |
 |                               |              |
 |  asyncRegThread_              |              |
 +----+------+------+------------+              |
      |      |      |                           |
      v      v      v                           v
  +------+ +----+ +------+             +---------------+
  |CtranIb| |Ipc | |TcpDm|             | CtranMapper   |
  | (IB)  | |Reg | |(TCP)|             | (per-comm)    |
  +-------+ +----+ +-----+             |               |
                                        | regMem()      |
  ncclCommRegister ---> regMem -------->|  -> cacheSegment
  ncclCommDeregister -> deregMem ------>|  -> remReleaseMem
                                        |  -> freeSegment
  collective time ----> searchRegHandle>|  -> regRange
                                        | exportCache_  |
                                        +---------------+
```

## Core Data Structures

### SegmentRange

A discovered physical memory range. Created by `pinRange()`, which uses
`cuMemGetAddressRange` to find the underlying physical allocation for a
virtual address. A single user buffer can span multiple physical segments
(especially with CUDA virtual memory / `cumem`).

```
SegmentRange
  buf  : const void*     -- base pointer of the physical segment
  len  : size_t          -- byte length
  type : DevMemType      -- kCudaMalloc, kCumem, kHostUnregistered, etc.
```

### Segment

A cached physical memory segment stored in the AVL tree. One user buffer
maps to one or more Segments.

```
Segment
  range      : SegmentRange
  cudaDev    : int          -- CUDA device ID
  ncclManaged: bool         -- whether NCCL allocated this buffer
  stateMnger : Synchronized<SegmentStateMnger>  -- refCount for multi-comm caching
  avlHdl_    : void*        -- opaque handle into the CtranAvlTree
```

### RegElem

A backend registration covering one or more Segments. Holds opaque
handles from each backend (IB, IPC, TCP).

```
RegElem
  buf        : const void*  -- registered address range start
  len        : size_t       -- registered address range length
  ibRegElem  : void*        -- IB backend handle
  ipcRegElem : void*        -- IPC/NVL backend handle
  tcpRegElem : void*        -- TCP device memory handle
  stateMnger : Synchronized<RegElemStateMnger>  -- REGISTERED / DEREGISTERED
  cudaDev_   : int                              -- CUDA device ID
  segments_  : vector<Segment*>                 -- backing segments
  isDynamic_ : bool         -- true for one-time uncached registrations
  type_      : DevMemType   -- memory type of the registered range
  ncclManaged_: bool        -- whether NCCL allocated this buffer
  inUseCnt    : atomic<int64_t>  -- live scoped-use count; does not control RegElem lifetime (see Scoped Registration)
```

### RegCache (class)

The singleton. Key private state:

```
RegCache
  segmentsAvl_   : Synchronized<CtranAvlTree>   -- segment cache (O(log N) lookup)
  regElemsMaps_  : Synchronized<RegElemMaps>     -- registration maps
    regHdlToElemMap  : unordered_map<RegElem*, unique_ptr<RegElem>>  -- ownership
    segToRegElemsMap : unordered_map<Segment*, vector<RegElem*>>   -- segment-to-reg correlation
  asyncRegThread_: thread                        -- background async registration
  globalBackends_: vector<bool>                  -- from NCCL_CTRAN_BACKENDS env
  ibSingleton_   : shared_ptr<CtranIbSingleton>  -- prevents use-after-free
```

## Registration Modes

Controlled by the `NCCL_CTRAN_REGISTER` environment variable:

| Mode  | Behavior |
|-------|----------|
| none  | ctran registration disabled entirely |
| eager | `regMem` caches AND registers with backends immediately |
| lazy  | `regMem` only caches; registration deferred to `searchRegHandle` at collective time |
| async | `regMem` caches; `regAsync` submits background registration; `searchRegHandle` waits for completion |

## Key Operations

### Caching: `cacheSegment()`

Discovers the physical segments underlying a buffer and inserts them into
the AVL tree. On cache hit (same physical segment already cached), returns
the existing Segment. One buffer may produce multiple Segments for `cumem`
allocations.

```
cacheSegment(buf, len, cudaDev) -> [Segment*, segHdl]

  1. pinRange(buf, cudaDev, len) -> [SegmentRange...]
  2. For each SegmentRange:
     a. AVL search by (base, len)
     b. Hit  -> reuse existing Segment
     c. Miss -> create Segment, AVL insert
  3. Return segments and their AVL handles
```

### Registration: `regRange()`

Registers a buffer range with backends. Uses a double-check pattern for
performance: fast read-lock path for lookup hits, slow write-lock path
for new registrations.

```
regRange(ptr, len, backends) -> RegElem*

  FAST PATH (regElemsMaps_ read lock):
    searchRegElem(ptr, len)
    If found -> return existing RegElem (lookup hit)

  SLOW PATH (segmentsAvl_ write lock):
    Re-check searchRegElem (double-check)
    pinRange -> discover physical segments
    For each segment: verify cached in AVL tree
    If all cached -> registerSegmentsTogether() -> new RegElem
    If any missing -> return nullptr (caller uses regDynamic)
```

### Segment Freeing: `freeSegment()`

Removes a cached segment and all associated registrations. Called by
`globalDeregister()` when physical memory is freed, and by
`CtranMapper::deregMem()` for per-communicator cleanup.

The segment uses reference counting (`SegmentStateMnger`): each
`cacheSegment` call increments the refcount, and `freeSegment`
decrements it. The segment is only actually freed when the refcount
reaches zero. If the segment handle is not found (already freed),
`freeSegment` is a no-op and returns success.

An optional `forceFree` parameter (default `false`) bypasses the
refcount check and always frees the segment. `globalDeregister` uses
`forceFree=true` because the underlying physical memory is about to be
freed, so the segment must be removed from cache regardless of how many
communicators have cached it.

```
freeSegment(segHdl, forceFree=false) -> freed, regElems

  1. Acquire both segmentsAvl_ and regElemsMaps_ locks
  2. Lookup Segment from AVL handle
     - Not found -> return (freed=false, commSuccess)
  3. If !forceFree: lock SegmentStateMnger
     - refCount > 1 -> decrement and return (freed=false, commSuccess)
     - refCount == 1 -> validate associated RegElems have inUseCnt == 0
       before setting refCount to 0
     - inUseCnt > 0 -> return commInvalidUsage without changing refCount
  4. Collect all RegElems via segToRegElemsMap
  5. Transfer ownership from regHdlToElemMap to output vector
  6. Remove Segment from AVL tree
  7. Release locks
  8. doDeregister() on each RegElem (IB, IPC, TCP)
  9. Delete Segment
```

### Dynamic Registration: `regDynamic()` / `deregDynamic()`

One-time registration for buffers NOT in the cache. Used when
`regRange()` returns nullptr (buffer wasn't pre-cached by the user).
Dynamic registrations are NOT correlated with segments and must be
explicitly deregistered after the collective completes.

### Bulk Re-registration: `regAll()` / `deregAll()`

Global APIs for bulk registration management. Used for BAR1 memory
management where all registrations need to be torn down and recreated
(e.g., to reclaim BAR1 space, then re-register when needed).

**`regAll()`** registers all cached segments with backends in bulk.
It discovers contiguous memory regions among cached segments and
creates one registration per region, reducing the number of backend
registration calls.

```
regAll()

  1. Acquire segmentsAvl_ write lock
  2. Get all Segments from AVL tree
  3. Sort segments by starting address
  4. Group into contiguous regions (adjacent segments where
     end address of one == start address of next)
  5. For each contiguous region:
     registerSegmentsTogether(regionPtr, regionLen, segments)
       -> doRegister(IB, IPC, TCP)
       -> add RegElem to regHdlToElemMap and segToRegElemsMap
  6. Release lock

  Example: segments at [0x1000-0x2000], [0x2000-0x3000], [0x5000-0x6000]
  -> Region 1: [0x1000-0x3000] (2 segments, 1 registration)
  -> Region 2: [0x5000-0x6000] (1 segment, 1 registration)
```

**`deregAll()`** removes all non-dynamic registrations but preserves
the cached segments. This allows segments to be re-registered later
via `regAll()` without needing to re-cache them.

```
deregAll()

  1. Acquire regElemsMaps_ write lock
  2. Iterate all RegElems:
     - Skip dynamic registrations (isDynamic_)
     - Remove non-dynamic RegElems from segToRegElemsMap
     - Transfer ownership to toDeregister vector
     - Remove from regHdlToElemMap
  3. Release lock
  4. For each collected RegElem:
     releaseFromAllClients() (notify remote peers via IPC)
  5. For each collected RegElem:
     deregElem() -> doDeregister(IB, IPC, TCP)
```

Key design points:
- **Segments are preserved**: `deregAll` only removes registrations,
  not the underlying cached segments. The AVL tree is untouched.
- **Dynamic registrations are preserved**: `deregAll` skips RegElems
  with `isDynamic_=true`, so one-time collective registrations are
  not affected.
- **No duplicate check in `regAll`**: Callers must call `deregAll()`
  before `regAll()` to avoid duplicate registrations.
- **IPC release**: `deregAll` notifies remote peers to release
  imported NVL memory before deregistering, preventing dangling
  remote references.

## Memory Lifecycle

There are two distinct paths for memory cleanup, depending on who
owns the buffer:

### User Buffers (PyTorch-managed)

User allocates with `cudaMalloc` or CUDA VMM APIs. PyTorch calls
`ncclMemFree` when done, which triggers `globalDeregister`.

```
ncclMemAlloc / cudaMalloc
  -> globalRegister -> cacheSegment [+ regRange if eager]

ncclCommRegister (per communicator)
  -> CtranMapper::regMem -> cacheSegment [+ regRange if eager]

[... collective operations use searchRegHandle ...]

ncclCommDeregister (per communicator)
  -> CtranMapper::deregMem
     -> remReleaseMem (notify remote peers)
     -> freeSegment (decrement segment refCount; freed if last ref)

ncclMemFree
  -> globalDeregister
     -> IpcRegCache::releaseFromAllClients (notify all mappers)
     -> freeSegment (remove from cache, deregister backends)
```

### NCCL-Managed Buffers (internal)

NCCL allocates temporary buffers internally (window, BufManager,
AllToAllDedup, CtranAlgo). These use `deregMem` which handles both
remote release and segment cache cleanup.

```
internal cudaMalloc
  -> CtranMapper::regMem -> cacheSegment [+ regRange]

[... internal use ...]

cleanup:
  -> CtranMapper::deregMem (notify remote peers + freeSegment)
  -> cudaFree
```

## Scoped Registration

Ctran's persistent collectives (persistent AllGatherP, "AGP") and window
collectives hold a registration for the lifetime of an operation, request, or
window — not the lifetime of the underlying memory — and must release it safely,
including from a CUDA graph-destroy callback where no CUDA calls are allowed.
Move-only RAII owners provide this on top of the existing cache without
disturbing the allocator (CCA) hook or the existing, non-refcounted lookup APIs.

`globalRegister` / `globalDeregister` are the allocator-hook APIs. They are keyed
on BUFFER (memory) lifetime — the allocator calls them around a buffer's
allocation and free — and `globalDeregister` force-frees the cached segment and
all of its registrations. They are therefore NOT suitable for scoped
registration inside Ctran: a persistent collective or window has an operation /
request / window lifetime, not a memory lifetime, and calling `globalDeregister`
from that code would tear down allocator-owned state that other consumers may
still need. Ctran-internal scopes must use `acquireScopedRegister` /
`ScopedRegHdl` (below) instead; `globalRegister` / `globalDeregister` must be
called ONLY from the allocator side (the CCA memory hook), never from collective
or window code.

### Local registration: `ScopedRegHdl`

Use tracking (`RegElem::inUseCnt`):

- `RegElem` lifetime is controlled by allocator hooks and explicit teardown, not
  by `inUseCnt`. A cached registration starts with `inUseCnt = 0` when it is
  created via `globalRegister` / the CCA hook or by first scoped use of an
  already-cached segment.
- `RegCache::acquireScopedRegister(buf, len, cudaDev, backends, &hdl)` resolves
  the cached `RegElem`, increments `inUseCnt`, and returns a move-only
  `ScopedRegHdl` owning that one use-side reference.
- Dynamic registrations (`regDynamic` / internal `regRange`) do not acquire
  `inUseCnt`. The explicit `deregDynamic` / internal `deregRange` call deletes
  the RegElem only if `inUseCnt == 0`; a non-zero `inUseCnt` there is invalid
  usage and returns `commInvalidUsage`.
- `~ScopedRegHdl` performs a pure software decrement (`releaseScopedRegHdl`)
  under the `regElemsMaps_` read lock — no deregistration, no delete, no CUDA —
  so it is safe to run from a graph-destroy callback.
- If `deregRange` or non-force `freeSegment` attempts to delete a registration
  while `inUseCnt > 0`, it returns `commInvalidUsage`. The allocator hook
  (`globalDeregister` -> force `freeSegment`) must not block memory free; it
  logs and force-frees the registration because freeing the buffer while scoped
  use is still live is already invalid usage. These errors are not retry
  contracts: mapper-side release may already have partially run, and regcache
  only preserves enough local state for whole-cache failure cleanup or
  allocator force-free to reclaim memory.

Compatibility with the allocator (CCA) hook:

- The allocator hook owns memory lifetime: `globalRegister` caches the segment
  and may create the `RegElem`; `globalDeregister` force-frees cached segments
  and associated backend registrations right before the memory is freed.
- `acquireScopedRegister` REQUIRES the buffer's segment to already be
  allocator-cached and returns `commInvalidUsage` otherwise — it never caches or
  frees segments. A scoped acquire/release only adjusts `inUseCnt`, so it can
  neither race nor double-free allocator-owned state; the allocator remains the
  owner of the registration's lifetime. If a buffer is freed while scoped use is
  still live, force `freeSegment` logs a contract-violation warning and tears
  down anyway (the memory is going away).

Compatibility with the existing (non-refcounted) lookup APIs:

- `searchRegHandle` / `searchIbRegHandle` / `getRegHandle` return a borrowed
  `RegElem*` WITHOUT touching `inUseCnt`. These are the established production
  callers (e.g. collective-time handle lookup) and are intentionally left
  uncounted because there is no matching release API. Their safety still comes
  from the allocator-hook contract: the buffer allocation must outlive borrowed
  lookup use.

### Remote IPC import: `ScopedIpcRegHdl`

Remote NVLink IPC imports live in `IpcRegCache`, separate from the local segment
cache, and use a symmetric-but-different scoped owner.

Lifecycle:

- An imported `IpcRemRegElem` is refcounted (`refCount`). `releaseRemReg` ALWAYS
  defers: it decrements the refcount and, on reaching zero, moves the elem out of
  the live map into `invalidImports_`. This is pure software work (no CUDA), so
  it is safe from a graph-destroy callback. Because rc==0 entries leave the live
  map, a released import is never resurrected by a later import.
- `cleanupInvalidImports()` performs the actual CUDA unmap of the parked
  entries. It runs on user-thread entries (`importMem`, `~IpcRegCache`) and is
  called explicitly by release callsites that want prompt teardown
  (`CtranMapper::deregRemReg`, the async-socket `kRelease` handler).
- `ScopedIpcRegHdl` is a move-only owner of one import reference, produced
  directly by `IpcRegCache::importMem` via its optional `outHdl` out-param: the
  single reference the import adds is ADOPTED by the returned handle (no extra
  reference is added). `~ScopedIpcRegHdl` performs a deferred `releaseRemReg`, so
  it is graph-destroy safe; the CUDA unmap drains later via an explicit
  `cleanupInvalidImports()` at a user-thread teardown point (window free, eager
  AGP destroy).

Difference from local `ScopedRegHdl`, and compatibility:

- `ScopedIpcRegHdl` owns a real import ref. Its last release CAN retire the
  import (deferred). Local `ScopedRegHdl` only owns a `RegElem::inUseCnt`
  use-side reference; releasing it never retires the local registration because
  local registration lifetime is controlled by allocator hooks or explicit
  teardown.
- `IpcRemRegElem` was already refcount-based for existing importers, and its
  release can be triggered either by an explicit importer-side call
  (`releaseRemReg` / `CtranMapper::deregRemReg`) or by the exporter's `kRelease`
  notification over the async socket. `ScopedIpcRegHdl` layers cleanly on top of
  this existing refcount — it introduces no new scheme and does not change the
  C-style release/notify paths.

## Component Interactions

### CtranMapper (per-communicator)

Each communicator has a `CtranMapper` that coordinates with the global
RegCache singleton. It implements `IpcExportClient` for IPC release
notifications.

```
CtranMapper
  regMem()          -> RegCache::cacheSegment [+ regRange]
  deregMem()        -> RegCache::getRegElems -> remReleaseMem per elem
                       -> RegCache::freeSegment (decrement segment refCount)
  searchRegHandle() -> RegCache::regRange [or regDynamic as fallback]
  remReleaseMem()   -> send IPC release to remote peers via AsyncSocket
```

### IpcRegCache (NVLink/IPC)

A separate singleton managing IPC memory registrations for NVLink peer
access. Handles CUDA IPC handle import/export and remote release
messaging.

```
IpcRegCache
  regMem()                  -> create IpcRegElem (CUDA IPC handle)
  deregMem()                -> destroy IpcRegElem
  importMem()               -> import remote memory via IPC descriptor
  releaseRemReg()           -> release imported remote registration
  releaseFromAllClients()   -> notify all mappers to send remote release
  notifyRemoteIpcRelease()  -> send async socket message to remote peer
```

### Interaction Diagram: Collective-time Registration

```
GPE Thread                    CtranMapper           RegCache
    |                              |                    |
    |-- searchRegHandle(buf) ----->|                    |
    |                              |-- regRange(buf) -->|
    |                              |                    |-- rlock: searchRegElem
    |                              |                    |   (fast path hit?)
    |                              |                    |
    |                              |                    |-- [if miss] wlock:
    |                              |                    |   pinRange + AVL lookup
    |                              |                    |   registerSegmentsTogether
    |                              |                    |     -> doRegister(IB,IPC,TCP)
    |                              |<-- RegElem* ------|
    |<-- regHdl ------------------|                    |
```

### Interaction Diagram: Memory Free (globalDeregister)

```
PyTorch                 RegCache              IpcRegCache         CtranMapper(s)
   |                       |                      |                    |
   |-- globalDeregister -->|                      |                    |
   |                       |-- lookupSegments --->|                    |
   |                       |                      |                    |
   |                       |-- releaseFromAll --->|                    |
   |                       |                      |-- remReleaseMem -->|
   |                       |                      |   (for each        |
   |                       |                      |    active mapper)  |
   |                       |                      |                    |
   |                       |-- freeSegment ------>|                    |
   |                       |   (remove from AVL,  |                    |
   |                       |    deregister IB/IPC)|                    |
   |<-- done --------------|                      |                    |
```

## Thread Safety

| Lock | Type | Protects |
|------|------|----------|
| `segmentsAvl_` | `Synchronized<CtranAvlTree>` | AVL tree of cached Segments |
| `regElemsMaps_` | `Synchronized<RegElemMaps>` | regHdlToElemMap + segToRegElemsMap |
| `RegElem::stateMnger` | `Synchronized<RegElemStateMnger>` | Per-RegElem state transitions |
| `asyncRegQueue_` | `Synchronized<queue, mutex>` | Async registration command queue |

Key patterns:
- **Double-check in `regRange()`**: Fast read-lock path for lookup hits,
  slow write-lock for new registrations
- **Atomic dual-lock in `freeSegment()`**: Uses `folly::acquireLocked`
  to acquire both `segmentsAvl_` and `regElemsMaps_` atomically
- **Backend ops outside locks**: `doRegister()` and `doDeregister()` run
  outside the global locks to avoid holding them during expensive RDMA
  operations

## Singleton Lifecycle

```
Program start
  -> RegCache::init()
       -> acquire CtranIbSingleton reference
       -> initialize globalBackends_ from NCCL_CTRAN_BACKENDS
       -> start asyncRegThread_ (if async mode)

  -> IpcRegCache::init()
       -> start AsyncSocket server for IPC messages

[... program runs, communicators created/destroyed ...]

Program exit
  -> CtranMapper instances destroyed (per-comm)
  -> RegCache::destroy()
       -> deregister all remaining RegElems
       -> remove all Segments from AVL tree
       -> stop asyncRegThread_
       -> release CtranIbSingleton reference
  -> IpcRegCache destroyed
       -> stop AsyncSocket server
  -> CtranIbSingleton destroyed (last reference gone)
```
