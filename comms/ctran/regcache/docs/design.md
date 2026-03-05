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
`freeSegment` returns `commInvalidUsage`.

```
freeSegment(segHdl) -> freed, regElems

  1. Acquire both segmentsAvl_ and regElemsMaps_ locks
  2. Lookup Segment from AVL handle
     - Not found -> return commInvalidUsage (error)
  3. askFree() -> decrement refCount
     - refCount > 0 -> return (freed=false, commSuccess)
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

## Component Interactions

### CtranMapper (per-communicator)

Each communicator has a `CtranMapper` that coordinates with the global
RegCache singleton. It implements `IpcExportClient` for IPC release
notifications.

```
CtranMapper
  regMem()          -> RegCache::cacheSegment [+ regRange]
  deregMem()        -> RegCache::getRegElems -> remReleaseMem per elem
                       -> RegCache::freeSegment (decrement refCount)
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
