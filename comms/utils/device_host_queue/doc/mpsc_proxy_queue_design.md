# DeviceHostQueue Design (MPSC / MPMC)

## Summary

`DeviceHostQueue<T, Direction, ConsumerPolicy>` is a bounded, lock-free typed ring in host-pinned mapped memory for passing fixed-size proxy commands **one way from GPU to CPU**: the GPU delegates work to a CPU/engine proxy that executes it. GPU threads are the producers; the CPU is the consumer.

The queue is intentionally independent of IBRC, copy-engine, or UniFlow descriptors. Backends provide the payload type `T` and decide what a consumed item means. (The IBRC proxy descriptor ring in `ibrc_proxy_design.md` is a specialization of this same protocol with `T = IbrcProxyDesc`.)

It is header-only and lives at `comms/utils/device_host_queue/DeviceHostQueue.cuh`. It is portable across CUDA and HIP/ROCm (see Portability).

Data flow:

```text
GPU/device producers -> host CPU consumer(s)
```

Two consumer policies are selected at compile time via the `ConsumerPolicy` template parameter:

- **`Single` (MPSC)** — one CPU consumer drains in strict sequence order via a private cursor.
- **`Multi` (MPMC)** — several CPU consumer threads drain concurrently; each published command is delivered to exactly one consumer, claimed in FIFO order. Processing across consumers is not globally ordered, but there is no loss or duplication.

The GPU producer protocol is identical for both policies.

> **Before you pick `Multi`, read "MPSC vs MPMC — the throughput inversion" below.** On a fast cross-domain link (GB200/NVLink-C2C) `Multi` is *slower* than `Single` for pure dequeue; it is for *processing*-bound consumers, not for higher dequeue throughput.

## Goals

- Provide a typed bounded queue in host-pinned mapped memory.
- Support many GPU producers reserving slots concurrently.
- Support either one CPU consumer (`Single`) or many concurrent CPU consumers (`Multi`).
- Preserve FIFO by logical sequence number (claim order under `Multi`).
- Keep the queue payload independent from backend protocol fields.
- Keep the GPU producer protocol policy-independent.
- Keep all consumer claim/completion bookkeeping off the cross-domain ring.
- Run on both NVIDIA (CUDA) and AMD (HIP/ROCm).

## Non-Goals

- Host producers or device consumers — the data path is strictly GPU → CPU. This is enforced by the `Direction` template parameter: only `Direction::D2H` (device producer → host consumer) is implemented; `Direction::H2D` is rejected at compile time via `static_assert`. H2D is not a drop-in: the host would become the publisher and would have to carry the system-scope half of the fence with an explicit `dmb sy` (aarch64), which `std::atomic`'s inner-shareable release does not provide — see Memory Model.
- Dynamic resizing.
- Non-trivially-copyable payloads.
- Cross-process queue sharing.
- A status/abort/timeout/heartbeat machinery. The queue is minimal by design: the blocking variants spin (with a relax hint, see Blocking variants) with no escape; bounded waits, abort, and teardown belong to the caller's progress loop / lifecycle, not to the FIFO.
- A generic replacement for backend completion queues or token/resource pools.

## When To Use This

Reach for `DeviceHostQueue` when a GPU kernel must hand work to a CPU/host agent *while the kernel keeps running*, and you want a typed, bounded, allocation-free, lock-free hand-off:

- **vs `cudaLaunchHostFunc` / stream callbacks** — those are stream-ordered, one-shot, coarse-grained, and cannot deliver work mid-kernel; this queue streams many fine-grained commands from a live kernel.
- **vs a hand-rolled flag in unified/managed memory** — this gives you a real bounded FIFO with credit-based backpressure, a torn-publish-safe publish protocol, and FIFO/exactly-once semantics, instead of a single ad-hoc flag.
- **vs a backend completion queue** — this is the generic transport; the backend layers its own descriptor type and completion accounting on top (see IBRC proxy).

## Type Requirements

`T` must be plain data:

```cpp
static_assert(std::is_trivially_copyable_v<T>);
static_assert(std::is_trivially_destructible_v<T>);
```

Payloads should be fixed-size and cacheline-conscious. Recommended payload sizes for proxy commands are 16B–256B; throughput is ops-bound across that range (see Performance), so larger payloads cost little extra until you hit link bandwidth.

`T` carries only the backend payload; it MUST NOT carry queue mechanics. The publish marker `readySeq` is owned by the queue and lives in the slot (`DeviceHostQueueSlot::readySeq`), never inside `T`. Publication writes the payload, then performs a *separate, system-scope release* store of the marker (see Sequence Protocol). A marker embedded in `T` could not implement that protocol — it would be copied as part of the payload blob with no ordering between payload and marker, and as a plain (non-atomic) field, so it could be observed torn or out of order.

## API

Device producer handle — returned by `producerHandle()`, passed BY VALUE into kernels. It holds no cursor, so it is safe to broadcast to every producer thread; `write()`/`blockingWrite()` may be called concurrently by any number of threads.

```cpp
template <class T, ConsumerPolicy Policy = ConsumerPolicy::Single>
class DeviceHostQueueProducer {
 public:
  __host__ __device__ uint32_t capacity() const;
  __device__ uint32_t sizeApprox() const;

  // Reserves a slot only if credit is available, so a Full return never leaves
  // an unpublished hole. Returns Ok or Full.
  __device__ QueueOpStatus write(const T& value);

  // Reserve a slot, spin for reuse credit, then publish. No escape.
  __device__ void blockingWrite(const T& value);
};
```

Host owner / CPU consumer — the RAII owner of the allocation and the consumer side.

```cpp
template <
    class T,
    Direction Dir = Direction::D2H,
    ConsumerPolicy Policy = ConsumerPolicy::Single>
class DeviceHostQueue {
 public:
  explicit DeviceHostQueue(uint32_t capacity); // power of two

  uint32_t capacity() const;
  size_t sizeApprox() const;

  struct ReadResult { T value; uint64_t seq; };

  // Copies the next published command out and releases its slot (advances ci)
  // for producer reuse — copy-out frees the slot. Returns Ok or Empty.
  // Single: call from one consumer only. Multi: safe from many threads; each
  // command is returned to exactly one caller, claimed in FIFO order.
  QueueOpStatus read(ReadResult& out);

  // Batched MPSC dequeue: copies up to `max` contiguous published commands into
  // out[0..n) and returns n, releasing the whole batch with a SINGLE ci store.
  // MPSC only (compile error under Multi). See "Batched dequeue".
  size_t readMulti(ReadResult* out, size_t max);  // Policy == Single only

  void blockingRead(ReadResult& out);  // spins with a cpuRelax() hint

  // Cursor-free, trivially-copyable device producer handle. Pass BY VALUE.
  Producer producerHandle() const;
};
```

Device `read` is not exposed: the consumer is always the CPU owner. `read()` writes the payload through `ReadResult` rather than returning `std::optional<T>`, keeping the contract uniform and copy-free.

Ownership contract:

```text
producerHandle().write()/blockingWrite():
  may be called by many GPU producer threads concurrently

read()/readMulti()/blockingRead():
  Single: exactly one CPU consumer
  Multi:  read()/blockingRead() from any number of CPU consumer threads;
          readMulti() is MPSC-only
```

## Memory Layout

Queue storage lives in one contiguous host-pinned mapped allocation:

```cpp
// flags = cudaHostAllocMapped on CUDA;
//       = hipHostMallocMapped | hipHostMallocCoherent on HIP/AMD (see Portability)
cudaHostAlloc(&host_ptr, bytes, flags);
cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
```

Shared, CPU↔GPU-mapped layout. Every word touched across the boundary is a PLAIN integer; atomicity and ordering are applied at the access site (see Memory Model).

```cpp
template <class T>
struct alignas(64) DeviceHostQueueSlot {
  alignas(alignof(T)) std::byte payload[sizeof(T)];
  uint64_t readySeq; // publish marker; MUST stay LAST (written after payload)
};

template <class T>
struct alignas(64) DeviceHostQueueControl {
  alignas(64) uint64_t pi; // producer reservation
  alignas(64) uint64_t ci; // reuse credit
};
// slots[] follows immediately after the control header in the same allocation.
```

`capacity` must be a power of two, so `slot = seq & mask`. `pi`, `ci`, and `readySeq` are 64-bit absolute logical sequence numbers — do not reduce them to 32-bit (the `readySeq == seq + 1` scheme and ABA-freedom depend on absolute sequences).

The cross-domain ring carries only `pi`, `ci`, and per-slot `readySeq`. **All consumer claim/completion bookkeeping lives on the host side, never in the ring:**

```text
Single: a private uint64_t cursor (nextToRead), owned by the one consumer.
Multi:  a shared std::atomic head (FIFO claim counter) plus a host-side done[]
        array of absolute completion markers (one per slot), each padded to its
        own cache line (alignas(64)) so consumers completing nearby seqs do not
        false-share. Both are ordinary host atomics, since all consumers are CPU
        threads.
```

## Memory Model

The device producer uses **system-scope atomics** via `comms::device` (`comms/common/AtomicUtils.cuh`): `ld_relaxed_sys_global` / `ld_acquire_sys_global` for `pi`/`ci`, `atomic_fetch_add_relaxed_sys_global` / `atomic_cas_relaxed_sys_global` to reserve `pi`, and `st_release_sys_global` to publish `readySeq`. These lower to `ld/st.{relaxed,acquire,release}.sys.global` + `atom.relaxed.sys` PTX on NVIDIA and to `__atomic_*` / `__hip_atomic_*` on AMD. Using these lean primitives instead of `cuda::atomic_ref<…, thread_scope_system>` is also a large throughput win on H100/PCIe (see Performance).

The host consumer uses `std::atomic_ref<uint64_t>` release/acquire, which is **inner-shareable** on aarch64/Grace. That suffices — not a gap — because (a) the ring is coherent host-pinned mapped memory and (b) the device peer always issues the system-scope half. The host fence only has to order the host's own loads/stores. This "host std-atomic + device system-scope over coherent pinned memory" split is the GB200-validated convention used elsewhere in comms (cf. ctran `MemFence.h` / `GpeKernelSync`). A host-side `dmb sy` is intentionally NOT emitted. This asymmetry is exactly why the queue is D2H-only (see Non-Goals): in a hypothetical H2D direction the host would be the *publisher*, so it would have to carry the system-scope half itself with an explicit `dmb sy` — `std::atomic`'s inner-shareable release would silently fail to order the payload into the GPU domain (and x86 would mask the bug). `Direction::H2D` is therefore a compile error rather than a footgun.

**No separate publish fence.** The marker store is a system-scope *release* store, which by itself orders all prior writes in the producer thread (the payload memcpy) before the marker becomes visible to a consumer that acquire-loads it. An explicit `__threadfence_system()` before the release store would be redundant (it would be needed only if the marker store were *relaxed*). Dropping it removes one system-scope op from every publish — a ~13–20% publish-latency reduction (see Performance) — and the torn-publish stress test confirms correctness without it on H100, GB200/Grace, and AMD MI300.

| word | device access | host access |
| ---- | ------------- | ----------- |
| `pi` (RMW) | `atomic_fetch_add_relaxed_sys_global` / `atomic_cas_relaxed_sys_global` | n/a (producers are GPU) |
| `ci` (load on device, store/CAS on host) | `ld_acquire_sys_global` | `std::atomic_ref` release store (Single) / release CAS (Multi) |
| `readySeq` | `st_release_sys_global` (release) | `std::atomic_ref` acquire load |

Multi-consumer coordination (`head`, `done[]`) is host-only (CPU↔CPU) and uses ordinary `std::atomic`. The cross-domain 64-bit host atomics must be lock-free for the torn-publish protection to hold; this is asserted with `static_assert(... is_always_lock_free)` and is always true on the targeted host ABIs (x86-64, aarch64/Grace).

Ordering contract:

- **Publish/consume:** producer writes the payload, then release-stores `readySeq`; the consumer acquire-loads `readySeq` and, only on match, reads the payload. The release store — paired with the consumer's acquire — orders all payload writes before the marker becomes visible, for any payload size.
- **Credit:** the producer acquire-loads `ci`; the consumer release-advances it after copy-out. This guarantees the consumer has finished copying a slot's previous occupant before any producer overwrites it.

## Initialization

`cudaHostAlloc` returns **uninitialized** memory, so the region MUST be explicitly zeroed before any producer can run.

```text
1. require capacity is a power of two; mask = capacity - 1
2. bytes = sizeof(DeviceHostQueueControl<T>) + capacity * sizeof(DeviceHostQueueSlot<T>)
3. cudaHostAlloc(&host_ptr, bytes, <mapped[/coherent on HIP]>)
   cudaHostGetDevicePointer(&device_ptr, host_ptr, 0)  # free host_ptr if this fails
4. memset(host_ptr, 0, bytes)                           # cudaHostAlloc does NOT zero
5. publish the device handle / launch producer kernels only AFTER steps 1-4
```

Zeroing establishes `pi = ci = 0` and every slot's `readySeq = 0`. Why that is correct (the `readySeq == seq + 1` scheme):

```text
- pi starts its fetch_add at 0, so the first command is seq = 0.
- The producer publishes readySeq = seq + 1, so the first marker is 1.
- The consumer at the head (seq 0) treats a slot ready iff readySeq == seq + 1.
- A zero-initialized slot has readySeq == 0 != 1, so it reads as "not ready"
  until actually published. pi = ci = 0 also make the credit test
  (seq - ci < capacity) and the slot map (seq & mask) well-defined from step 0.
```

Physical slot `s` is reused by sequences `s, s+capacity, s+2*capacity, …`; the marker for occupant `seq` is `seq + 1`, which differs from the previous occupant's marker by `capacity` and from the zero sentinel by at least 1. With 64-bit absolute sequences the exact-match test is unambiguous (ABA-free).

## Sequence Protocol

`pi` is the producer reservation sequence. `ci` is the producer-visible reuse-credit prefix: a producer may overwrite a slot only once `ci` has advanced past it. Under copy-out semantics the consumer advances `ci` as part of `read()` — dequeuing a command frees its slot — so there is no separate `advance_ci` step.

Producer (both policies):

```text
# non-blocking write(): claim only if credit is available
seq = ld_relaxed(pi)
loop:
  if seq >= ci and seq - ci.load(acquire) >= capacity: return Full   # underflow-safe guard
  prev = atomic_cas_relaxed(pi, seq, seq + 1)        # claim; returns prior value
  if prev == seq: break                              # no hole on Full
  seq = prev                                         # refresh and retry

# blockingWrite(): unconditional reservation, then spin for credit
seq = atomic_fetch_add_relaxed(pi, 1)
while seq - ci.load(acquire) >= capacity: spin
publish(seq, value)

publish(seq, value):
  copy payload into slot[seq & mask]
  st_release_sys_global(readySeq[seq & mask], seq + 1)   # release store; no separate fence
```

The Full guard is `seq >= c && seq - c >= capacity`: `seq` (a relaxed snapshot of `pi`) can be stale because other producers advance `pi` and the consumer advances `ci`, so the explicit `seq >= c` check prevents an unsigned underflow that would spuriously report Full; when `seq < c` the CAS simply fails and refreshes `seq`. This path is contention-tested (256 threads racing for 8 slots — exactly `capacity` claim Ok, the rest Full, no holes).

Reserve-then-publish obligation: `blockingWrite` reserves with an unconditional `fetch_add` and is therefore committed to publishing. A thread that reserves and then never publishes (aborts mid-call) leaves a permanent hole that wedges the in-order consumer. `write()` has no such obligation — it claims `pi` only when it is about to publish.

Single consumer (private `nextToRead` cursor):

```text
read(out):
  seq = nextToRead
  if readySeq[seq & mask].load(acquire) != seq + 1: return Empty
  copy payload out
  nextToRead = seq + 1
  ci.store(seq + 1, release)        # copy-out frees the slot
  return Ok (seq)
```

Multi consumer (shared `head` + host `done[]`):

```text
read(out):
  loop:
    h = head.load(acquire)
    if readySeq[h & mask].load(acquire) != h + 1: return Empty   # head not published
    if not head.compare_exchange_weak(h, h + 1): continue          # lost the claim; retry
    # exclusive owner of seq h
    copy payload out
    done[h & mask].store(h + 1, release)     # mark consumed (absolute seq)
    advance_reuse_credit()
    return Ok (h)

advance_reuse_credit():               # advance ci over the contiguous consumed prefix
  loop:
    c = ci.load(relaxed)
    if done[c & mask].load(acquire) != c + 1: break   # end of contiguous freed prefix
    if ci.compare_exchange_weak(c, c + 1, release): continue
```

`advance_reuse_credit` is **opportunistic, not cooperative**: whoever wins the `ci` CAS walks the whole contiguous prefix; a consumer that loses just reloads, finds the winner has already drained to the last visible marker, and bails via the `break`. So one consumer drains `ci` at a time while the others fall through — this is **not a livelock** (each pass either advances `ci` or breaks, and `ci` is monotonic and bounded by the published markers). Reclaim is therefore eventually-consistent: `ci` may briefly lag a just-completed consume until a later advance observes the marker, which never corrupts and never deadlocks a producer (a producer blocks only on a full ring, i.e. commands are waiting, so consumers keep draining).

## Batched Dequeue (`readMulti`, MPSC)

`readMulti(out, max)` scans `readySeq` from `nextToRead` until the first unpublished slot or `max`, copies the N contiguous published commands into `out[0..N)`, then advances `ci` with a **single** release store for the whole batch (instead of one per command in `read()`). It returns N (0 if the head is unpublished) and is MPSC-only (`static_assert(Policy == Single)`).

Why it matters: the per-item `ci` release store is the cross-domain write that bounds drain throughput; amortizing it over a batch is the main consumer-side perf lever. On GB200 (where that store is the bottleneck) `readMulti` is ~1.7× faster than per-item `read()` at small payloads; on H100 it is neutral because the producer-side publish, not the consumer `ci` store, is the limiter there (see Performance). Per-slot `readySeq` acquire loads remain per item — those are cheap reads — only the credit store is batched.

`readMulti` is **not** provided for `Multi`: MPMC's bottleneck is the shared `head` CAS, not the `ci` store, and a batched multi-consumer claim is a separate, more complex design. Multi consumers should use per-item `read()` (see the inversion note).

## MPSC vs MPMC — the throughput inversion

**`Multi` is not "MPSC plus more throughput".** Measured on GB200/NVLink-C2C, a single `Single` consumer reaches ~50–80 Mops/s, while 2/4/8 `Multi` consumers each land at ~15–19 Mops/s aggregate — adding consumers *reduces* aggregate dequeue throughput. Once the cross-domain coherence is cheap (NVLink-C2C), the shared `head` CAS plus per-slot `done[]` markers become the limiter, and more consumers means more contention on those. On H100/PCIe the effect is hidden — both sit at ~16 Mops/s — because the PCIe round-trip dominates either way.

Guidance:

- **Dequeue-bound** (cheap per-command work, you just want commands off the GPU fast) → use **`Single` + `readMulti`**. Do not reach for `Multi` for throughput.
- **Processing-bound** (each command triggers real CPU work that one thread can't keep up with) → use **`Multi`** to parallelize the *processing*. The dequeue contention is then amortized by the per-command work, which is the regime `Multi` is designed for.

Picking `Multi` to "go faster" on a pure hand-off path will regress you.

## Blocking Variants And Spin Policy

`blockingWrite` (device) spins for reuse credit; `blockingRead` (host) spins until the head is published. Neither has an escape — they assume the peer makes progress.

`blockingRead` issues a `cpuRelax()` hint on each empty poll: the `YIELD` instruction on aarch64/Grace, `PAUSE` on x86. This relaxes the core (lets an SMT sibling progress, saves power) **without descheduling**, so an always-busy proxy consumer keeps its latency. It does *not* yield the core to the scheduler — backoff policy is the caller's, by design. A consumer that may idle, shares a core, or must also poll other things should loop on the non-blocking `read()` / `readMulti()` with its own backoff:

```cpp
ReadResult r;
while (q.read(r) == QueueOpStatus::Empty) {
  // caller's policy: cpuRelax(), std::this_thread::yield(), exponential backoff, ...
}
```

## Capacity Sizing

`capacity` (power of two) sets how many commands can be in flight before `write()` returns Full / `blockingWrite` spins. Size it from the producer/consumer rate gap and the consumer's drain batch:

- Steady state needs `capacity >= producer_burst` so producers don't stall on a transient consumer hiccup. As a starting point, pick `capacity` a few × the consumer's `readMulti` `max` batch.
- At ~16–80 Mops/s (see Performance) one slot frees every ~12–60 ns, so even a 1024-slot ring (≈12–60 µs of buffering at 64–128 B/slot) is small; start at 1024 and tune. Memory is `sizeof(Control) + capacity * sizeof(Slot)` of pinned mapped memory.
- Larger rings hide consumer jitter but cost pinned memory and can hurt cache locality of the `readySeq` scan; do not oversize blindly.

## Teardown And Lifetime

`DeviceHostQueue<T>` is a host RAII owner (non-copyable, non-movable) of the `cudaHostAlloc` allocation. `producerHandle()` returns a trivially-copyable, non-owning device handle passed by value into kernels; it holds only device-resolved pointers and frees nothing.

Destroying the queue while a device producer is still running is a use-after-free: a kernel may be spinning in `blockingWrite` (or mid `pi.fetch_add` / payload copy) holding device pointers into the ring, and `cudaFreeHost` on that memory while a kernel still touches it is undefined behavior.

Teardown protocol (caller-driven, since the queue has no abort word):

```text
1. signal producers to stop (caller's own abort flag) and stop issuing writes
2. quiesce every producer stream (cudaStreamSynchronize / cudaDeviceSynchronize)
3. ensure consumers have stopped calling read()
4. destroy the queue (frees control + slots)
```

The owner must outlive every kernel handed a producer handle.

## Portability (CUDA + HIP/ROCm)

The queue compiles and runs on both NVIDIA and AMD:

- **Device atomics** route through `comms::device` (`AtomicUtils.cuh`), which is `#if defined(__HIP_PLATFORM_AMD__)`-branched to `__atomic_*`/`__hip_atomic_*` on AMD and PTX on NVIDIA. There is no `<cuda/atomic>` / `cuda::atomic_ref` dependency.
- **Host allocation** uses the CUDA runtime API, which hipifies 1:1 (`cudaHostAlloc → hipHostMalloc`, etc.). On AMD the flags are `hipHostMallocMapped | hipHostMallocCoherent` — the **coherent (fine-grained)** flag is mandatory so a GPU system-scope release store is visible to a concurrently-polling CPU consumer mid-kernel with no manual cache flush; on a non-coherent allocation the consumer would only see writes at kernel completion.
- **BUCK**: the lib/tests are `comms_gpu_cpp_library` / `comms_gpu_cpp_unittest` with `hip_compatible = True`; the lib depends on `//comms/common:atomic_utils`.

Validated build + run on NVIDIA H100, AMD MI300, and (build) aarch64/Grace for GB200.

## Performance

Representative numbers, **median of 7 runs** per point (256 GPU producers; throughput is steady-state; latency is host-request → publish → host-read round trip). The GB200 fast path carries real run-to-run variance — the benchmark prints a stddev column per throughput point.

| (64 B) | H100 (x86 / PCIe) | GB200 (aarch64 / C2C) | MI300X (AMD / ROCm) |
| --- | --- | --- | --- |
| MPSC throughput | ~16 Mops/s | ~52 Mops/s | ~8 Mops/s |
| MPSC `readMulti` | ~17 Mops/s (neutral) | ~93 Mops/s (~1.8×) | ~9 Mops/s |
| MPMC (4 consumers) | ~13 Mops/s | ~15 Mops/s (< MPSC) | ~6 Mops/s |
| MPSC p50 latency | ~6.2 µs | ~3.9 µs | ~5.8 µs |
| Contention p50 (1/32/256 producers) | 6.0 / 7.0 / 18 µs | 3.9 / 3.9 / 6.4 µs | 5.4 / 5.6 / 8.4 µs |

Reading the data:

- **Ops-bound, not bandwidth-bound.** Mops/s is roughly flat across payload size while GB/s scales linearly — the limiter is the per-publish cross-domain coordination (system-scope release store + `pi` RMW), not the link. Even at 256 B on GB200 the NVLink-C2C link is only a few percent utilized.
- **`readMulti` helps where the consumer is the bottleneck.** On GB200 it is ~1.8× MPSC (the per-item `ci` store dominates there); on H100 it is neutral (the producer-side publish over PCIe is the limiter).
- **Capacity matters.** Throughput plateaus once the ring is a few hundred slots; a too-small ring starves it — 64-slot 64 B MPSC is ~2 Mops/s on H100 (~14 on GB200, ~3 on MI300) vs the plateau at cap ≥ 256. Start at ~1024 and tune (see Capacity Sizing).
- **MPMC is slower than MPSC for pure dequeue** on every architecture (head-CAS + `done[]` contention); the consumer-scaling sweep drops monotonically. Use `Multi` for processing parallelism, not dequeue throughput.
- **Latency degrades gracefully under producer contention** — p50 rises from the empty-queue path to the backlogged path as producers go 1 → 256 (e.g. GB200 3.9 → 6.4 µs); the queue does not fall over.
- **The `comms::device` PTX atomics matter on H100:** they took H100 throughput from ~1 → ~16 Mops/s versus libcu++ `cuda::atomic_ref<system>`. GB200 was already fast either way (cheap C2C coherence).
- **Dropping the publish fence** lowered p50 latency ~13–20% and was throughput-neutral.

## Tests

`comms/utils/device_host_queue/tests/DeviceHostQueueUT.cu`:

```text
capacity is power-of-two validated
empty read returns Empty (Single and Multi)
device-write / host-read round trip
producer Full/credit path (non-blocking write returns Full, then Ok after a read)
concurrent write() under contention (256 threads, cap 8): exactly `cap` Ok, every
  attempt Ok-or-Full, exactly `cap` drain back (CAS credit guard, no holes)
single-producer wraparound FIFO across many capacity cycles
readMulti batched drain: same sequence/values as per-item read(), no loss/dup
sizeApprox backlog
many GPU producers -> single host consumer: strict global sequence, per-producer
  FIFO, no loss/duplication, and a per-command guard-word torn-publish check
many GPU producers -> many concurrent host consumers (Multi): exactly-once
  delivery (every sequence and every (producer,value) once) plus torn-publish
```

The torn-publish guard fills the rest of the payload cache line with the command's value and verifies it on consume, so a publish observed without all payload writes visible (a weak-memory ordering bug) is detected. It is run on **H100 (NVIDIA), MI300 (AMD — relaxed memory), and aarch64/Grace** to exercise the actual weak-memory ordering of the GPU→CPU publish path and the `Multi` `done[]`/`ci` release-acquire chain — the strongest evidence that the release-store-only publish (no fence) is correct.

IBRC-specific tests (QP post order follows queue sequence order; credit advances only after the backend no longer needs the slot) belong with the IBRC backend, not this queue primitive.
```
