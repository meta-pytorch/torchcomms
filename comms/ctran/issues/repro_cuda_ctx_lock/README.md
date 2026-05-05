# CUDA primary context lock — reproducers

Two minimal, self-contained reproducers for behavior of the per-context
mutex inside `libcuda.so`:

| Binary | Demonstrates | Bounded? |
|--------|--------------|----------|
| `memcpyasync_blocked` | A `cudaMemcpyAsync` call can block on the host for the entire time another thread holds the primary context lock — even when the two calls target separate non-blocking streams. | Yes (~12s on H100) |
| `memcpyasync_deadlock` | The same lock-holding pattern becomes a true deadlock when the kernel occupying the GPU is *persistent* and only exits on a `cudaMemcpyAsync` that itself is blocked on the lock. | No (watchdog fires at 30s) |

Both rely on the same root cause: with `CUDA_MODULE_LOADING=LAZY` (the
default in CUDA 11.7+), the very first `cudaLaunchKernel` for a kernel
whose module hasn't been loaded yet calls `cuLibraryLoadData` *under the
primary context lock*, and `cuLibraryLoadData` has to wait for the GPU to
become available before it can register the module. Any other thread that
needs the same lock blocks for the duration.

## Build & run

```
make
make run-blocked          # demo 1 LAZY  (host blocks ~12s)
make run-blocked-eager    # demo 1 EAGER (microseconds — control)
make run-deadlock         # demo 2 LAZY  (deadlocks; watchdog fires at 30s)
make run-deadlock-eager   # demo 2 EAGER (completes in <1s — control)
```

Override the GPU architecture with `make CUDA_ARCH=sm_90`. Default is
`sm_80`. Tested on CUDA 12.9 / driver 580.82.07 / H100. Should work on
any CUDA 12.x toolkit and any sm_70+ device.

---

## Demo 1: `memcpyasync_blocked`

```
main thread:                            child thread:
  busyKernel<<<...,sA>>>(20e9 ticks)      sleep 50 ms
  sleep 50 ms                             cudaMemcpyAsync(...,sB)   // timed
  spawn child
  firstTimeKernel<<<...,sA>>>()  // timed
```

`busyKernel` and `firstTimeKernel` live in **separate translation units**
so each gets its own lazily-loaded CUDA module. The order of operations:

1. Main launches `busyKernel`. With LAZY loading this triggers
   `cuLibraryLoadData` for `busyKernel`'s module — the GPU is idle, so
   the load is fast, and the kernel starts spinning on the GPU.
2. Main spawns a child (which sleeps 50 ms before doing anything) and
   immediately calls `firstTimeKernel<<<>>>`. This is the first launch of
   `firstTimeKernel`, so the runtime calls `cuLibraryLoadData` for *its*
   module — but the GPU is now occupied by `busyKernel`, and the driver
   has to wait for the GPU to become available. **`cuLibraryLoadData`
   blocks while still holding the primary context lock.**
3. After its 50 ms sleep, the child issues `cudaMemcpyAsync` on a
   *different* non-blocking stream. The call has to acquire the same
   primary context lock to enqueue the copy. It blocks on the host until
   the main thread's `cudaLaunchKernel` returns.

`busyKernel` exits on its own after a fixed number of clock ticks, so
the deadlock here is bounded — eventually both calls return.

### Expected output

```
$ make run-blocked
[B] cudaMemcpyAsync returned in   12.478s
[A] cudaLaunchKernel returned in 12.529s

$ make run-blocked-eager
[A] cudaLaunchKernel returned in 0.000s
[B] cudaMemcpyAsync returned in   0.000s
```

The 51 ms gap between B and A in LAZY mode matches the child's
pre-memcpy sleep — confirming both calls were serialized on the same
lock. Same binary in EAGER mode: both API calls return in microseconds
because no `cuLibraryLoadData` happens under the lock.

---

## Demo 2: `memcpyasync_deadlock`

Replaces `busyKernel` with `persistentKernel(volatile int* shutdown)`,
which spins on the GPU until `*shutdown != 0`. The only thing that flips
that flag is the child thread's `cudaMemcpyAsync` — which is the very
call that gets blocked on the lock.

```
persistentKernel  ─── waits for shutdown=1 ──────────────┐
        ▲                                                │
        │ holds GPU                                      │
        │                                                │
cuLibraryLoadData ─── waits for GPU ────────             │
        │                                       │        │
        │ holds context lock                    │        │
        ▼                                       │        │
cudaMemcpyAsync(shutdown=1) ─── waits for lock ─┘        │
        │                                                │
        └──── shutdown=1 never reaches GPU ──────────────┘
                              (cycle complete)
```

The process wedges forever. A watchdog thread declares the deadlock and
calls `_Exit(0)` after 30 seconds — the only safe way out of a wedged
CUDA process.

### Expected output

```
$ make run-deadlock      # LAZY -- true deadlock
DEADLOCK DETECTED: process wedged for 30s. Forcing exit.
```

Note the absence of `[A]` and `[B]` lines: both threads stayed inside
their respective CUDA calls for the full 30 seconds. The `firstTimeKernel`
was never even enqueued onto the stream, because the runtime hadn't
finished `cuLibraryLoadData` yet.

```
$ make run-deadlock-eager  # EAGER -- no deadlock, completes <1s
[A] cudaLaunchKernel returned in 0.000s
[B] cudaMemcpyAsync (shutdown=1) returned in 0.000s
Process completed without deadlock.
```

Same binary, same kernels, same logic — only `CUDA_MODULE_LOADING`
changes. With EAGER, `firstTimeKernel`'s module is loaded at process
init, so the first-launch path doesn't enter `cuLibraryLoadData`, the
lock isn't held, the `cudaMemcpyAsync` proceeds, the GPU receives
`shutdown=1`, the persistent kernel exits, the queued `firstTimeKernel`
runs, the process completes.

---

## Why this matters

These reproducers isolate the CUDA driver behavior behind a deadlock
observed in production at Meta. A persistent compression-job worker was
occupying the GPU; the GPE thread's `cudaMemcpyAsync` (sending work to
that worker via a command queue) blocked on the primary context lock;
the lock holder was the main thread inside a first-time
`cudaLaunchKernel` (FBGEMM TBE forward) waiting on `cuLibraryLoadData`;
`cuLibraryLoadData` was waiting for the worker to free the GPU; the
worker was waiting for the command-queue update that the blocked
`cudaMemcpyAsync` would have delivered. Three-way deadlock.

`memcpyasync_blocked` shows the load-bearing precondition (an "async"
CUDA API call really does block on the host when another thread holds
the per-context lock). `memcpyasync_deadlock` shows that the same
mechanism becomes terminal when the GPU work depends on the blocked
call.

---

## Notes on the construction

* Two kernels in **separate `.cu` files** is necessary in both demos. If
  both kernels live in the same translation unit they share a single
  CUDA module, the first launch lazy-loads both at once (while the GPU
  is idle), and the second launch never enters the slow
  `cuLibraryLoadData` path.
* The 50 ms pre-launch sleep on the main thread ensures the
  busy/persistent kernel is actually running on the GPU before we try to
  load the second module.
* The 50 ms pre-memcpy sleep on the child thread ensures the main
  thread has had time to enter the locked driver call before the child
  issues its `cudaMemcpyAsync`. Spawning a thread + entering the runtime
  takes microseconds in practice, so 50 ms is a comfortable margin.
* `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)` is used for
  both streams to rule out implicit synchronization through the legacy
  default stream.
* The deadlock demo uses an unbuffered stdout (`setvbuf(_IOLBF)`) so
  prints become visible even when the watchdog force-exits the process.
