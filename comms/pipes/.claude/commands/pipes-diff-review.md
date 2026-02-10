---
name: pipes-diff-review
description: Review Phabricator diffs related to the Pipes library (comms/pipes). Use when asked to review a diff touching Pipes code, or when the user invokes /pipes-diff-review with a diff number. Pipes is a high-performance GPU communication primitives library.
---

# Pipes Library Diff Review

Review the specified Phabricator diff(s) for the **Pipes library** - a high-performance communication primitives library for writing custom collectives on NVIDIA GPUs.

## Arguments

The user should provide one or more diff numbers (e.g., `D12345678`) as arguments. If no diff number is provided, ask the user which diff to review.

## Library Context

Pipes provides low-level, device-side abstractions for NVLink and RDMA communication with a focus on the following guiding principles:
- **Zero-cost abstractions**: Closing the gap between prototype and optimized implementations
- **Composability**: Building complex collectives from reusable primitives
- **Dual-layer design**: High-level device primitives + high-level host APIs
- **Fault tolerance**: Error handling at both device and host levels

Within the Pipes library, there are two distinct layers:
1. **High-Level Device Layer** - Async NVLink/RDMA APIs, ThreadGroup parallelism, ChunkIterator patterns
2. **High-Level Host Layer** - RAII resource management, fusion capabilities, Python bindings

Key components include: `ThreadGroup`, `P2pNvlTransportDevice`, `ChunkState`, `SignalState`, `Transport`, `Timeout`, and collective implementations.

---

## Evaluation Criteria

### 1. Clarity
- Is the code easy to read and understand?
- Are variable and function names descriptive and follow Pipes conventions?
  - Classes: `PascalCase` (e.g., `DeviceBarrier`, `ThreadGroup`)
  - Functions: `snake_case` (e.g., `make_warp_group()`, `signal_peer()`)
  - Factory functions: `make_` prefix
  - Member variables: `name_` suffix (e.g., `myRank_`, `nRanks_`)
  - Constants: `kCamelCase` (e.g., `kWarpSize`)
- Is the logic straightforward? Are complex algorithms well-commented?

### 2. Completeness
- Does the diff fully address the intended change or feature?
- Are all necessary files updated following the Pipes file organization?
  - Headers: `.cuh` for device code, `.h` for host-only
  - Implementation: `.cc` for host, `.cu` for device kernels
  - Tests: `{Component}Test.{cc,cu,cuh}` triplet pattern
  - Benchmarks: `{Component}Bench.{cc,cu,cuh}` triplet pattern
- Are BUCK targets properly defined using `comms_gpu_cpp_library` for GPU code?

### 3. Stylistic Correctness
- Does the code follow Pipes coding conventions?
- Is Javadoc-style documentation used (`/** ... */` with `@param`, `@return`)?
- Are `__forceinline__` annotations used for hot-path device functions?
- Are proper `#ifdef __CUDA_ARCH__` guards used for host/device code?
- Is 128-byte alignment used for synchronization primitives?
- Are CUDA printf format specifiers correct? (e.g., `%llu` for `uint64_t`, not `%lu`)

### 4. Functional Correctness
- Does the code implement the intended functionality correctly?
- Are there any obvious bugs or logical errors?
- **Memory Ordering**: Are acquire/release semantics correctly applied?
  - Signal operations: release semantics (all prior writes visible before signal)
  - Wait operations: acquire semantics (subsequent reads see peer's writes)
  - Uses `st_release_sys_global` and `ld_acquire_sys_global` for NVLink coherence
- **ThreadGroup Usage**:
  - Is `group.sync()` called appropriately after collective operations?
  - Are leader-only operations guarded with `group.is_leader()`?
  - Do all threads in a group call collective operations together?
- **Error Handling**:
  - Device code: Uses `__trap()` for fatal errors
  - Host code: Throws `std::runtime_error` with descriptive messages

### 5. NVLink/Transport Semantics
- Is the design consistent with Pipes transport patterns?
- **Host-Device Object Mapping**: Is there a clean 1:1 mapping between host RAII objects and device objects?
- **Transport Dispatch**: Are SELF vs P2P_NVL transport types handled correctly?
- **Buffer Management**: Are data/state/signal buffers properly sized and aligned?
- **Pipelining**: Are pipeline depth and chunk sizes configured appropriately?
- **Inbox Model**: For signals/barriers, is the inbox model correctly implemented (one inbox per rank, peers write to specific slots)?

### 6. Tech Debt
- Does this diff introduce any technical debt?
- Are there shortcuts, hacks, or areas that may require future refactoring?
- Are TODOs properly tracked with diff numbers (e.g., `TODO(D12345678)`)?
- Is the code avoiding over-engineering while remaining extensible?

### 7. Documentation
- Is there sufficient documentation for new or modified code?
- Are public APIs documented with Javadoc-style comments?
- Are memory ordering semantics documented for synchronization primitives?
- Are usage examples provided for complex APIs?
- Is the architecture/design clearly explained for new components?

### 8. Unit Test Coverage
- Are there appropriate unit tests for new or changed functionality?
- Do tests follow the triplet pattern (`.cc`, `.cu`, `.cuh`)?
- Do tests cover edge cases and failure modes?
- **Multi-GPU tests**: Do tests properly skip when insufficient GPUs are available?

For any unit tests introduced or modified, please evaluate whether the following criteria are met:

- **Simplicity**: Avoid overly complicated or redundant tests. Focus on robust coverage of relevant scenarios and use-cases.
- **API Compliance**: Do not introduce or mandate behaviors that are not already documented or defined within the Pipes APIs.
- **Infrastructure Reuse**: Reuse as much of the existing test infrastructure and helper methods as possible (e.g., existing transport setup utilities, test fixtures).
- **Clear Assertions**: Use clear assertions to validate expected outcomes for both success and failure paths.
- **Documentation**: Provide comments explaining the purpose and expected outcome of each test, especially for complex async or multi-peer scenarios.
- **Necessity**: Only generate additional unit tests if they are needed. Only modify existing unit tests if it is necessary.
- **No Sleep-Based Orchestration**: Avoid using sleeps to orchestrate tests. Prefer mocking, `EXPECT_CALL`, synchronization primitives, or Pipes' own signaling/barrier mechanisms.
- **Device/Host Coverage**: For components with both device and host aspects, ensure both are tested.
- **Multi-Peer Scenarios**: For transport or collective tests, include scenarios with multiple peers where applicable.

### 9. Benchmark Coverage (if applicable)
- Are benchmarks provided for performance-critical code paths?
- Do benchmarks follow the `{Component}Bench.{cc,cu,cuh}` pattern?
- Are warmup iterations and configurable iteration counts supported?
- Is latency/throughput reporting clear and accurate?

---

## Common Pitfalls to Check

### Device Code
- [ ] Missing `__forceinline__` on hot-path device functions
- [ ] Missing `#ifdef __CUDA_ARCH__` guards for host/device code
- [ ] Missing `group.sync()` after collective operations
- [ ] Incorrect memory alignment (should be 128-byte for sync primitives)
- [ ] Wrong CUDA printf format specifiers

### ThreadGroup Usage
- [ ] Multiple threads calling leader-only operations
- [ ] Not all threads calling collective operations
- [ ] Incorrect partitioning (`num_partitions` > `total_groups`)

### Memory Access
- [ ] Capturing `DeviceSpan` in lambdas (extract raw pointers first)
- [ ] Non-vectorized copies where `uint4`/`memcpy_vectorized` should be used
- [ ] Non-contiguous work assignment hurting cache locality

### Transport/Signaling
- [ ] Missing memory barriers before signaling
- [ ] Incorrect signal slot indexing
- [ ] Self-transport send/recv (traps - use `put()` instead)
- [ ] Buffer overflow from incorrect size calculations

---

## Output Format

Please provide:

1. **Summary table** with ratings (pass, minor issues, needs attention) for each criterion
2. **Detailed findings** for each criterion, highlighting:
   - Strengths
   - Areas requiring attention
   - Specific line numbers for issues
3. **Recommendations** with prioritized action items
4. **Overall assessment** of the diff's readiness for landing

---

## Additional Context
- **Oncall**: ncclx team
- **Target platforms**: H100, GB200 (up to 72 ranks for NVLink)
- **Source code**: `fbcode/comms/pipes/`
