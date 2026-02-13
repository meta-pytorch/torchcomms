---
name: pipes-audit-plan
description: Audit and evaluate implementation plans for completeness, correctness, and quality. Identifies dead code, complexity issues, gaps, and optimization opportunities. Use when asked to review a plan or when the user invokes /pipes-audit-plan with a path to a plan file.
---

# Pipes Plan Audit

Carefully and thoroughly audit the specified implementation plan pertaining to the **Pipes library** -- a high-performance communication primitives library for writing custom collectives on NVIDIA GPUs.

## Audit Objectives

Your audit should identify and evaluate:

### Code Quality Issues
- **Dead or redundant code**: Unused functions, duplicate logic, obsolete implementations
- **Scattered logic**: Functionality spread across too many files or layers
- **Overly complex logic**: Areas that can be simplified while maintaining functionality
- **Unnecessary abstractions**: Over-engineering that adds complexity without benefit

### Completeness Issues
- **Gaps in implementation**: Missing functionality required by the plan's goals
- **Incomplete refactors**: Partially migrated code, lingering old patterns
- **Loose ends**: TODOs, FIXMEs, placeholder implementations, unfinished work
- **Missing error handling**: Uncovered failure modes or edge cases

### Sources of Confusion
- **Inconsistent patterns**: Mixed approaches to similar problems
- **Poor naming**: Unclear variable, function, or class names
- **Missing documentation**: Undocumented public APIs or complex algorithms
- **Ambiguous ownership**: Unclear responsibilities between components

---

## Arguments

The user should provide one or more paths to files containing plans developed by Claude Code.

---

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

### 1. Architectural Soundness
- Does the plan follow Pipes' dual-layer design philosophy?
- Are responsibilities clearly separated between host and device layers?
- Does the design maintain composability with existing primitives?
- Is the host-device object mapping clean and 1:1?

### 2. Completeness
- Does the plan fully address the intended feature or change?
- Are all necessary components identified?
  - Headers: `.cuh` for device code, `.h` for host-only
  - Implementation: `.cc` for host, `.cu` for device kernels
  - Tests: `{Component}Test.{cc,cu,cuh}` triplet pattern
  - BUCK targets using `comms_gpu_cpp_library` for GPU code
- Are edge cases and failure modes addressed?
- Is error handling comprehensive for both device (`__trap()`) and host (`std::runtime_error`)?

### 3. Correctness
- Is the proposed logic sound and free of obvious bugs?
- **Memory Ordering**: Are acquire/release semantics correctly planned?
  - Signal operations: release semantics (all prior writes visible before signal)
  - Wait operations: acquire semantics (subsequent reads see peer's writes)
- **ThreadGroup Usage**:
  - Are `group.sync()` calls planned appropriately?
  - Are leader-only operations properly guarded?
- **Transport Semantics**: Are SELF vs P2P_NVL transport types handled correctly?

### 4. Simplicity & Clarity
- Can any proposed logic be simplified while maintaining functionality?
- Are there opportunities to reuse existing primitives instead of creating new ones?
- Does the plan avoid over-engineering?
- Are naming conventions followed?
  - Classes: `PascalCase` (e.g., `DeviceBarrier`, `ThreadGroup`)
  - Functions: `snake_case` (e.g., `make_warp_group()`, `signal_peer()`)
  - Factory functions: `make_` prefix
  - Member variables: `name_` suffix
  - Constants: `kCamelCase`

### 5. Maintainability
- Will the resulting code be easy to understand and modify?
- Are abstractions at the right level?
- Is the plan consistent with existing patterns in the codebase?
- Does it avoid introducing technical debt?

### 6. Performance Considerations
- Are hot paths identified and optimized appropriately?
- Is `__forceinline__` planned for hot-path device functions?
- Are vectorized copies (`uint4`/`memcpy_vectorized`) used where appropriate?
- Is 128-byte alignment used for synchronization primitives?
- Are buffer sizes and pipeline depths appropriate?

### 7. Test Strategy
- Does the plan include adequate test coverage?
- Are tests following the triplet pattern (`.cc`, `.cu`, `.cuh`)?
- Are multi-GPU scenarios addressed with proper skip logic?
- Are edge cases and failure modes covered?

### 8. Documentation Plan
- Are public APIs planned with Javadoc-style documentation?
- Are memory ordering semantics documented for synchronization primitives?
- Are complex algorithms explained?

---

## Common Pitfalls to Flag

### Architectural
- [ ] Breaking the host-device 1:1 object mapping pattern
- [ ] Mixing host and device responsibilities inappropriately
- [ ] Creating non-composable primitives
- [ ] Over-abstracting simple functionality

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

### Completeness
- [ ] Missing error handling paths
- [ ] Incomplete migration from old patterns
- [ ] Orphaned TODOs or FIXMEs without clear resolution plan
- [ ] Missing test coverage for new functionality

---

## Output Format

Provide your audit in the following structure:

### 1. Executive Summary
A brief (2-3 sentence) assessment of the plan's overall quality and readiness.

### 2. Audit Findings Table

| Category | Rating | Key Issues |
|----------|--------|------------|
| Architectural Soundness | ✅ Pass / ⚠️ Minor Issues / ❌ Needs Attention | Brief description |
| Completeness | ... | ... |
| Correctness | ... | ... |
| Simplicity & Clarity | ... | ... |
| Maintainability | ... | ... |
| Performance | ... | ... |
| Test Strategy | ... | ... |
| Documentation | ... | ... |

### 3. Detailed Findings

For each category with issues, provide:
- **Issue**: Clear description of the problem
- **Location**: Where in the plan this occurs
- **Impact**: Why this matters
- **Recommendation**: How to address it

### 4. Action Plan

Provide a prioritized, step-by-step action plan:

**Critical (Must Fix)**
1. [Issue] - [Specific action to take]
2. ...

**Important (Should Fix)**
1. [Issue] - [Specific action to take]
2. ...

**Recommended (Nice to Have)**
1. [Issue] - [Specific action to take]
2. ...

### 5. Optimization Opportunities

List specific opportunities to:
- Simplify complex logic
- Remove dead or redundant code
- Improve performance
- Enhance reusability

### 6. Questions for Clarification

List any ambiguities or questions that need answers before implementation can proceed.

---

## Iterative Enhancement

After presenting the initial audit, be prepared to:

1. **Incorporate user feedback** to refine the action plan
2. **Provide deeper analysis** on specific areas of concern
3. **Suggest alternative approaches** when issues are identified
4. **Update recommendations** as new information is provided

The goal is to ensure a robust and reliable final implementation that is:
- **Flexible**: Adapts to changing requirements
- **Extensible**: Easy to add new functionality
- **Maintainable**: Simple to understand and modify
- **Precise**: Crafted with attention to detail

---

## Additional Context
- **Oncall**: ncclx team
- **Target platforms**: H100, GB200 (up to 72 ranks for NVLink)
- **Source code**: `fbcode/comms/pipes/`
