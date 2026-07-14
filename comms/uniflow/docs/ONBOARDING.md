# Uniflow Developer Onboarding Guide

A minimal guide for engineers contributing to Uniflow or extending it for new platforms.

> **Uniflow** (Unified Transport for Heterogeneous LLM Systems) is a host-based point-to-point
> data transfer library for LLM workloads — disaggregated inference, RL tensor transfer,
> and checkpoint retrieval/loading.

---

## Table of Contents

- [Codebase Overview](#codebase-overview)
- [Building & Hello World](#building--hello-world)
- [Running Tests & Benchmarks](#running-tests--benchmarks)
- [Extending RDMA Backend for New HW Types](#extending-rdma-backend-for-new-hw-types)
- [Adding a New Backend](#adding-a-new-backend)
- [Further Reading](#further-reading)

---

## Codebase Overview

Source: `fbcode/comms/uniflow/`

```
comms/uniflow/
├── Uniflow.h/cpp          # UniflowAgent — top-level per-process orchestrator
├── Connection.h/cpp       # User-facing connection (control + data plane)
├── MultiTransport.h/cpp   # Aggregates backends, routes to optimal transport
├── Segment.h/cpp          # Memory abstraction (DRAM, VRAM, NVMe)
├── Result.h               # Error handling (Status, Result<T>)
├── controller/            # Control plane (TCP-based connection establishment)
├── executor/              # Async primitives (EventBase, ScopedEventBaseThread)
├── transport/
│   ├── Transport.h        # Abstract transport interface + TransportFactory
│   ├── Topology.h/cpp     # PCIe/NIC topology discovery (GPU↔NIC affinity)
│   ├── rdma/              # RDMA backend (InfiniBand / RoCE)
│   └── nvlink/            # NVLink backend (intra-node / MNNVL)
├── drivers/               # Hardware abstraction (cuda, ibverbs, nvml, sysfs)
├── core/                  # Low-level utilities (MPSC queue, Func)
├── tests/                 # Unit + integration tests
├── benchmarks/            # Performance benchmark suite
└── .claude/docs/          # Detailed design documentation per module
```

For detailed per-module design docs, see [`.claude/docs/`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/.claude/docs/):

| Document | Contents |
|----------|----------|
| [`overview.md`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/.claude/docs/overview.md) | System goals, architecture, key differentiators |
| [`transport.md`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/.claude/docs/transport.md) | Transport interface, factory lifecycle, backend types |
| [`rdma-transport.md`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/.claude/docs/rdma-transport.md) | RDMA put/get architecture, multi-NIC parallelism |
| [`rdma-copy-send-recv.md`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/.claude/docs/rdma-copy-send-recv.md) | Copy-based send/recv with slab pools |
| [`connection.md`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/.claude/docs/connection.md) | Connection design, segment exchange |
| [`segment.md`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/.claude/docs/segment.md) | Memory model (Segment, RegisteredSegment, Spans) |
| [`executor.md`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/.claude/docs/executor.md) | EventBase, async execution primitives |

Design doc: https://fburl.com/uniflow

---

## Building & Hello World

### Prerequisites

- devserver or on-demand with GPU access (≥2 GPUs for NVLink tests)
- Buck2 build system

### Build the Core Library

```bash
# Build the entire uniflow library
buck2 build //comms/uniflow:uniflow

# Build individual components
buck2 build //comms/uniflow:segment
buck2 build //comms/uniflow:multi-transport
buck2 build //comms/uniflow:connection
```

### Build & Run the Python Bindings (Hello World)

```bash
# Build Python extension
buck2 build //comms/uniflow:_core

# Quick sanity check — runs a basic Python test
buck2 test //comms/uniflow/tests/py:test_uniflow
```

### Hello World — Python (Requires 2 GPUs)

The integration test is the best "Hello World" — it exercises the full path:
agent creation → segment registration → connection → data transfer → verification.

```bash
buck2 test //comms/uniflow/tests/py:test_uniflow_integration
```

What it does (see [`tests/py/test_uniflow_integration.py`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/tests/py/test_uniflow_integration.py)):
1. Creates two `UniflowAgent` instances (one server, one client)
2. Registers GPU memory segments on each
3. Establishes a connection via TCP controller
4. Performs a `get()` transfer from GPU:0 → GPU:1 over NVLink
5. Verifies data correctness

### Hello World — C++ (Unit Tests)

```bash
# Run all unit tests
buck2 test //comms/uniflow/tests/unit:

# Run a specific test
buck2 test //comms/uniflow/tests/unit:segment_test
buck2 test //comms/uniflow/tests/unit:multi_transport_test
buck2 test //comms/uniflow/tests/unit:uniflow_agent_test
```

---

## Running Tests & Benchmarks

### Unit Tests (No Hardware Required)

```bash
buck2 test //comms/uniflow/tests/unit:                       # All core unit tests
buck2 test //comms/uniflow/executor/tests:                   # Executor tests
buck2 test //comms/uniflow/controller/tests:                 # Controller tests
buck2 test //comms/uniflow/core/tests:                       # Core utility tests
buck2 test //comms/uniflow/drivers/ibverbs/tests:            # IB verbs API tests
buck2 test //comms/uniflow/drivers/cuda/tests:               # CUDA driver tests
buck2 test //comms/uniflow/drivers/nvml/tests:               # NVML API tests
```

### Integration Tests (Require GPUs)

```bash
# Single-host NVLink test (requires ≥2 GPUs on one host)
buck2 test //comms/uniflow/tests/integration:multi_transport_single_host_test

# Cross-host RDMA test (requires 2 nodes, run via MAST or MPI)
buck2 test //comms/uniflow/tests/integration:multi_transport_cross_host_test
```

### NVLink Benchmarks

```bash
# Build the benchmark binary
buck2 build //comms/uniflow/benchmarks:uniflow_bench

# Run NVLink benchmark (local, 2 GPUs)
bash fbcode/comms/uniflow/benchmarks/scripts/run_nvlink_benchmark.sh

# Remote GB200 hosts (cross-compiled for aarch64)
bash fbcode/comms/uniflow/benchmarks/scripts/run_nvlink_benchmark.sh \
  --hosts <gb200_host0>,<gb200_host1> --gpu b200
```

Key options: `--iterations`, `--warmup`, `--min-size`, `--max-size`, `--direction put|get|both`

### RDMA Benchmarks

```bash
# End-to-end RDMA benchmark comparing uniflow vs ib_write_bw
bash fbcode/comms/uniflow/benchmarks/scripts/rdma_benchmark.sh \
  --host0 <host0> --host1 <host1>

# Options
#   --iterations 500  --warmup 10
#   --min-size 1      --max-size 1073741824
#   --chunks 524288,1048576,2097152,4194304
#   --num-nics 0 (0 = all available)
#   --skip-ib         (uniflow only)
#   --skip-uniflow    (ib_write_bw only)
```

### General Benchmark Launcher (torchrun)

```bash
# Local multi-rank using torchrun
bash fbcode/comms/uniflow/benchmarks/scripts/run_benchmark.sh \
  --nproc 2 -- --benchmark rdma_bandwidth --iterations 50
```

For benchmark design details, see [`benchmarks/DESIGN.md`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/benchmarks/DESIGN.md).

---

## Extending RDMA Backend for New HW Types

The RDMA backend supports new hardware through three extension points:
**topology discovery**, **NIC filtering**, and **device adaptation**.

### 1. Topology Discovery (`transport/Topology.h/.cpp`)

Topology discovery determines GPU↔NIC affinity by walking the PCIe tree via sysfs.

**How it works:**
- [`Topology.h`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/transport/Topology.h) defines the graph model: `TopoNode` (GPU/CPU/NIC), `TopoLink`, `PathType`
- [`Topology.cpp`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/transport/Topology.cpp) builds the graph via:
  1. CUDA device enumeration → GPU nodes with PCIe BDF
  2. IB device enumeration → NIC nodes with PCIe BDF
  3. Sysfs walk → PCIe ancestor chains for each device
  4. BFS shortest-path → PathType classification (PIX, PXB, PHB, SYS)
  5. PCIe link speed probing → bandwidth weights

**To support new hardware:**

| Step | Action | Where |
|------|--------|-------|
| 1 | Add a driver wrapper for your device's enumeration API (like `NvmlApi`, `IbvApi`) | `drivers/<your_hw>/` |
| 2 | Expose PCIe BDF discovery so topology can find your device in the sysfs tree | Your driver's `getDeviceSysfsPath()` or similar |
| 3 | Register nodes in the topology builder | `Topology.cpp` — add enumeration alongside GPU/NIC discovery |
| 4 | Update `NicFilter` if your NIC naming convention differs | `Topology.h` — `NicFilter` class |

**Key abstractions:**
- `SysfsApi` (`drivers/sysfs/`) — abstracts sysfs reads (mockable for tests)
- `IbvApi` (`drivers/ibverbs/`) — abstracts libibverbs (mockable)
- `NicFilter` — NCCL_IB_HCA-style NIC selection (prefix/exact include/exclude)
- `PathType` enum — ordered best→worst: NVL > C2C > PIX > PXB > PXN > PHB > SYS > DIS

### 2. Device Adapter (`drivers/DeviceAdapter.h`)

The [`DeviceAdapter`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/drivers/DeviceAdapter.h) interface abstracts host-pinned memory allocation for DMA:

```cpp
class DeviceAdapter {
 public:
  virtual Result<void*> pinnedHostAlloc(size_t size) = 0;
  virtual Status pinnedHostFree(void* ptr) = 0;
  virtual Result<void*> hostGetDevicePointer(void* hostPtr) = 0;
};
```

Platform selection is done via Buck `select()` in [`drivers/BUCK`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/drivers/BUCK):

```python
oss_cpp_library(
    name = "device-adapter",
    exported_deps = [":device-adapter-interface"] + select({
        "DEFAULT": ["//comms/uniflow/drivers/cuda:cuda-device-adapter"],
        "ovr_config//gpu:mtia": ["//comms/uniflow/fb:mtia-device-adapter"],
    }),
)
```

**To add support for your platform:**
1. Implement `DeviceAdapter` for your HW in `drivers/<your_hw>/`
2. Add a Buck `select()` entry for your platform config
3. Implement `createDeviceAdapter()` factory function

### 3. Testing a New HW Integration

1. **Unit tests** — Mock the driver APIs (see `drivers/ibverbs/mock/`, `drivers/cuda/mock/`)
2. **Topology test** — Validate GPU↔NIC path detection with your hardware's PCIe layout
3. **Integration test** — Use `multi_transport_single_host_test` pattern on your platform
4. **Benchmark** — Run `rdma_benchmark.sh` against your HW to validate bandwidth

---

## Adding a New Backend

To add a new transport backend (e.g., TCP, EFA, custom fabric):

### Step 1: Define Transport + Factory

Create `transport/<backend>/`:

```
transport/<backend>/
├── <Backend>Transport.h
├── <Backend>Transport.cpp
├── BUCK
└── CMakeLists.txt
```

Implement the two core interfaces from [`Transport.h`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/transport/Transport.h):

**Transport** (data plane — one instance per connection):
```cpp
class MyTransport : public Transport {
  const std::string& name() const noexcept override;
  TransportType transportType() const noexcept override;
  TransportState state() const noexcept override;
  TransportInfo bind() override;                              // Serialize local endpoint
  Status connect(std::span<const uint8_t> remoteInfo) override; // Connect to peer
  std::future<Status> put(std::span<const TransferRequest>, ...) override;
  std::future<Status> get(std::span<const TransferRequest>, ...) override;
  std::future<Status> send(RegisteredSegment::Span, ...) override;
  std::future<Status> recv(RegisteredSegment::Span, ...) override;
  std::future<Status> send(Segment::Span, ...) override;
  std::future<Status> recv(Segment::Span, ...) override;
  void shutdown() override;
};
```

**TransportFactory** (lifecycle management — one per process):
```cpp
class MyTransportFactory : public TransportFactory {
  Result<std::unique_ptr<RegistrationHandle>> registerSegment(Segment&) override;
  Result<std::unique_ptr<RemoteRegistrationHandle>> importSegment(...) override;
  Result<std::unique_ptr<Transport>> createTransport(std::span<const uint8_t> peerTopology) override;
  std::vector<uint8_t> getTopology() override;
  Status canConnect(std::span<const uint8_t> peerTopology) override;
};
```

### Step 2: Register in TransportType Enum

Add your backend to [`transport/TransportType.h`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/transport/TransportType.h).

### Step 3: Integrate into MultiTransportFactory

In [`MultiTransport.cpp`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/MultiTransport.cpp):
1. Add a `supported()` check in `MultiTransportFactory::supported()`
2. Instantiate your factory in the `MultiTransportFactory` constructor
3. Add topology serialization in `getTopology()` / `parse()`

### Step 4: Add Tests

Follow the existing patterns:
- **Unit test**: Mock-based test in `transport/<backend>/tests/`
- **Integration**: Add a case to `tests/integration/MultiTransportSingleHostTest.cpp`
- **Benchmark**: Add a `<Backend>BandwidthBenchmark` in `benchmarks/bench/`

### Step 5: Reference Implementations

Study these as templates:
- **NVLink** (simpler, GPU-only): [`transport/nvlink/`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/transport/nvlink/)
- **RDMA** (full-featured, multi-NIC): [`transport/rdma/`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/transport/rdma/)

---

## Further Reading

| Resource | Link |
|----------|------|
| Design Doc | https://fburl.com/uniflow |
| RFC | [UniFlow RFC (Google Doc)](https://docs.google.com/document/d/1UkX7OgV4xtekjGnoe0VJAj4-5jXDIp47kzIUhm7hyMo) |
| Coding Standards | [`.claude/CLAUDE.md`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/.claude/CLAUDE.md) |
| Benchmark Design | [`benchmarks/DESIGN.md`](https://www.internalfb.com/code/fbsource/fbcode/comms/uniflow/benchmarks/DESIGN.md) |
| Oncall | `ncclx` |

### Key Design Principles

- **No folly dependency** — C++20 standard library only (OSS portability)
- **Lock-free data path** — All mutable state accessed on EventBase thread
- **Mockable drivers** — Each HW driver has a mock in `drivers/<hw>/mock/`
- **Topology-aware routing** — MultiTransport picks optimal backend automatically
