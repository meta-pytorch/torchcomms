# IBVerbX Library

IBVerbX is a C++ wrapper library for InfiniBand Verbs that provides both dynamic loading and direct linking options.

## Build Options

### Default Build (Dynamic Loading)
```bash
buck build //comms/ctran/ibverbx:ibverbx
```

This builds the library with dynamic loading of InfiniBand verbs functions using `dlopen`/`dlsym`. This is the default behavior and allows the library to work without requiring InfiniBand libraries to be linked at compile time.

### Direct Linking Build
```bash
buck build //comms/ctran/ibverbx:ibverbx-rdma-core
```

This builds the library with the `-DIBVERBX_BUILD_RDMA_CORE` compiler flag, which:
- Includes `<infiniband/verbs.h>` directly
- Bypasses the `dlopen` path in `buildIbvSymbols()`
- Links directly against the `rdma-core` library
- Provides better performance by avoiding function pointer indirection

## Compiler Flag: `-DIBVERBX_BUILD_RDMA_CORE`

When the `-DIBVERBX_BUILD_RDMA_CORE` flag is set:

1. **Header files** (`Ibverbx.h` and `Ibvcore.h`):
   - Include `<infiniband/verbs.h>` directly

2. **Implementation** (`Ibverbx.cc`):
   - The `buildIbvSymbols()` function assigns function pointers directly to the real InfiniBand functions
   - Skips the dynamic loading code path entirely

3. **Build configuration** (`BUCK`):
   - Links against the `rdma-core` external dependency
   - Adds the compiler flag to enable conditional compilation

## Usage

Both build variants provide the same API and can be used interchangeably. Choose the appropriate variant based on your deployment requirements:

- Use the default build when you need runtime flexibility and don't want to require InfiniBand libraries at link time
- Use the linked build when you want maximum performance and know InfiniBand libraries will be available

## Implementation Details

The library always uses `ibverbx::` struct definitions that are identical to the real InfiniBand types, regardless of build configuration. All types are defined in the `ibverbx` namespace (e.g., `ibverbx::ibv_device`, `ibverbx::ibv_context`, `ibverbx::ibv_qp`, etc.) to avoid namespace conflicts with system InfiniBand headers.

The conditional compilation (`#ifdef IBVERBX_BUILD_RDMA_CORE`) only affects:
- Whether to include `<infiniband/verbs.h>` directly or use dynamic loading
- Function pointer assignment in `buildIbvSymbols()` (direct assignment vs dlsym)
- Build dependencies and linking against rdma-core libraries

This design ensures type compatibility across all build variants while maintaining clear namespace separation and avoiding conflicts with system InfiniBand installations.

## IbvVirtualQp: Virtual Queue Pair Abstraction in Ibverbx
Ibverbx introduces new abstractions to efficiently partition large messages and load balance across multiple data queue pairs in RDMA applications.

### Key abstractions
- IbvVirtualQp:
A virtual queue pair that internally manages multiple data queue pairs (QPs). This enables partitioning a large message into sub-messages, which can be sent across multiple data QPs for higher throughput and better load balancing.
- IbvVirtualCq:
A virtual completion queue designed to work with IbvVirtualQp, managing work request completions from all underlying QPs.

### Load Balancing Modes
`IbvVirtualQp` supports two load balancing modes:

- In DQPLB mode, all data is sent using IBV_WR_RDMA_WRITE_WITH_IMM on normal data QPs. Each message includes a sequence number in the immediate data, helping the receiver track message order and completeness. One bit in the immediate data is reserved to indicate if a WQE should trigger a higher-layer notification.

- In Spray mode, all data will be sent using IBV_WR_RDMA_WRITE WQEs, and after all writes complete, a single zero-byte IBV_WR_RDMA_WRITE_WITH_IMM is posted to notify the remote side of completion.

### Usage Overview

In both load balancing modes, the usage of `IbvVirtualQp` is very similar to that of a standard IbvQp. The typical workflow is as follows:

1. Prepare work requests. Sender prepares send work requests (ibv_send_wr). Receiver prepares receive work requests (ibv_recv_wr).
2. Post work requests. Sender calls IbvVirtualQp::postSend, and receiver calls IbvVirtualQp::postRecv.
3. Poll for completion. Both sender and receiver call IbvVirtualCq::pollCq to track completions.

```
// Receiver side
ibv_recv_wr recvWr = ...; // prepare receive work request
ibv_recv_wr recvWrBad;
IbvVirtualQp::postRecv(&recvWr, &recvWrBad);

// Sender side
ibv_send_wr sendWr = ...; // prepare send work request
ibv_send_wr sendWrBad;
IbvVirtualQp::postSend(&sendWr, &sendWrBad);

// Poll sender virtual CQ in a loop until one virtual CQE is polled
while (!stop) {
   auto maybeSendWcsVector = sendVirtualCq.pollCq(1);
   if (maybeSendWcsVector->size() == 1) {
      // add code to process the polled CQE
      break;
   }
}

// Poll receiver virtual CQ in a loop until one virtual CQE is polled
while (!stop) {
   auto maybeRecvWcsVector = recvVirtualCq.pollCq(1);
   if (maybeRecvWcsVector->size() == 1) {
      // add code to process the polled CQE
      break;
   }
}
```
