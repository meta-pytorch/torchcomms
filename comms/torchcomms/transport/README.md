# RdmaTransport: High-Performance RDMA Communication Library

The `RdmaTransport` module provides a high-level C++ API for RDMA (Remote Direct Memory Access) protocol communication over RoCE, enabling zero-copy, low-latency data transfers between CUDA devices across different hosts. Designed for high-performance distributed computing scenarios where low-latency, high-bandwidth communication between GPU memory regions is critical, RdmaTransport abstracts the complexity of low level RDMA operations while providing easy to use memory management and asynchronous operation support.


## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [API Reference](#api-reference)
- [Usage Guide](#usage-guide)
- [Integration Examples](#integration-examples)
- [Building and Testing](#building-and-testing)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)


### Key Components

- **`RdmaTransport`**: Main transport class for establishing connections and performing RDMA operations
- **`RdmaMemory`**: RAII wrapper for RDMA-registered memory buffers
- **`RdmaMemory::View`**: Lightweight view into registered memory regions
- **`RdmaRemoteBuffer`**: Structure representing remote memory access information

## Features

- **Zero-Copy Data Transfer**: Direct GPU-to-GPU memory transfers without CPU involvement
- **Asynchronous Operations**: Non-blocking RDMA writes with future-based completion handling
- **Memory Management**: Automatic registration/deregistration of CUDA memory
- **Event-Driven Architecture**: Integration with folly::EventBase for efficient I/O multiplexing, and ease of integration with other event-driven applications (e.g. Thrift)
- **Thread-Safe**: Safe for use across multiple threads
- **High Performance**: Optimized for low latency and high throughput scenarios

## Requirements

### Hardware
- RDMA capable network interface (e.g., Mellanox ConnectX series)
- CUDA-capable GPUs

### Platform Support
- Check support at runtime using `RdmaTransport::supported()`

```cpp
if (!torch::comms::RdmaTransport::supported()) {
    // Fallback to alternative transport mechanism
    return;
}
```

## API Reference

### RdmaTransport Class

#### Constructor
```cpp
RdmaTransport(int cudaDev, folly::EventBase* evb = nullptr);
```
- `cudaDev`: CUDA device ID to bind the transport to
- `evb`: Event base for asynchronous operations (optional)

#### Core Methods

**Connection Management**
```cpp
static bool supported();                     // Check platform support
std::string bind();                           // Bind and get local URL
commResult_t connect(const std::string& peerUrl);  // Connect to peer
bool connected() const;                       // Check connection status
```

**Data Transfer**
```cpp
folly::SemiFuture<commResult_t> write(
    RdmaMemory::View localBuffer,
    RdmaRemoteBuffer remoteBuffer,
    bool notify
);
folly::SemiFuture<commResult_t> waitForWrite();
```

### RdmaMemory Class

#### Constructor
```cpp
RdmaMemory(const void* buf, size_t len, int cudaDev);
```
- `buf`: Pointer to CUDA memory buffer
- `len`: Size of buffer in bytes (must be > 4097 bytes)
- `cudaDev`: CUDA device ID

#### Methods
```cpp
void* localKey() const;                       // Get local memory key
std::string remoteKey() const;                // Get remote access key
int getDevice() const;                        // Get CUDA device ID
bool contains(const void* buf, size_t len) const;  // Check if buffer is contained

// Create views into memory regions
View createView(size_t offset, size_t length) const;
View createView(const void* buf, size_t length) const;
```

#### Memory Views
```cpp
class RdmaMemory::View {
    const void* data() const;                 // Get view data pointer
    size_t size() const;                      // Get view size
    const RdmaMemory* operator->() const;     // Access parent memory
};
```

### RdmaRemoteBuffer Structure
```cpp
struct RdmaRemoteBuffer {
    void* ptr;                                // Remote memory pointer
    const std::string accessKey;             // Remote access key
};
```

## Usage Guide

### Basic Setup

#### 1. Initialize Transport Endpoints

**Server Side:**
```cpp
#include "comms/torchcomms/transport/RdmaTransport.h"
using namespace torch::comms;

// Create event base for async operations
auto evbThread = std::make_unique<folly::ScopedEventBaseThread>();
folly::EventBase* evb = evbThread->getEventBase();

// Create transport bound to CUDA device 0
auto serverTransport = std::make_unique<RdmaTransport>(0, evb);

// Bind and get server URL
std::string serverUrl = serverTransport->bind();
// Share serverUrl with client through your coordination mechanism

// Get the other end url and connect to it
auto result = clientTransport->connect(clientUrl);
if (result != commSuccess) {
    // Handle connection error
}
```

**Client Side:**
```cpp
// Create client transport on CUDA device 1
auto clientTransport = std::make_unique<RdmaTransport>(1, evb);

// Bind client and connect to server
std::string clientUrl = clientTransport->bind();
auto result = clientTransport->connect(serverUrl);
if (result != commSuccess) {
    // Handle connection error
}
```

#### 2. Memory Registration

```cpp
// Allocate CUDA memory
void* gpuBuffer = nullptr;
size_t bufferSize = 1024 * 1024; // 1MB
cudaMalloc(&gpuBuffer, bufferSize);

// Register memory for RDMA (requires > 4097 bytes)
RdmaMemory rdmaMemory(gpuBuffer, bufferSize, cudaDevice);

// Get remote access information
RdmaRemoteBuffer remoteInfo{
    .ptr = gpuBuffer,
    .accessKey = rdmaMemory.remoteKey()
};
// Share remoteInfo with peer
```

#### 3. Data Transfer Operations

**RDMA Write (Sender):**
```cpp
// Create view of data to send
auto dataView = rdmaMemory.createView(sendBuffer, dataSize);

// Perform asynchronous write
auto writeFuture = transport->write(
    dataView,
    peerRemoteBuffer,
    true  // notify receiver
);

// Wait for completion
auto result = std::move(writeFuture).get();
if (result == commSuccess) {
    // Transfer completed successfully
}
```

**Wait for Incoming Data (Receiver):**
```cpp
// Wait for notification from sender
auto waitFuture = transport->waitForWrite();
auto result = std::move(waitFuture).get();
if (result == commSuccess) {
    // Data has been received
}
```

#### 4. Batch Operations

```cpp
// Collect multiple write operations
std::vector<folly::SemiFuture<commResult_t>> writeFutures;

for (const auto& transfer : transfers) {
    auto view = rdmaMemory.createView(transfer.src, transfer.size);
    writeFutures.emplace_back(transport->write(view, transfer.dst, false));
}

// Wait for all operations to complete
auto results = folly::collectAll(std::move(writeFutures)).get();
```

## Integration Examples - Simple Ping-Pong Test

```cpp
// Complete example from test suite
void rdmaPingPong() {
    const size_t bufferSize = 8192;
    const int serverDev = 0, clientDev = 1;

    auto evbThread = std::make_unique<folly::ScopedEventBaseThread>();

    // Setup transports
    auto server = std::make_unique<RdmaTransport>(serverDev, evbThread->getEventBase());
    auto client = std::make_unique<RdmaTransport>(clientDev, evbThread->getEventBase());

    // Establish connection
    auto serverUrl = server->bind();
    auto clientUrl = client->bind();
    server->connect(clientUrl);
    client->connect(serverUrl);

    // Allocate and register memory
    void* serverBuf, *clientBuf;
    cudaSetDevice(serverDev);
    cudaMalloc(&serverBuf, bufferSize);
    cudaSetDevice(clientDev);
    cudaMalloc(&clientBuf, bufferSize);

    RdmaMemory serverMem(serverBuf, bufferSize, serverDev);
    RdmaMemory clientMem(clientBuf, bufferSize, clientDev);

    // Initialize data
    std::vector<uint8_t> testData(bufferSize, 0xAB);
    cudaMemcpy(serverBuf, testData.data(), bufferSize, cudaMemcpyHostToDevice);

    // Transfer data: server -> client
    RdmaRemoteBuffer clientRemote{clientBuf, clientMem.remoteKey()};
    auto writeFuture = server->write(
        serverMem.createView(serverBuf, bufferSize),
        clientRemote,
        true);

    // Client waits for data
    auto waitFuture = client->waitForWrite();

    // Verify completion
    assert(writeFuture.get() == commSuccess);
    assert(waitFuture.get() == commSuccess);

    // Cleanup
    cudaFree(serverBuf);
    cudaFree(clientBuf);
}
```

## Building and Testing

### Building

The module is built using Buck:

```bash
# Build the library
buck build //comms/torchcomms/transport:rdma_transport

# Build tests
buck build //comms/torchcomms/transport/tests:rdma_transport_test

# Build benchmarks
buck build //comms/torchcomms/transport/benchmarks:rdma_transport_bench
```

### Running Tests

```bash
# Run unit tests (requires 2 GPUs)
buck test //comms/torchcomms/transport/tests:rdma_transport_test

# Test platform support
buck test //comms/torchcomms/transport/tests:rdma_transport_support_ib_plat_test

# Test on CPU-only platforms
buck test //comms/torchcomms/transport/tests:rdma_transport_support_cpu_plat_test
```

### Running Benchmarks

```bash
# Run all benchmarks
buck run @//mode/opt //comms/torchcomms/transport/benchmarks:rdma_transport_bench

# Run specific benchmarks
buck run @//mode/opt //comms/torchcomms/transport/benchmarks:rdma_transport_bench -- --benchmark_filter="BM_RdmaTransport_Write"

# Generate JSON output
buck run @//mode/opt //comms/torchcomms/transport/benchmarks:rdma_transport_bench -- --benchmark_format=json --benchmark_out=results.json
```

## Performance

### Key Performance Characteristics

- **Latency**: Sub-microsecond for small transfers on modern InfiniBand
- **Bandwidth**: Close to Line Rate depending on message size
- **Memory Registration**: One-time cost, amortized over multiple transfers
- **CPU Overhead**: Minimal due to RDMA hardware offload

### Optimization Tips

1. **Memory Reuse**: Register large buffers once, use views for transfers
2. **Batch Operations**: Group multiple small transfers into batches
4. **Buffer Sizes**: Use buffers > 4097 bytes for registration requirements
5. **Event Base Sharing**: Share EventBase across multiple transports when possible

### Benchmark Results

Typical performance on H100 Server with 400Gbps ConnextX-7 NICs

---------------------------------------------------------------------------------------
Benchmark                                             Time             CPU   Iterations
---------------------------------------------------------------------------------------
BM_RdmaTransport_Write/8192/real_time              27.0 us         9.01 us        27612 bytes_per_second=288.978M/s
BM_RdmaTransport_Write/16384/real_time             27.4 us         6.21 us        25826 bytes_per_second=569.583M/s
BM_RdmaTransport_Write/32768/real_time             27.1 us         5.60 us        24296 bytes_per_second=1.1242G/s
BM_RdmaTransport_Write/65536/real_time             29.2 us         6.23 us        23751 bytes_per_second=2.09021G/s
BM_RdmaTransport_Write/131072/real_time            33.7 us         9.31 us        20269 bytes_per_second=3.62653G/s
BM_RdmaTransport_Write/262144/real_time            33.2 us         6.37 us        21033 bytes_per_second=7.35603G/s
BM_RdmaTransport_Write/524288/real_time            38.6 us         5.99 us        18880 bytes_per_second=12.6647G/s
BM_RdmaTransport_Write/1048576/real_time           49.2 us         7.16 us        14214 bytes_per_second=19.8673G/s
BM_RdmaTransport_Write/2097152/real_time           70.7 us         8.09 us         9882 bytes_per_second=27.6343G/s
BM_RdmaTransport_Write/4194304/real_time            113 us         10.1 us         5961 bytes_per_second=34.5553G/s
BM_RdmaTransport_Write/8388608/real_time            200 us         13.7 us         3477 bytes_per_second=39.1316G/s
BM_RdmaTransport_Write/16777216/real_time           371 us         18.3 us         1889 bytes_per_second=42.1151G/s
BM_RdmaTransport_Write/33554432/real_time           717 us         27.4 us          971 bytes_per_second=43.5943G/s
BM_RdmaTransport_Write/67108864/real_time          1409 us         46.7 us          496 bytes_per_second=44.3678G/s
BM_RdmaTransport_Write/134217728/real_time         2790 us         79.5 us          251 bytes_per_second=44.7995G/s
BM_RdmaTransport_Write/268435456/real_time         5552 us          142 us          126 bytes_per_second=45.0305G/s


## Troubleshooting

### Common Issues

#### "RdmaTransport is not supported"
- **Cause**: Missing InfiniBand hardware or drivers
- **Solution**: Verify IB setup with `ibstat` and `ibv_devices`
- **Fallback**: Use `RdmaTransport::supported()` to detect and fallback to alternative transport

#### "Failed to register memory"
- **Cause**: Buffer too small (< 4097 bytes) or memory permissions
- **Solution**: Ensure buffer size > 4097 bytes and proper CUDA context

#### Performance Issues
- **Cause**: Small transfers, frequent registration, or CPU bottlenecks
- **Solution**: Batch operations, reuse registered memory, use appropriate buffer sizes

### Debugging Tips

1. **Enable Logging**: Set `NCCL_DEBUG=INFO` for detailed logs
2. **Check Platform Support**: Always call `RdmaTransport::supported()` first
3. **Verify Network**: Use InfiniBand diagnostic tools (`ibping`, `ibv_rc_pingpong`)
4. **Monitor Completion**: Check return values from async operations
5. **Memory Validation**: Use `RdmaMemory::contains()` for bounds checking

---

For more examples and detailed implementation, see:
- Test suite: `comms/torchcomms/transport/tests/`
- Integration example: `msl/rl/tensor_transfer/`
- Benchmarks: `comms/torchcomms/transport/benchmarks/`
