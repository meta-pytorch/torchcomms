# CTRAN Adaptive Data Transfer Mechanism

This document describes CTRAN's adaptive data transfer mechanism, including copy-based and zero-copy detection, auto-switch logic, and background memory registration workflow for both NVL and IB backends.

## Overview

CTRAN implements an adaptive data transfer mechanism that automatically switches between **copy-based** and **zero-copy** transfers based on buffer registration status. The core workflow is:
1. **Check RegCache**: Determine if the buffer is registered in the registration cache
2. **If Registered**: Run zero-copy transfer using the handle from cache
3. **If Not Registered**: Request async thread to register memory, then run copy-based transfer

```mermaid
flowchart LR
    START([Buffer Access]) --> CHECK{RegCache Lookup}
    CHECK -->|Hit| ZC[Zero-Copy Path]
    CHECK -->|Miss| REG[Async Registration]
    REG --> CB[Copy-Based Path]
    REG -.->|Background| CACHE[Update RegCache]
    CACHE -.->|Next Access| CHECK

    style CHECK fill:#ff9,stroke:#333,stroke-width:2px
    style ZC fill:#9f9,stroke:#333,stroke-width:2px
    style CB fill:#f99,stroke:#333,stroke-width:2px
```

### Benefits

- **Easier Adoption**: Users can seamlessly adopt CTRAN collectives and achieve best performance regardless of whether memory is pre-registered or not
- **Better NVL Zero-Copy Performance**: The new design eliminates CPU-kernel synchronization overhead on the fast path, closing performance gaps with baseline kernels (see [Ctran Zero-Copy Kernel Performance Gap and New Design](https://docs.google.com/document/d/1DSsM5mvzv1esQ09Yd5t6uf0jaeXx4wMGGapYLo8hyd4/edit?tab=t.0#heading=h.g0rbvjc2q7zx))
- **Optimal Performance for Repeated Operations**: Once buffers are registered, subsequent operations use zero-copy path
- **Graceful Fallback**: First-time or unregistered buffers safely use copy-based transfer

---

## NVL Backend

### Key Concepts

- **Export Handle**: Local GPU memory handle that can be shared with remote peers
- **Import Handle**: Remote peer's exported handle that has been imported locally
- **RemoteImportRegCache**: Caches imported handles from remote peers
  - **Key**: (receive buffer address, peer rank)
  - **Value**: Peer-imported buffer address
- **AsyncSocketThread**: Background thread that handles handle export/import via socket

### NVL Workflow Diagram

```mermaid
flowchart TD
    subgraph Receiver["Receiver Side"]
        R_START([SendRecv Initiated]) --> R_CHECK{Check RemoteImportRegCache: Is recv buffer imported by peer?}
        R_CHECK -->|Yes| R_KERNEL_ZC[Launch Kernel: Send ctrl + imported handle via NVL to sender]
        R_CHECK -->|No| R_ASYNC[Send task to AsyncSocketThread]
        R_ASYNC --> R_KERNEL_CB[Launch Kernel: Send ctrl 'Use Copy-Based' via NVL to sender]
        R_KERNEL_ZC --> R_ZC_RECV[Zero-Copy Receive]
        R_KERNEL_CB --> R_CB_RECV[Copy-Based Receive]
    end

    subgraph RecvAsync["Receiver AsyncSocketThread (Background)"]
        A_EXPORT[Export recv buffer handle] --> A_SEND[Send exported handle via socket]
        A_RECV[Receive imported handle] --> A_CACHE[Cache in RemoteImportRegCache]
    end

    subgraph SendAsync["Sender AsyncSocketThread (Background)"]
        SA_RECV[Receive exported handle] --> SA_IMPORT[Import recv buffer handle]
        SA_IMPORT --> SA_SEND[Send imported handle via socket]
    end

    subgraph Sender["Sender Side"]
        S_START([SendRecv Initiated]) --> S_KERNEL[Launch Kernel: Wait for ctrl msg from receiver]
        S_KERNEL --> S_CHECK{Ctrl Message from Receiver?}
        S_CHECK -->|Zero-Copy + Handle| S_ZC_SEND[Zero-Copy Send using received handle]
        S_CHECK -->|Copy-Based| S_CB_SEND[Copy-Based Send]
    end

    R_ASYNC --> A_EXPORT
    A_SEND --> SA_RECV
    SA_SEND --> A_RECV
    R_KERNEL_ZC -.->|NVL: ctrl + imported handle| S_KERNEL
    R_KERNEL_CB -.->|NVL: ctrl only| S_KERNEL

    style R_CHECK fill:#f9f,stroke:#333,stroke-width:2px
    style S_CHECK fill:#f9f,stroke:#333,stroke-width:2px
    style A_CACHE fill:#9f9,stroke:#333,stroke-width:2px
```

### NVL Transfer Sequence

```mermaid
sequenceDiagram
    participant SendAsync as Sender AsyncSocketThread
    participant Sender
    participant Receiver
    participant RecvAsync as Receiver AsyncSocketThread
    participant RegCache as RemoteImportRegCache


    Note over Receiver,Sender: First Transfer (Not Registered)

    Receiver->>RegCache: Check if buffer imported by peer
    RegCache-->>Receiver: Not found
    Receiver->>RecvAsync: Request handle registration (background)

    Note over Receiver: Launch Kernel
    Receiver->>Sender: Ctrl msg: "Use Copy-Based" (via NVL inside kernel)
    Sender->>Receiver: Copy-Based Kernel Transfer

    Note over RecvAsync,SendAsync: Background Handle Exchange
    RecvAsync->>RecvAsync: Export local recv buffer handle
    RecvAsync->>SendAsync: Send exported handle (via socket)
    SendAsync->>SendAsync: Import recv buffer handle
    SendAsync->>RecvAsync: Send imported handle (via socket)
    RecvAsync->>RegCache: Cache imported handle

    Note over Receiver,Sender: Subsequent Transfer (Registered)

    Receiver->>RegCache: Check if buffer imported by peer
    RegCache-->>Receiver: Found - return imported handle

    Note over Receiver: Launch Kernel
    Receiver->>Sender: Ctrl msg: "Use Zero-Copy" + imported handle (via NVL inside kernel)
    Sender->>Receiver: Zero-Copy Kernel Transfer
```

---

## IB Backend

### Key Concepts

- **Memory Registration (MR)**: Register GPU/host memory with IB hardware for RDMA operations
- **RegCache**: Caches local MR (memory region) handles for send/recv buffers
- **AsyncThread**: Background thread that registers memory with IB
- **rkey**: Remote key that allows remote peer to access registered memory via RDMA
- **Staging Buffer**: Pre-registered intermediate buffer used when user buffer is not registered

### IB Workflow Diagram

The sender and receiver sides operate **independently** - the only coordination is the receiver sending either:
- **recv buffer rkey** (for zero-copy receive)
- **staging buffer rkey** (for copy-based receive)

The sender always performs zero-copy RDMA write to the given rkey, but may need to copy to its own staging buffer first if the send buffer is not registered.

```mermaid
flowchart TD
    subgraph Receiver["Receiver Side"]
        R_START([SendRecv Initiated]) --> R_CHECK{Check RegCache: Is recv buffer registered to IB?}
        R_CHECK -->|Yes| R_GPE_ZC[Start Zero-Copy Recv in GPE]
        R_CHECK -->|No| R_ASYNC[Request AsyncThread to register recv buffer]
        R_ASYNC --> R_GPE_CB[Start Copy-Based Recv in GPE]
        R_GPE_ZC --> R_CTRL_ZC[Send ctrl msg via IB: recv buffer rkey]
        R_GPE_CB --> R_CTRL_CB[Send ctrl msg via IB: staging buffer rkey]
        R_CTRL_ZC --> R_WAIT_ZC[Wait for RDMA write to recv buffer]
        R_CTRL_CB --> R_CB_FLOW[Wait for RDMA write to staging buffer & copy to recv buffer chunk by chunk]
    end

    subgraph RecvAsync["Receiver AsyncThread (Background)"]
        R_ASYNC -.-> R_REG[Register recv buffer to IB & cache in RegCache]
    end

    subgraph Sender["Sender Side"]
        S_START([SendRecv Initiated]) --> S_CHECK{Check RegCache: Is send buffer registered to IB?}
        S_CHECK -->|Yes| S_GPE_ZC[Start Zero-Copy Send in GPE]
        S_CHECK -->|No| S_ASYNC[Request AsyncThread to register send buffer]
        S_ASYNC --> S_GPE_CB[Start Copy-Based Send in GPE]

        %% Zero-Copy Send Path
        S_GPE_ZC --> S_WAIT_ZC[Wait for ctrl msg with remote rkey]
        S_WAIT_ZC --> S_RKEY_CHECK_ZC{Received rkey type?}
        S_RKEY_CHECK_ZC -->|User recv buffer rkey| S_RDMA_ZC_RECV[RDMA write send buffer to remote recv buffer]
        S_RKEY_CHECK_ZC -->|Staging buffer rkey| S_RDMA_ZC_STAGING[Split: RDMA write send buffer to remote staging buffer chunk by chunk]

        %% Copy-Based Send Path
        S_GPE_CB --> S_WAIT_CB[Wait for ctrl msg with remote rkey]
        S_WAIT_CB --> S_RKEY_CHECK_CB{Received rkey type?}
        S_RKEY_CHECK_CB -->|User recv buffer rkey| S_CB_RECV[Split: Copy to local staging & RDMA write to remote recv buffer chunk by chunk]
        S_RKEY_CHECK_CB -->|Staging buffer rkey| S_CB_STAGING[Split: Copy to local staging & RDMA write to remote staging buffer chunk by chunk]
    end

    subgraph SendAsync["Sender AsyncThread (Background)"]
        S_ASYNC -.-> S_REG[Register send buffer to IB & cache in RegCache]
    end

    R_CTRL_ZC -.->|IB ctrl channel| S_WAIT_ZC
    R_CTRL_ZC -.->|IB ctrl channel| S_WAIT_CB
    R_CTRL_CB -.->|IB ctrl channel| S_WAIT_ZC
    R_CTRL_CB -.->|IB ctrl channel| S_WAIT_CB

    style R_CHECK fill:#f9f,stroke:#333,stroke-width:2px
    style S_CHECK fill:#f9f,stroke:#333,stroke-width:2px
    style S_RKEY_CHECK_ZC fill:#f9f,stroke:#333,stroke-width:2px
    style S_RKEY_CHECK_CB fill:#f9f,stroke:#333,stroke-width:2px
    style R_REG fill:#9f9,stroke:#333,stroke-width:2px
    style S_REG fill:#9f9,stroke:#333,stroke-width:2px
```

> **Layout Note**: Receiver is on the left, Sender is on the right.

### IB Transfer Sequence

```mermaid
sequenceDiagram
    participant SenderCache as Sender RegCache
    participant SenderAsync as Sender AsyncThread
    participant Sender
    participant Receiver
    participant RecvAsync as Receiver AsyncThread
    participant RecvCache as Receiver RegCache

    Note over Sender,Receiver: First Transfer (Not Registered on Both Sides)

    Sender->>SenderCache: Check if send buffer registered
    SenderCache-->>Sender: Not found
    Sender->>SenderAsync: Request registration (background)

    Receiver->>RecvCache: Check if recv buffer registered
    RecvCache-->>Receiver: Not found
    Receiver->>RecvAsync: Request registration (background)

    Note over Receiver: Start Copy-Based Recv in GPE
    Note over Sender: Start Copy-Based Send in GPE

    Receiver->>Sender: Ctrl msg: staging buffer rkey (via IB)

    Sender->>Sender: Copy send buffer to staging buffer
    Sender->>Receiver: RDMA write staging buffer to remote staging buffer
    Receiver->>Receiver: Copy staging buffer to recv buffer

    Note over SenderAsync,RecvAsync: Background Registration
    SenderAsync->>SenderAsync: Register send buffer to IB
    SenderAsync->>SenderCache: Cache MR handle
    RecvAsync->>RecvAsync: Register recv buffer to IB
    RecvAsync->>RecvCache: Cache MR handle

    Note over Sender,Receiver: Subsequent Transfer (Registered on Both Sides)

    Sender->>SenderCache: Check if send buffer registered
    SenderCache-->>Sender: Found - return MR handle

    Receiver->>RecvCache: Check if recv buffer registered
    RecvCache-->>Receiver: Found - return MR handle

    Note over Receiver: Start Zero-Copy Recv in GPE
    Note over Sender: Start Zero-Copy Send in GPE

    Receiver->>Sender: Ctrl msg: recv buffer rkey (via IB)

    Sender->>Receiver: RDMA write send buffer directly to recv buffer
```

---

## Key Implementation Notes

1. **Thread Safety**: RegCache must be thread-safe as it's accessed by both data path and async registration threads
2. **Memory Lifetime**: Registered memory must remain valid while handles are cached
3. **Cache Invalidation**: Handles must be invalidated when memory is freed or remapped
4. **Ctrl Message Ordering**: For NVL, ctrl message is sent inside kernel; for IB, GPE starts first then sends ctrl message
5. **IB Independence**: Sender and receiver sides make independent decisions about copy-based vs zero-copy based on their own buffer registration status
