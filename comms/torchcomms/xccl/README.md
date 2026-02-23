
# XCCL Backend in TorchComms

The **XCCL backend** in **TorchComms** enables high‑performance distributed computing on **Intel® GPUs** for both **scale‑up** (single-node, multi‑GPU) and **scale‑out** (multi‑node) environments. In PyTorch, the device type used for Intel GPUs is **`xpu`**.

## Overview

XCCL provides a communication backend that integrates tightly with the PyTorch Distributed stack through TorchComms. It offers efficient implementations of collective operations such as:

- `all_reduce`
- `broadcast`
- `all_gather`
- `reduce_scatter`
- `reduce`
- `barrier`
- and more

These primitives enable scalable data-parallel and model-parallel workloads on Intel GPU clusters.

The backend is optimized for Intel GPU architectures and supports a wide range of distributed configurations, including:

- **Intra-node GPU communication** using high-bandwidth GPU–GPU links
- **Inter-node communication** over high-performance network fabrics
- **Hybrid communication paths** that combine GPU, host, and network transport layers

## Powered by oneCCL

The XCCL backend is built on top of the [**Intel® oneAPI Collective Communications Library (oneCCL) C API**](https://uxlfoundation.github.io/oneCCL/v2/index.html#) library designed for HPC and deep learning workloads.

oneCCL provides:

- Highly optimized collective communication primitives
- Asynchronous and stream‑ordered execution
- Transport support for GPU‑direct, PCIe, and high‑speed network fabrics
- Flexible runtime tuning for performance, scalability, and reproducibility

TorchComms leverages oneCCL through its C API bindings to deliver native-level performance and tight integration with Intel GPU execution pipelines.

## Key Features

- **Intel GPU Support**

  Native integration with PyTorch’s `xpu` device type.

- **Scalable Performance**

  Optimized for multi‑GPU, multi‑node clusters.

- **Full Collective Operation Coverage**

  Implements all major collectives required by PyTorch Distributed.

- **Asynchronous Execution**

  Supports non-blocking operations and PyTorch stream compatibility.

- **Interoperability**

  To provide seamless compatibility with PyTorch Distributed APIs (`torch.distributed`, `c10d`) via TorchComms.
