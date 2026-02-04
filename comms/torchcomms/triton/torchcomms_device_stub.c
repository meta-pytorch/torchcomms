// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Triton Device API - Stub file for LLVM bitcode generation
//
// This file is compiled with clang to produce LLVM bitcode (.bc) that Triton
// can link at compile time. The bitcode contains function stubs that are
// resolved at runtime by linking with the nvcc-compiled implementation.
//
// Compile with:
//   clang -c -emit-llvm -O1 --target=nvptx64-nvidia-cuda -o
//   libtorchcomms_device.bc torchcomms_device_stub.c

// Use void* for opaque handle types (no CUDA header dependencies)
typedef void* TorchCommsWindowHandle;
typedef void* TorchCommsBufferHandle;

// RMA Operations

int torchcomms_put(
    TorchCommsWindowHandle win,
    unsigned long long dst_offset,
    TorchCommsBufferHandle src_buf,
    unsigned long long src_offset,
    int dst_rank,
    unsigned long long bytes,
    int signal_id,
    int counter_id) {
  return 0;
}

// Signal Operations

int torchcomms_signal(
    TorchCommsWindowHandle win,
    int peer,
    int signal_id,
    unsigned long long value) {
  return 0;
}

int torchcomms_wait_signal(
    TorchCommsWindowHandle win,
    int signal_id,
    unsigned long long expected_value) {
  return 0;
}

unsigned long long torchcomms_read_signal(
    TorchCommsWindowHandle win,
    int signal_id) {
  return 0;
}

void torchcomms_reset_signal(TorchCommsWindowHandle win, int signal_id) {}

// Counter Operations

int torchcomms_wait_local(
    TorchCommsWindowHandle win,
    int counter_id,
    unsigned long long expected_value) {
  return 0;
}

unsigned long long torchcomms_read_counter(
    TorchCommsWindowHandle win,
    int counter_id) {
  return 0;
}

void torchcomms_reset_counter(TorchCommsWindowHandle win, int counter_id) {}

// Synchronization & Completion

int torchcomms_fence(TorchCommsWindowHandle win) {
  return 0;
}

int torchcomms_flush(TorchCommsWindowHandle win) {
  return 0;
}

// Window Properties

int torchcomms_rank(TorchCommsWindowHandle win) {
  return 0;
}

int torchcomms_num_ranks(TorchCommsWindowHandle win) {
  return 0;
}

void* torchcomms_base(TorchCommsWindowHandle win) {
  return (void*)0;
}

unsigned long long torchcomms_size(TorchCommsWindowHandle win) {
  return 0;
}
