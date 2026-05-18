// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Convenience header — HRDWRingBufferDeviceHandle is defined in
// HRDWRingBuffer.h so that deviceHandle() can return it by value.
// Include this header from .cu files for discoverability.
//
// USAGE
//   // Host: obtain handle from an existing ring buffer.
//   HRDWRingBuffer<MyEvent> buf(4096);
//   auto handle = buf.deviceHandle();
//
//   // Kernel: write events inline — no kernel launch overhead.
//   // Data is copied by value into the ring entry.
//   __global__ void myKernel(HRDWRingBufferDeviceHandle<MyEvent> rb, ...) {
//     rb.write(myEvent);              // atomicAdd + globaltimer + threadfence
//     // ... do work ...
//     rb.write(anotherEvent);
//   }
//
//   // Host: poll as usual.
//   HRDWRingBufferReader<MyEvent> reader(buf);
//   reader.poll([](const auto& entry, uint64_t slot) {
//     const MyEvent& evt = entry.data;
//   });

#pragma once // NOLINT(clang-diagnostic-pragma-once-outside-header)

#include "comms/utils/HRDWRingBuffer.h" // IWYU pragma: export
