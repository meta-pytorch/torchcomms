// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/tests/Checks.h"
#include "comms/pipes/tests/P2pIbgdaTransportDeviceTest.cuh"

namespace comms::pipes::tests {

// =============================================================================
// Device-side test kernels
// =============================================================================

__global__ void testP2pTransportConstruction(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    bool* success) {
  // Create transport on device with the given buffers (no real QP)
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf);

  *success = true;

  // Verify buffer accessors work
  auto localSig = transport.getLocalSignalBuffer();
  auto remoteSig = transport.getRemoteSignalBuffer();

  if (localSig.ptr != localBuf.ptr || localSig.lkey != localBuf.lkey) {
    *success = false;
  }
  if (remoteSig.ptr != remoteBuf.ptr || remoteSig.rkey != remoteBuf.rkey) {
    *success = false;
  }

  // QP should be null in this test (no real DOCA setup)
  if (transport.getQp() != nullptr) {
    *success = false;
  }
}

__global__ void testP2pTransportDefaultConstruction(bool* success) {
  // Default construction should initialize all members
  P2pIbgdaTransportDevice transport;

  *success = true;

  // QP should be null
  if (transport.getQp() != nullptr) {
    *success = false;
  }

  // Local signal buffer should have null ptr
  auto localSig = transport.getLocalSignalBuffer();
  if (localSig.ptr != nullptr) {
    *success = false;
  }

  // Remote signal buffer should have null ptr
  auto remoteSig = transport.getRemoteSignalBuffer();
  if (remoteSig.ptr != nullptr) {
    *success = false;
  }

  // Default numSignals should be 1
  if (transport.getNumSignals() != 1) {
    *success = false;
  }
}

__global__ void testP2pTransportNumSignals(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, numSignals);

  *success = (transport.getNumSignals() == numSignals);
}

__global__ void testP2pTransportSignalPointerArithmetic(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, numSignals);

  *success = true;

  // The transport stores base pointers and calculates signal[i] as base + i
  // We can verify this by checking the buffer accessors still point to base
  auto localSig = transport.getLocalSignalBuffer();
  auto remoteSig = transport.getRemoteSignalBuffer();

  // Base pointers should match what we passed in
  if (localSig.ptr != localBuf.ptr) {
    *success = false;
  }
  if (remoteSig.ptr != remoteBuf.ptr) {
    *success = false;
  }

  // Keys should be preserved
  if (localSig.lkey != localBuf.lkey) {
    *success = false;
  }
  if (remoteSig.rkey != remoteBuf.rkey) {
    *success = false;
  }
}

__global__ void testP2pTransportReadSignal(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* success) {
  // The localBuf should point to d_signalBuf which is pre-initialized with
  // known values: d_signalBuf[i] = (i + 1) * 100
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, numSignals);

  *success = true;

  // Test read_signal for each slot
  for (int i = 0; i < numSignals; ++i) {
    uint64_t expected = static_cast<uint64_t>(i + 1) * 100;
    uint64_t actual = transport.read_signal(i);
    if (actual != expected) {
      *success = false;
    }
  }
}

__global__ void testIbgdaWork(bool* success) {
  *success = true;

  // Test default construction
  IbgdaWork defaultWork;
  if (defaultWork.value != 0) {
    *success = false;
  }

  // Test explicit construction with a value
  doca_gpu_dev_verbs_ticket_t testTicket = 12345;
  IbgdaWork workWithValue(testTicket);
  if (workWithValue.value != testTicket) {
    *success = false;
  }

  // Test copy
  IbgdaWork copiedWork = workWithValue;
  if (copiedWork.value != testTicket) {
    *success = false;
  }
}

// =============================================================================
// wait_signal test kernels
// =============================================================================

__global__ void testWaitSignalEQ(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t targetValue,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, 1);

  // Signal buffer is pre-set to targetValue by host
  // wait_signal with EQ should return immediately
  transport.wait_signal(0, IbgdaCmpOp::EQ, targetValue);

  // If we get here, the wait completed successfully
  *success = true;
}

__global__ void testWaitSignalNE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, 1);

  // Signal buffer is pre-set to signalValue (which != targetValue)
  // wait_signal with NE should return immediately
  transport.wait_signal(0, IbgdaCmpOp::NE, targetValue);

  // If we get here, the wait completed successfully
  *success = true;
}

__global__ void testWaitSignalGE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, 1);

  // Signal buffer is pre-set to signalValue (which >= targetValue)
  // wait_signal with GE should return immediately
  transport.wait_signal(0, IbgdaCmpOp::GE, targetValue);

  // If we get here, the wait completed successfully
  *success = true;
}

__global__ void testWaitSignalGT(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, 1);

  // Signal buffer is pre-set to signalValue (which > targetValue)
  // wait_signal with GT should return immediately
  transport.wait_signal(0, IbgdaCmpOp::GT, targetValue);

  // If we get here, the wait completed successfully
  *success = true;
}

__global__ void testWaitSignalLE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, 1);

  // Signal buffer is pre-set to signalValue (which <= targetValue)
  // wait_signal with LE should return immediately
  transport.wait_signal(0, IbgdaCmpOp::LE, targetValue);

  // If we get here, the wait completed successfully
  *success = true;
}

__global__ void testWaitSignalLT(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, 1);

  // Signal buffer is pre-set to signalValue (which < targetValue)
  // wait_signal with LT should return immediately
  transport.wait_signal(0, IbgdaCmpOp::LT, targetValue);

  // If we get here, the wait completed successfully
  *success = true;
}

__global__ void testWaitSignalMultipleSlots(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* success) {
  P2pIbgdaTransportDevice transport(nullptr, localBuf, remoteBuf, numSignals);

  *success = true;

  // Signal buffer is pre-set: slot[i] = (i + 1) * 100
  // Test wait_signal on each slot with matching EQ condition
  for (int i = 0; i < numSignals; ++i) {
    uint64_t expectedValue = static_cast<uint64_t>(i + 1) * 100;
    transport.wait_signal(i, IbgdaCmpOp::EQ, expectedValue);

    // Verify read_signal returns the same value
    uint64_t readValue = transport.read_signal(i);
    if (readValue != expectedValue) {
      *success = false;
    }
  }
}

// =============================================================================
// Wrapper functions to launch the kernels (called from .cc test file)
// =============================================================================

void runTestP2pTransportConstruction(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    bool* d_success) {
  testP2pTransportConstruction<<<1, 1>>>(localBuf, remoteBuf, d_success);
}

void runTestP2pTransportDefaultConstruction(bool* d_success) {
  testP2pTransportDefaultConstruction<<<1, 1>>>(d_success);
}

void runTestP2pTransportNumSignals(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* d_success) {
  testP2pTransportNumSignals<<<1, 1>>>(
      localBuf, remoteBuf, numSignals, d_success);
}

void runTestP2pTransportSignalPointerArithmetic(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* d_success) {
  testP2pTransportSignalPointerArithmetic<<<1, 1>>>(
      localBuf, remoteBuf, numSignals, d_success);
}

void runTestP2pTransportReadSignal(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* d_success) {
  testP2pTransportReadSignal<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, numSignals, d_success);
}

void runTestIbgdaWork(bool* d_success) {
  testIbgdaWork<<<1, 1>>>(d_success);
}

void runTestWaitSignalEQ(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalEQ<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, targetValue, d_success);
}

void runTestWaitSignalNE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalNE<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
}

void runTestWaitSignalGE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalGE<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
}

void runTestWaitSignalGT(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalGT<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
}

void runTestWaitSignalLE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalLE<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
}

void runTestWaitSignalLT(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success) {
  testWaitSignalLT<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
}

void runTestWaitSignalMultipleSlots(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* d_success) {
  testWaitSignalMultipleSlots<<<1, 1>>>(
      d_signalBuf, localBuf, remoteBuf, numSignals, d_success);
}

} // namespace comms::pipes::tests
