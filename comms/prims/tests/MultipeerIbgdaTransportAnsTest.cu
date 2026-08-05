// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// ANS (variable-size CopyOp) send/recv correctness kernels for the IBGDA
// transport test. Built with `--device-c` + `-DPIPES_ENABLE_ANS_COMPRESSION`
// and device-linked against the nvcompdx fatbin (see
// `:multipeer_ibgda_transport_ans_test_kernels` in BUCK) so
// `AnsCompress<...>::send/recv` resolve. Lives in its own translation unit +
// device-link target (mirroring `:ans_copy_op_bench`) so the plain
// MultipeerIbgdaTransportTest.cu kernels stay nvcompdx-free.
//
// This exercises the variable-size compressed send/recv branch added in
// D111967119 end-to-end over two ranks (rank 0 compresses+sends, rank 1
// recvs+decompresses), and the caller verifies the payload round-trips.

#include "comms/prims/tests/MultipeerIbgdaTransportTest.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims::test {

namespace {

// Send and recv are SEPARATE kernels (rather than one `if (send)` kernel) so
// each instantiates only ONE direction's nvcompdx `__shared__` scratch, and
// each carries `__launch_bounds__(NumWarps*32, 1)` to cap per-thread registers.
// Together these keep the launch within the SM resource budget: a single
// 512-thread kernel holding both the compress and decompress AnsCompress
// instantiations (and uncapped registers) overflows it with
// cudaErrorLaunchOutOfResources. NumWarps is the block's cooperative warp count
// (blockDim.x / 32). The trap-safe signaled chunk size is derived from the
// per-block staging slot so both ranks agree on the chunking without exchange
// (a compressed sub-chunk's worst case is ~1.3x its uncompressed input, so the
// max_signal_bytes==0 / chunkSize==perBlockSlot default would always trap).
template <int NumWarps>
__global__ __launch_bounds__(NumWarps * 32, 1) void sendAnsKernel(
    P2pIbgdaTransportDevice* transport,
    const void* buffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes) {
  using Comp = AnsCompress<NumWarps, PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES>;
  auto group = make_block_group();
  Timeout timeout(kDefaultDeviceTimeoutCycles);
  timeout.start();

  // maxSignalBytes == 0 exercises the transport's 0-sentinel (derives a
  // trap-safe chunk size via CopyOp::max_safe_chunk_size_for_slot()); a
  // non-zero value drives the explicit signaled-chunk-size path. Trailing
  // nullptr is AnsCompress::send()'s alignedAuxBuf (src is 16-byte aligned and
  // chunk offsets are 512-byte multiples, so the internal realign path is
  // dead).
  transport->send<Comp>(
      group,
      buffer,
      nbytes,
      maxSignalBytes,
      timeout,
      /*alignedAuxBuf=*/static_cast<char*>(nullptr));
}

template <int NumWarps>
__global__ __launch_bounds__(NumWarps * 32, 1) void recvAnsKernel(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes) {
  using Comp = AnsCompress<NumWarps, PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES>;
  auto group = make_block_group();
  Timeout timeout(kDefaultDeviceTimeoutCycles);
  timeout.start();

  // maxSignalBytes == 0 exercises the transport's 0-sentinel; a non-zero value
  // drives the explicit signaled-chunk-size path. Sender and receiver must pass
  // the same value so they derive the identical chunking.
  transport->recv<Comp>(group, buffer, nbytes, maxSignalBytes, timeout);
}

} // namespace

void testSendRecvAns(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    bool send,
    int numBlocks,
    int blockSize) {
  const int numWarps = blockSize / 32;
  if (send) {
    switch (numWarps) {
      case 4:
        sendAnsKernel<4><<<numBlocks, blockSize>>>(
            transport, buffer, nbytes, maxSignalBytes);
        break;
      case 8:
        sendAnsKernel<8><<<numBlocks, blockSize>>>(
            transport, buffer, nbytes, maxSignalBytes);
        break;
      case 16:
        sendAnsKernel<16><<<numBlocks, blockSize>>>(
            transport, buffer, nbytes, maxSignalBytes);
        break;
      default:
        throw std::runtime_error(
            "testSendRecvAns: unsupported blockSize " +
            std::to_string(blockSize) + " (need 128, 256, or 512)");
    }
  } else {
    switch (numWarps) {
      case 4:
        recvAnsKernel<4><<<numBlocks, blockSize>>>(
            transport, buffer, nbytes, maxSignalBytes);
        break;
      case 8:
        recvAnsKernel<8><<<numBlocks, blockSize>>>(
            transport, buffer, nbytes, maxSignalBytes);
        break;
      case 16:
        recvAnsKernel<16><<<numBlocks, blockSize>>>(
            transport, buffer, nbytes, maxSignalBytes);
        break;
      default:
        throw std::runtime_error(
            "testSendRecvAns: unsupported blockSize " +
            std::to_string(blockSize) + " (need 128, 256, or 512)");
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("ANS send/recv kernel launch failed: ") +
        cudaGetErrorString(err));
  }
}

} // namespace comms::prims::test
