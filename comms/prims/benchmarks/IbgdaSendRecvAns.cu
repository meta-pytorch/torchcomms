// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// ANS (variable-size CopyOp) send/recv benchmark kernels for the IBGDA
// transport. Built with `--device-c` + `PIPES_ENABLE_ANS_COMPRESSION` and
// device-linked against the nvcompdx fatbin (see `:ibgda_sendrecv_ans_kernels`
// in BUCK) so `AnsCompress<...>::send/recv` resolve. This TU is the transport
// benchmark's first client of the variable-size compressed send/recv path
// added in D111967119; it lives in its own translation unit + device-link
// target (mirroring `:ans_copy_op_bench`) so the plain `Memcpy` kernels in
// IbgdaSendRecv.cu stay nvcompdx-free and pay no device-link cost.

#include "comms/prims/benchmarks/IbgdaSendRecvAns.h"

#include "comms/prims/core/AbortCheck.cuh"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims::benchmark {

namespace {

// Each block is one cooperative compress/decompress group. AnsCompress is
// instantiated for 8 warps (256 threads): the warp count drives the per-block
// nvcompdx scratch + register footprint, so raising it too far trips
// cudaErrorLaunchOutOfResources for the combined ANS + transport kernel. The
// grid launches one such block per channel (numBlocks), matching the plain
// Memcpy launchers.
constexpr int kAnsNumWarps = 8;
// Both kernels ask ptxas for 8 resident blocks/SM (8 * 256 == 2048 threads ==
// a fully occupied SM). That caps the compiler at 65536/2048 == 32 registers
// per thread, which is well under what the ANS codec wants: send() drops from
// ~195 registers / 256 B of stack to 32 / 800 B, i.e. the codec's live state
// moves into local memory. Occupancy is being bought with spills here.
constexpr int kAnsMinBlocksPerSm = 8;
using BenchAnsCompress =
    AnsCompress<kAnsNumWarps, PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES>;

__device__ __forceinline__ std::size_t ans_section_bytes(
    P2pIbgdaTransportDevice* transport,
    std::size_t totalBytes) {
  return min(transport->channel_layout().data_buffer_size(), totalBytes);
}

} // namespace

__global__ void __launch_bounds__(kAnsNumWarps * 32, kAnsMinBlocksPerSm)
    ibgda_send_ans_kernel(
        P2pIbgdaTransportDevice* transport,
        char* src,
        std::size_t totalBytes,
        AbortDevice abortDevice) {
  auto group = make_block_group();

  const std::size_t sectionBytes = ans_section_bytes(transport, totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    TiledBuffer<char> tiles(src + s * sectionBytes, sectionBytes, group);
    // max_signal_bytes = 0 exercises the transport's 0-sentinel, which derives
    // the trap-safe chunk size via CopyOp::max_safe_chunk_size_for_slot().
    // Trailing nullptr is AnsCompress::send()'s alignedAuxBuf (src is a
    // cudaMalloc base stepped by 512-byte-aligned chunk offsets => 16B
    // aligned).
    transport->send<BenchAnsCompress>(
        group,
        tiles.data(),
        tiles.bytes(),
        /*max_signal_bytes=*/0,
        abortDevice,
        /*alignedAuxBuf=*/static_cast<char*>(nullptr));
  }
}

__global__ void __launch_bounds__(kAnsNumWarps * 32, kAnsMinBlocksPerSm)
    ibgda_recv_ans_kernel(
        P2pIbgdaTransportDevice* transport,
        char* dst,
        std::size_t totalBytes,
        AbortDevice abortDevice) {
  auto group = make_block_group();

  const std::size_t sectionBytes = ans_section_bytes(transport, totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    TiledBuffer<char> tiles(dst + s * sectionBytes, sectionBytes, group);
    // max_signal_bytes = 0: exercise the transport's 0-sentinel (derives the
    // trap-safe chunk size via CopyOp::max_safe_chunk_size_for_slot()).
    transport->recv<BenchAnsCompress>(
        group,
        tiles.data(),
        tiles.bytes(),
        /*max_signal_bytes=*/0,
        abortDevice);
  }
}

void launch_ibgda_send_ans(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    AbortDevice abortDevice) {
  ibgda_send_ans_kernel<<<numBlocks, kAnsNumWarps * 32, 0, stream>>>(
      transport, src, nbytes, abortDevice);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "[PIPES] ANS send kernel launch failed: %s\n", cudaGetErrorString(err));
  }
}

void launch_ibgda_recv_ans(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    AbortDevice abortDevice) {
  ibgda_recv_ans_kernel<<<numBlocks, kAnsNumWarps * 32, 0, stream>>>(
      transport, dst, nbytes, abortDevice);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "[PIPES] ANS recv kernel launch failed: %s\n", cudaGetErrorString(err));
  }
}

} // namespace comms::prims::benchmark
