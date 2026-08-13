// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Shared 2-rank send/recv correctness sweep, parameterized by the per-protocol
// launch function (launchLLSendRecv / launchSimpleSendRecv). Host-only: keeps
// device/DOCA headers out via the forward-declared P2pIbgdaTransportDevice in
// LLIbgdaSendRecv.cuh.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/prims/transport/llx/tests/LLIbgdaSendRecv.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

namespace comms::prims::test {

using SendRecvLaunchFn = void (*)(
    P2pIbgdaTransportDevice*,
    void*,
    std::size_t,
    int,
    std::size_t,
    bool,
    int,
    int);

// Rank 0 -> rank 1 send/recv over a sweep of PAYLOAD sizes on one shared
// transport, via `launch` (LL or Simple). `label` tags failure messages.
// Sizes span: sub-slot, one slot payload, the full pipeline window, past the
// window (forces slot reuse + gen advance), and a non-4B tail. A fresh pattern
// per size guards against stale-slot matches. Skips when RDMA is unavailable.
inline void runSendRecvSweep(
    int globalRank,
    int numRanks,
    int localRank,
    SendRecvLaunchFn launch,
    const char* label) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  // dataBufferSize is WIRE bytes per slot; LL payload capacity is half.
  const std::size_t dataBufferSize = 64 * 1024;
  const int pipelineDepth = 2;
  const int numBlocks = 1; // maxChannels
  const int blockSize = 128;
  const std::size_t maxSignalBytes = 0; // full-slot chunks
  const int peerRank = (globalRank == 0) ? 1 : 0;

  const std::vector<std::size_t> sizes = {
      4096, // sub-slot
      32 * 1024, // one LL slot payload
      64 * 1024, // full LL pipeline window
      192 * 1024, // past the window -> slot reuse + gen advance (both protos)
      6001, // non-4B tail (partial last LL packet)
  };
  std::size_t maxBytes = 0;
  for (auto s : sizes) {
    maxBytes = std::max(maxBytes, s);
  }

  try {
    MultipeerIbgdaTransportConfig config{
        .cudaDevice = localRank,
        .perChannelSize = dataBufferSize / numBlocks,
        .max_num_channels = numBlocks,
        .pipelineDepth = pipelineDepth,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    auto transport = std::make_unique<MultipeerIbgdaTransport>(
        globalRank, numRanks, bootstrap, config);
    transport->exchange();

    P2pIbgdaTransportDevice* peerTransport =
        transport->getP2pTransportDevice(peerRank);

    meta::comms::DeviceBuffer sendBuffer(maxBytes);
    meta::comms::DeviceBuffer recvBuffer(maxBytes);
    meta::comms::DeviceBuffer errorCountBuf(sizeof(int));
    auto* d_errorCount = static_cast<int*>(errorCountBuf.get());

    for (std::size_t i = 0; i < sizes.size(); ++i) {
      const std::size_t nbytes = sizes[i];
      const uint8_t pattern = static_cast<uint8_t>(0x11 + i);

      if (globalRank == 0) {
        fillPattern(sendBuffer.get(), nbytes, pattern, numBlocks, blockSize);
      } else {
        CUDACHECK_TEST(cudaMemset(recvBuffer.get(), 0, nbytes));
      }
      CUDACHECK_TEST(cudaDeviceSynchronize());
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      launch(
          peerTransport,
          globalRank == 0 ? sendBuffer.get() : recvBuffer.get(),
          nbytes,
          numBlocks,
          maxSignalBytes,
          /*send=*/globalRank == 0,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      if (globalRank == 1) {
        CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));
        verifyPattern(
            recvBuffer.get(),
            nbytes,
            pattern,
            d_errorCount,
            numBlocks,
            blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        int h_errorCount = 0;
        CUDACHECK_TEST(cudaMemcpy(
            &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));
        EXPECT_EQ(h_errorCount, 0)
            << label << " send/recv corrupted " << h_errorCount << " of "
            << nbytes << " bytes (size index " << i << ")";
      }
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

} // namespace comms::prims::test
