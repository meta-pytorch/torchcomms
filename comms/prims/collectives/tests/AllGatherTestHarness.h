// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

namespace comms::prims::test {

struct AllGatherTestConfig {
  std::size_t sendcount;
  int num_blocks;
  std::string name;
};

class AllGatherTestBase : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void fill_sendbuf(char* sendbuf_d, std::size_t sendcount) {
    std::vector<char> h_send(sendcount);
    uint8_t pattern = static_cast<uint8_t>(globalRank + 1);
    for (std::size_t i = 0; i < sendcount; i++) {
      h_send[i] = static_cast<char>((pattern + i) % 256);
    }
    CUDACHECK_TEST(cudaMemcpy(
        sendbuf_d, h_send.data(), sendcount, cudaMemcpyHostToDevice));
  }

  void verify_allgather(const char* recvbuf_d, std::size_t sendcount) {
    std::size_t total = sendcount * worldSize;
    std::vector<char> h_recv(total);
    CUDACHECK_TEST(
        cudaMemcpy(h_recv.data(), recvbuf_d, total, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int rank = 0; rank < worldSize; rank++) {
      uint8_t pattern = static_cast<uint8_t>(rank + 1);
      for (std::size_t i = 0; i < sendcount; i++) {
        char expected = static_cast<char>((pattern + i) % 256);
        char actual = h_recv[rank * sendcount + i];
        if (actual != expected) {
          if (errors < 10) {
            EXPECT_EQ(actual, expected)
                << "Mismatch at rank=" << rank << " offset=" << i
                << " (my_rank=" << globalRank << ")";
          }
          errors++;
        }
      }
    }
    EXPECT_EQ(errors, 0) << "Total mismatches: " << errors << " out of "
                         << total << " bytes (my_rank=" << globalRank << ")";
  }
};

} // namespace comms::prims::test
