// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/tests/perf/cpp/PerfTestHelpers.h"

#include <fmt/format.h>
#include <iostream>

void runSendRecvPerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params) {
  int rank = comm->getRank();
  int num_ranks = comm->getSize();

  if (num_ranks < 2) {
    if (rank == 0) {
      std::cout << "SendRecv test requires at least 2 ranks, skipping"
                << std::endl;
    }
    return;
  }

  if (rank == 0) {
    std::string mode = params.async ? "Asynchronous" : "Synchronous";
    std::cout << "\n=== " << mode << " SendRecv Performance ===" << std::endl;
  }
  printPerfHeader(rank);

  // Only ranks 0 and 1 participate in send/recv
  int peer = (rank == 0) ? 1 : 0;

  int64_t element_size = at::elementSize(params.dtype);

  for (int64_t msg_size = params.min_size; msg_size <= params.max_size;
       msg_size *= params.size_scaling_factor) {
    int64_t num_elements = msg_size / element_size;
    if (num_elements == 0) {
      num_elements = 1;
    }
    auto tensor = createTensor(
        num_elements, rank, comm->getDevice().type(), params.dtype);

    // Warmup
    for (int i = 0; i < params.warmup_iterations; i++) {
      if (rank == 0) {
        auto work = comm->send(tensor, peer, params.async);
        if (params.async) {
          work->wait();
        }
        auto work2 = comm->recv(tensor, peer, params.async);
        if (params.async) {
          work2->wait();
        }
      } else if (rank == 1) {
        auto work = comm->recv(tensor, peer, params.async);
        if (params.async) {
          work->wait();
        }
        auto work2 = comm->send(tensor, peer, params.async);
        if (params.async) {
          work2->wait();
        }
      }
    }

    // Synchronize all ranks before measurement
    comm->barrier(false);

    // Measure ping-pong latency using CUDA events
    PerfTimer timer;
    timer.start();

    for (int i = 0; i < params.measure_iterations; i++) {
      if (rank == 0) {
        auto work = comm->send(tensor, peer, params.async);
        if (params.async) {
          work->wait();
        }
        auto work2 = comm->recv(tensor, peer, params.async);
        if (params.async) {
          work2->wait();
        }
      } else if (rank == 1) {
        auto work = comm->recv(tensor, peer, params.async);
        if (params.async) {
          work->wait();
        }
        auto work2 = comm->send(tensor, peer, params.async);
        if (params.async) {
          work2->wait();
        }
      }

      if (params.iteration_window > 0 &&
          (i + 1) % params.iteration_window == 0) {
        at::cuda::getCurrentCUDAStream().synchronize();
      }
    }

    timer.stop();

    // Calculate statistics (half the ping-pong time for one-way latency)
    double total = timer.elapsed_us();
    double avg_time = total / params.measure_iterations;

    PerfResult result{};
    result.message_size_bytes = num_elements * element_size;
    result.num_ranks = 2; // Only 2 ranks participate
    result.iterations = params.measure_iterations;
    result.total_time_us = total;
    result.avg_time_us = avg_time / 2; // One-way latency
    result.min_time_us = avg_time / 2;
    result.max_time_us = avg_time / 2;
    // SendRecv bus bandwidth: size / one-way time
    double algo_bw =
        result.message_size_bytes / result.avg_time_us; // bytes/us = MB/s
    result.bus_bw_gbps = algo_bw / 1000.0; // Convert to GB/s

    printPerfResult(result, rank);
  }
}
