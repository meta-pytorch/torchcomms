// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/tests/perf/cpp/PerfTestHelpers.h"

#include <fmt/format.h>
#include <iostream>

void runReducePerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params,
    int root) {
  int rank = comm->getRank();
  int num_ranks = comm->getSize();

  if (rank == 0) {
    std::string mode = params.async ? "Asynchronous" : "Synchronous";
    std::cout << "\n=== " << mode << " Reduce Performance ===" << std::endl;
  }
  printPerfHeader(rank);

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
      auto work =
          comm->reduce(tensor, root, torch::comms::ReduceOp::SUM, params.async);
      if (params.async) {
        work->wait();
      }
    }

    // Synchronize all ranks before measurement
    comm->barrier(false);

    // Measure using CUDA events
    PerfTimer timer;
    timer.start();

    for (int i = 0; i < params.measure_iterations; i++) {
      auto work =
          comm->reduce(tensor, root, torch::comms::ReduceOp::SUM, params.async);
      if (params.async) {
        work->wait();
      }

      if (params.iteration_window > 0 &&
          (i + 1) % params.iteration_window == 0) {
        at::cuda::getCurrentCUDAStream().synchronize();
      }
    }

    timer.stop();

    // Calculate statistics
    double total = timer.elapsed_us();
    double avg_time = total / params.measure_iterations;

    PerfResult result{};
    result.message_size_bytes = num_elements * element_size;
    result.num_ranks = num_ranks;
    result.iterations = params.measure_iterations;
    result.total_time_us = total;
    result.avg_time_us = avg_time;
    result.min_time_us = avg_time;
    result.max_time_us = avg_time;
    // Reduce bus bandwidth: size / time (no reduction factor)
    double algo_bw =
        result.message_size_bytes / avg_time; // bytes per us = MB/s
    result.bus_bw_gbps = algo_bw / 1000.0; // Convert to GB/s

    printPerfResult(result, rank);
  }
}
