// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/tests/perf/cpp/PerfTestHelpers.h"

#include <iostream>

void runBarrierPerf(
    std::shared_ptr<torch::comms::TorchComm> comm,
    const PerfParams& params) {
  int rank = comm->getRank();
  int num_ranks = comm->getSize();

  if (rank == 0) {
    std::string mode = params.async ? "Asynchronous" : "Synchronous";
    std::cout << "\n=== " << mode << " Barrier Performance ===" << std::endl;
  }
  printPerfHeader(rank);

  // Barrier has no message size, run once
  // Warmup
  for (int i = 0; i < params.warmup_iterations; i++) {
    auto work = comm->barrier(params.async);
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
    auto work = comm->barrier(params.async);
    if (params.async) {
      work->wait();
    }

    if (params.iteration_window > 0 && (i + 1) % params.iteration_window == 0) {
      at::cuda::getCurrentCUDAStream().synchronize();
    }
  }

  timer.stop();

  // Calculate statistics
  double total = timer.elapsed_us();
  double avg_time = total / params.measure_iterations;

  PerfResult result{};
  result.message_size_bytes = 0;
  result.num_ranks = num_ranks;
  result.iterations = params.measure_iterations;
  result.total_time_us = total;
  result.avg_time_us = avg_time;
  result.min_time_us = avg_time;
  result.max_time_us = avg_time;
  // Barrier has no data transfer, so bus bandwidth is 0
  result.bus_bw_gbps = 0.0;

  printPerfResult(result, rank);
}
