// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * Simple example demonstrating synchronous AllReduce communication using
 * TorchComm.
 *
 * This example shows how to perform collective AllReduce operations between all
 * ranks in a synchronous manner, where the operation blocks until completion.
 */

#include <iostream>

#include <torch/torch.h>
#include "comms/torchcomms/TorchComm.hpp"

using namespace torch::comms;

int main() {
  // Initialize TorchComm with proper parameters
  at::Device device(at::kCUDA);
  auto torchcomm = new_comm("ncclx", device, "main_comm");

  // Get rank and world size
  int rank = torchcomm->getRank();
  int world_size = torchcomm->getSize();

  // Calculate device ID based on rank and number of available devices
  int num_devices = torch::cuda::device_count();
  int device_id = rank % num_devices;
  at::Device target_device(at::kCUDA, device_id);

  std::cout << "Rank " << rank << "/" << world_size
            << ": Starting synchronous AllReduce example on device "
            << device_id << std::endl;

  // Create a tensor with rank-specific data
  const int tensor_size = 1024;
  auto tensor = torch::full(
      {tensor_size},
      static_cast<float>(rank + 1),
      torch::dtype(torch::kFloat32).device(target_device));

  std::cout << "Rank " << rank
            << ": Initial tensor value: " << tensor[0].item<float>()
            << std::endl;

  // Perform synchronous AllReduce operation (sum reduction)
  torchcomm->all_reduce(tensor, ReduceOp::SUM, false);

  // Synchronize CUDA stream
  torch::cuda::synchronize();

  std::cout << "Rank " << rank << ": Synchronous AllReduce example completed"
            << std::endl;

  // Finalize TorchComm before returning
  torchcomm->finalize();

  return 0;
}
