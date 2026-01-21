// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * Simple example demonstrating synchronous send/recv communication using
 * TorchComm.
 *
 * This example shows how to perform point-to-point communication between ranks
 * in a synchronous manner, where operations block until completion.
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
            << ": Starting synchronous send/recv example on device "
            << device_id << std::endl;

  // Create a tensor with rank-specific data
  const int tensor_size = 1024;
  auto send_tensor = torch::full(
      {tensor_size},
      static_cast<float>(rank),
      torch::dtype(torch::kFloat32).device(target_device));
  auto recv_tensor = torch::zeros(
      {tensor_size}, torch::dtype(torch::kFloat32).device(target_device));

  // Calculate send and receive ranks (ring topology)
  int send_rank = (rank + 1) % world_size;
  int recv_rank = (rank - 1 + world_size) % world_size;

  std::cout << "Rank " << rank << ": Sending to rank " << send_rank
            << ", receiving from rank " << recv_rank << std::endl;

  // Perform synchronous send/recv operations
  // Use alternating pattern to avoid deadlock
  if (rank % 2 == 0) {
    // Even ranks: send first, then receive
    torchcomm->send(send_tensor, send_rank, false);
    torchcomm->recv(recv_tensor, recv_rank, false);
  } else {
    // Odd ranks: receive first, then send
    torchcomm->recv(recv_tensor, recv_rank, false);
    torchcomm->send(send_tensor, send_rank, false);
  }

  // Synchronize CUDA stream
  torch::cuda::synchronize();

  std::cout << "Rank " << rank << ": Synchronous send/recv example completed"
            << std::endl;

  // Finalize TorchComm before returning
  torchcomm->finalize();

  return 0;
}
