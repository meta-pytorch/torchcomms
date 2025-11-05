// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * Simple example demonstrating asynchronous send/recv communication using
 * TorchComm.
 *
 * This example shows how to perform point-to-point communication between ranks
 * in an asynchronous manner, where operations return immediately and can be
 * waited on later for completion.
 */

#include <iostream>
#include <memory>

#include <torch/torch.h>
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/TorchWork.hpp"

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
            << ": Starting asynchronous send/recv example on device "
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

  // Perform asynchronous send/recv operations
  // Use alternating pattern to avoid deadlock
  c10::intrusive_ptr<TorchWork> send_work = nullptr;
  c10::intrusive_ptr<TorchWork> recv_work = nullptr;

  if (rank % 2 == 0) {
    // Even ranks: send first, then receive
    send_work = torchcomm->send(send_tensor, send_rank, true);
    recv_work = torchcomm->recv(recv_tensor, recv_rank, true);
  } else {
    // Odd ranks: receive first, then send
    recv_work = torchcomm->recv(recv_tensor, recv_rank, true);
    send_work = torchcomm->send(send_tensor, send_rank, true);
  }

  // Wait for operations to complete
  if (send_work != nullptr) {
    send_work->wait();
    std::cout << "Rank " << rank << ": Send operation completed" << std::endl;
  }

  if (recv_work != nullptr) {
    recv_work->wait();
    std::cout << "Rank " << rank << ": Receive operation completed"
              << std::endl;
  }

  // Synchronize CUDA stream
  torch::cuda::synchronize();

  std::cout << "Rank " << rank << ": Asynchronous send/recv example completed"
            << std::endl;

  // Finalize TorchComm before returning
  torchcomm->finalize();

  return 0;
}
