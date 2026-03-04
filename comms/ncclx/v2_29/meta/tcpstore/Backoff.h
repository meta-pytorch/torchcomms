#pragma once

#include <chrono>
#include <random>
#include <thread>

#include "comms/utils/cvars/nccl_cvars.h"

namespace ncclx::tcpstore {

class Backoff {
 public:
  virtual ~Backoff() = default;

  virtual std::chrono::milliseconds nextBackoff() = 0;
};

class ExponentialBackoffWithJitter : public Backoff {
 public:
  ExponentialBackoffWithJitter();

  std::chrono::milliseconds nextBackoff() override;

 public:
  std::chrono::milliseconds initialInterval{
      NCCL_TCPSTORE_BACKOFF_INITIAL_INTERVAL};
  double randomizationFactor{NCCL_TCPSTORE_BACKOFF_RANDOMIZATION_FACTOR};
  double multiplier{NCCL_TCPSTORE_BACKOFF_MULTIPLIER};
  std::chrono::milliseconds maxInterval{NCCL_TCPSTORE_BACKOFF_MAX_INTERVAL};

 private:
  std::mt19937 gen_;
  std::chrono::milliseconds currentInterval_{0};
};

class FixedBackoff : public Backoff {
 public:
  FixedBackoff(std::chrono::milliseconds interval);

  std::chrono::milliseconds nextBackoff() override;

 private:
  std::chrono::milliseconds interval_;
};

} // namespace ncclx::tcpstore
