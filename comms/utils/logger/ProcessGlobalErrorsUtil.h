// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// This allows us to keep track of errors that affect the entire process,
// not necessarily a specific communicator
class ProcessGlobalErrorsUtil {
 public:
  struct ErrorAndStackTrace {
    // timestamp this error was reported
    std::chrono::milliseconds timestampMs{};
    std::string errorMessage;
    std::vector<std::string> stackTrace;
  };

  struct NicError {
    // timestamp this error was reported
    std::chrono::milliseconds timestampMs{};
    std::string errorMessage;
  };

  struct IbCompletionError {
    std::chrono::milliseconds timestampMs{};
    std::string peer;
    std::string statusStr;
    int status{0};
    std::string opcodeStr;
    int opcode{0};
    int reqSize{0};
    uint32_t vendorErr{0};
    std::string reqType;
    std::string localGid;
    std::string remoteGid;
    std::string hcaName;
    std::string scaleupDomain;
    std::string localHostname;
  };

  struct CudaError {
    std::chrono::milliseconds timestampMs{};
    std::string errorString; // from cudaGetErrorString(err)
    int errorCode{0}; // the raw cudaError_t value
  };

  struct State {
    // Map of device name -> port -> error message
    std::unordered_map<std::string, std::unordered_map<int, NicError>> badNics;
    std::deque<ErrorAndStackTrace> errorAndStackTraces;
    std::deque<IbCompletionError> ibCompletionErrors;
    std::string scaleupDomain;
    std::string hostname;
    std::deque<CudaError> cudaErrors;
  };

  // Report an error on a NIC. If errorMessage is std::nullopt, then
  // the error is cleared.
  static void setNic(
      const std::string& devName,
      int port,
      std::optional<std::string> errorMessage);

  // Report an internal error and stack trace
  static void addErrorAndStackTrace(
      std::string errorMessage,
      std::vector<std::string> stackTrace);

  // Report an IB completion error
  static void addIbCompletionError(IbCompletionError error);

  // Clear all stored IB completion errors (called after reporting to avoid
  // duplicate reporting)
  static void clearIbCompletionErrors();

  // Get cached hostname (reads /etc/fbwhoami on first call)
  static std::string getHostname();

  // Get cached scaleup domain (reads /etc/fbwhoami on first call)
  static std::string getScaleupDomain();

  // Report a CUDA error
  static void addCudaError(CudaError error);

  // Clear all stored CUDA errors (called after reporting to avoid
  // duplicate reporting)
  static void clearCudaErrors();

  static State getAllState();
};
