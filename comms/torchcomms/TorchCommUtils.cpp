// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/TorchCommUtils.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <torch/csrc/distributed/c10d/FileStore.hpp> // @manual
#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // @manual
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual
#include "comms/torchcomms/TorchCommLogging.hpp"

namespace torch {
namespace comms {

bool string_to_bool(const std::string& str) {
  std::string lowercase_str = str;
  std::transform(
      lowercase_str.begin(),
      lowercase_str.end(),
      lowercase_str.begin(),
      [](unsigned char c) { return std::tolower(c); });

  bool is_true =
      (lowercase_str == "1" || lowercase_str == "true" ||
       lowercase_str == "yes" || lowercase_str == "y");
  bool is_false =
      (lowercase_str == "0" || lowercase_str == "false" ||
       lowercase_str == "no" || lowercase_str == "n");

  if (!is_true && !is_false) {
    throw std::runtime_error("Invalid value for string " + str);
  } else {
    return is_true;
  }
}

template <typename T>
T env_to_value(const std::string& env_key, const T& default_value) {
  const char* env_value = std::getenv(env_key.c_str());
  if (!env_value) {
    return default_value; // Environment variable not set, return default
  }

  std::string value(env_value);

  if constexpr (std::is_same_v<T, bool>) {
    return string_to_bool(value);
  } else if constexpr (std::is_same_v<T, std::string>) {
    // String conversion (just return the value as is)
    return std::move(value);
  } else {
    // For all other types, use stringstream for conversion
    try {
      T result;
      std::istringstream ss(value);
      ss >> result;

      if (ss.fail() || !ss.eof()) {
        throw std::runtime_error("Conversion failed");
      }

      return result;
    } catch (const std::exception&) {
      throw std::runtime_error(
          "Invalid value for environment variable " + env_key + ": " + value);
    }
  }
}

// Explicit instantiations for common types
template bool env_to_value<bool>(const std::string&, const bool&);
template int env_to_value<int>(const std::string&, const int&);
template float env_to_value<float>(const std::string&, const float&);
template double env_to_value<double>(const std::string&, const double&);
template std::string env_to_value<std::string>(
    const std::string&,
    const std::string&);

std::pair<int, int> query_ranksize() {
  // Constants for ranksize query methods
  const std::string kRanksizeQueryMethodAuto = "auto";
  const std::string kRanksizeQueryMethodTorchrun = "torchrun";
  const std::string kRanksizeQueryMethodMPI = "mpi";
  const std::string& kRanksizeQueryMethodDefault = kRanksizeQueryMethodAuto;

  // Get the ranksize query method from environment variable
  const char* ranksize_env =
      std::getenv("TORCHCOMM_BOOTSTRAP_RANKSIZE_QUERY_METHOD");
  std::string ranksize_query_method;
  if (ranksize_env == nullptr) {
    ranksize_query_method = kRanksizeQueryMethodDefault;
  } else {
    ranksize_query_method = ranksize_env;
  }

  // Convert to lowercase for comparison
  std::transform(
      ranksize_query_method.begin(),
      ranksize_query_method.end(),
      ranksize_query_method.begin(),
      [](unsigned char c) { return std::tolower(c); });

  int rank = -1;
  int comm_size = -1;

  do {
    // Read from TORCHCOMM_RANK and TORCHCOMM_SIZE environment variables
    rank = env_to_value<int>("TORCHCOMM_RANK", -1);
    comm_size = env_to_value<int>("TORCHCOMM_SIZE", -1);
    if (rank != -1 && comm_size != -1) {
      break;
    }

    // See if we are in an MPI environment
    if (ranksize_query_method == kRanksizeQueryMethodAuto ||
        ranksize_query_method == kRanksizeQueryMethodMPI) {
      // See if we are in an OpenMPI environment
      rank = env_to_value<int>("OMPI_COMM_WORLD_RANK", -1);
      comm_size = env_to_value<int>("OMPI_COMM_WORLD_SIZE", -1);
      if (rank != -1 && comm_size != -1) {
        break;
      }

      // See if we are in an MPICH environment
      rank = env_to_value<int>("PMI_RANK", -1);
      comm_size = env_to_value<int>("PMI_SIZE", -1);
      if (rank != -1 && comm_size != -1) {
        break;
      }
    }

    // See if we are in a Torchrun environment
    if (ranksize_query_method == kRanksizeQueryMethodAuto ||
        ranksize_query_method == kRanksizeQueryMethodTorchrun) {
      rank = env_to_value<int>("RANK", -1);
      comm_size = env_to_value<int>("WORLD_SIZE", -1);
      if (rank != -1 && comm_size != -1) {
        break;
      }
    }
  } while (0);

  if (rank == -1 || comm_size == -1) {
    throw std::runtime_error(
        "Unable to determine rank and size from environment variables. "
        "Please set TORCHCOMM_RANK and TORCHCOMM_SIZE, or ensure you are "
        "running in a supported environment (Torchrun or MPI).");
  }

  return std::make_pair(rank, comm_size);
}

} // namespace comms
} // namespace torch
