// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/TorchCommUtils.hpp"
#include <algorithm>
#include <fstream>
#include <iterator>
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

int count_file_lines(const std::string& filepath, bool ignore_empty_lines) {
  std::ifstream filestream(filepath);
  if (!filestream.is_open()) {
    throw std::runtime_error("Failed to open file for reading: " + filepath);
  }

  int line_count = 0;
  if (ignore_empty_lines) {
    std::string line;
    while (std::getline(filestream, line)) {
      if (!line.empty()) {
        ++line_count;
      }
    }
  } else {
    line_count = std::count(
        std::istreambuf_iterator<char>(filestream),
        std::istreambuf_iterator<char>(),
        '\n');

    // Clear the stream state so we can use tellg/seekg
    filestream.clear();

    // Check if the file is empty
    const bool is_empty_file =
        filestream.tellg() == std::ifstream::pos_type(0) &&
        filestream.peek() == std::ifstream::traits_type::eof();

    // If the file does not end with a newline, we need to add one to the count
    if (!is_empty_file) {
      filestream.seekg(-1, std::ios::end);
      char last_char;
      filestream.get(last_char);
      if (last_char != '\n') {
        ++line_count;
      }
    }
  }

  if (filestream.bad()) {
    throw std::runtime_error("Error while reading file: " + filepath);
  }

  return line_count;
}

std::pair<int, int> query_pals_ranksize() {
  // Try to get rank and size directly from PALS environment variables first
  auto rank = env_to_value<int>("PALS_RANKID", -1);
  // PALS_SIZE is currently not available but may be included in future versions
  auto comm_size = env_to_value<int>("PALS_SIZE", -1);

  // If rank & size are both found from PALS return them
  if (rank > -1 && comm_size > 0) {
    TC_LOG(INFO) << "Found rank and size from PALS environment (rank=" << rank
                 << ", size=" << comm_size << ").";
    return {rank, comm_size};
  }

  // PALS currently supports only the PBS workload manager.
  // We calculate the size using the PBS_NODEFILE (which lists one node per
  // line) to get number of nodes and PALS_LOCAL_SIZE to get the number of ranks
  // per node:
  //   size = (number of nodes) * (ranks per node)
  // which assumes all nodes have the same number of ranks.
  const auto num_ranks_per_node = env_to_value<int>("PALS_LOCAL_SIZE", -1);
  const auto nodefile_path = env_to_value<std::string>("PBS_NODEFILE", "");

  if (comm_size == -1 && !nodefile_path.empty() && num_ranks_per_node > 0) {
    try {
      const auto num_nodes = count_file_lines(nodefile_path);
      if (num_nodes > 0) {
        comm_size = num_nodes * num_ranks_per_node;
      }
    } catch (const std::exception& e) {
      TC_LOG(ERROR) << "Failed to determine size from PALS/PBS environment. "
                    << "Could not count lines in PBS_NODEFILE ("
                    << nodefile_path << "): " << e.what();
      comm_size = -1;
    }
  }

  // Only log warnings if we have partial information, indicating a possible
  // broken PALS/PBS environment. If neither rank nor size are found, we are
  // likely not in a PALS environment.
  if (rank > -1 && comm_size == -1) {
    TC_LOG(WARNING)
        << "Found rank from PALS environment but unable to determine size. "
        << "Please set PALS_SIZE or ensure PBS_NODEFILE and PALS_LOCAL_SIZE are available.";
  } else if (rank == -1 && comm_size > 0) {
    TC_LOG(WARNING)
        << "Found size from PALS/PBS environment but unable to determine rank. "
        << "Please set PALS_RANKID.";
  } else if (rank > -1 && comm_size > 0) {
    TC_LOG(INFO) << "Found rank and size from PALS/PBS environment (rank="
                 << rank << ", size=" << comm_size << ").";
  }

  return {rank, comm_size};
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
  const std::string kRanksizeQueryMethodPALS = "pals";
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
    if (rank > -1 && comm_size > 0) {
      break;
    }

    // See if we are in an MPI environment
    if (ranksize_query_method == kRanksizeQueryMethodAuto ||
        ranksize_query_method == kRanksizeQueryMethodMPI) {
      // See if we are in an OpenMPI environment
      rank = env_to_value<int>("OMPI_COMM_WORLD_RANK", -1);
      comm_size = env_to_value<int>("OMPI_COMM_WORLD_SIZE", -1);
      if (rank > -1 && comm_size > 0) {
        break;
      }

      // See if we are in an MPICH environment
      rank = env_to_value<int>("PMI_RANK", -1);
      comm_size = env_to_value<int>("PMI_SIZE", -1);
      if (rank > -1 && comm_size > 0) {
        break;
      }
    }

    // See if we are in a PALS environment
    if (ranksize_query_method == kRanksizeQueryMethodAuto ||
        ranksize_query_method == kRanksizeQueryMethodPALS) {
      std::tie(rank, comm_size) = query_pals_ranksize();
      if (rank > -1 && comm_size > 0) {
        break;
      }
    }

    // See if we are in a Torchrun environment
    if (ranksize_query_method == kRanksizeQueryMethodAuto ||
        ranksize_query_method == kRanksizeQueryMethodTorchrun) {
      rank = env_to_value<int>("RANK", -1);
      comm_size = env_to_value<int>("WORLD_SIZE", -1);
      if (rank > -1 && comm_size > 0) {
        break;
      }
    }
  } while (0);

  if (rank < 0 || comm_size < 1) {
    throw std::runtime_error(
        "Unable to determine rank and size from environment variables. "
        "Please set TORCHCOMM_RANK and TORCHCOMM_SIZE, or ensure you are "
        "running in a supported environment (Torchrun, MPI, PALS).");
  }

  return std::make_pair(rank, comm_size);
}

} // namespace comms
} // namespace torch
