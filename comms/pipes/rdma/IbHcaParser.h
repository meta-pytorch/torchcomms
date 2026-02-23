// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <vector>

namespace comms::pipes {

/**
 * Match mode for IB HCA filtering.
 *
 * Determines how device names are matched against filter entries:
 * - PREFIX_INCLUDE: Include devices whose name starts with an entry (default)
 * - EXACT_INCLUDE: Include devices whose name exactly matches an entry (=)
 * - PREFIX_EXCLUDE: Exclude devices whose name starts with an entry (^)
 * - EXACT_EXCLUDE: Exclude devices whose name exactly matches an entry (^=)
 */
enum class HcaMatchMode {
  PREFIX_INCLUDE,
  EXACT_INCLUDE,
  PREFIX_EXCLUDE,
  EXACT_EXCLUDE,
};

/**
 * A single HCA filter entry parsed from the NCCL_IB_HCA string.
 *
 * Each entry specifies a device name and an optional port number.
 * For example, "mlx5_0:1" means device "mlx5_0", port 1.
 */
struct HcaEntry {
  std::string name;
  int port{-1}; // -1 means "any port"
};

/**
 * IbHcaParser - Parses and applies NCCL_IB_HCA-style device filters.
 *
 * The NCCL_IB_HCA environment variable format is:
 *   [^][=]<device_name>[:<port>][,<device_name>[:<port>],...]
 *
 * Prefix modifiers:
 *   (none) - PREFIX_INCLUDE: include devices matching any entry prefix
 *   =      - EXACT_INCLUDE: include devices matching any entry exactly
 *   ^      - PREFIX_EXCLUDE: exclude devices matching any entry prefix
 *   ^=     - EXACT_EXCLUDE: exclude devices matching any entry exactly
 *
 * Examples:
 *   "mlx5_0,mlx5_1"      - Include devices starting with mlx5_0 or mlx5_1
 *   "=mlx5_0,=mlx5_1"    - Include only exactly mlx5_0 or mlx5_1
 *   "^mlx5_1"             - Exclude devices starting with mlx5_1
 *   "^=mlx5_1"            - Exclude only exactly mlx5_1
 *   "mlx5_0:1"            - Include mlx5_0 port 1 only
 *
 * Usage:
 *   IbHcaParser parser("mlx5_0,mlx5_1");
 *   if (parser.matches("mlx5_0")) { ... }
 *
 *   IbHcaParser empty;  // No filter, matches everything
 *   assert(empty.matches("any_device"));
 */
class IbHcaParser {
 public:
  static constexpr int kMaxHcaDevices = 32;

  /**
   * Construct a parser from the raw NCCL_IB_HCA-style string.
   *
   * @param hcaStr The filter string to parse
   * @throws std::runtime_error if more than kMaxHcaDevices entries
   */
  explicit IbHcaParser(const std::string& hcaStr);

  /**
   * Default constructor - empty filter that matches everything.
   */
  IbHcaParser() = default;

  /**
   * Check if a device name (and optional port) passes the filter.
   *
   * @param devName Device name to check (e.g., "mlx5_0")
   * @param port Port number to check (-1 means ignore port matching)
   * @return true if the device passes the filter
   */
  bool matches(const std::string& devName, int port = -1) const;

  /**
   * Get the match mode.
   */
  HcaMatchMode match_mode() const {
    return matchMode_;
  }

  /**
   * Get the parsed entries.
   */
  const std::vector<HcaEntry>& entries() const {
    return entries_;
  }

  /**
   * Check if the filter is empty (no entries, matches everything).
   */
  bool empty() const {
    return entries_.empty();
  }

 private:
  void parse_prefix(const std::string& hcaStr, size_t& pos);
  HcaEntry parse_entry(const std::string& token) const;

  HcaMatchMode matchMode_{HcaMatchMode::PREFIX_INCLUDE};
  std::vector<HcaEntry> entries_;
};

} // namespace comms::pipes
