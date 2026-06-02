// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <numeric>
#include <optional>
#include <vector>

namespace comms::pipes {

struct RingNeighbors {
  int prev_rank;
  int next_rank;
};

/**
 * Return the prev/next neighbor ranks for my_rank in each of num_rings
 * distinct rings over world_size ranks.
 *
 * Each ring is a Hamiltonian cycle defined by a coprime stride s:
 * next = (my_rank + s) % world_size, prev = (my_rank - s + world_size) %
 * world_size. Different rings use different strides so they traverse
 * different physical network paths.
 *
 * Returns nullopt if fewer than num_rings coprime strides exist
 * (the count of valid strides is Euler's totient phi(world_size)).
 */
inline std::optional<std::vector<RingNeighbors>>
make_standard_rings(int world_size, int my_rank, int num_rings) {
  std::vector<RingNeighbors> rings;
  for (int stride = 1;
       stride < world_size && static_cast<int>(rings.size()) < num_rings;
       ++stride) {
    if (std::gcd(stride, world_size) != 1) {
      continue;
    }
    rings.push_back({
        .prev_rank = (my_rank - stride + world_size) % world_size,
        .next_rank = (my_rank + stride) % world_size,
    });
  }
  if (static_cast<int>(rings.size()) < num_rings) {
    return std::nullopt;
  }
  return rings;
}

/**
 * Return a 2-ring {forward, reverse} topology for bi-directional
 * communication. Ring 0 uses stride 1, ring 1 uses stride W-1
 * (the reverse direction). Returns nullopt for world_size < 3
 * where forward and reverse are identical.
 */
inline std::optional<std::vector<RingNeighbors>> make_bidir_rings(
    int world_size,
    int my_rank) {
  if (world_size < 3) {
    return std::nullopt;
  }
  int fwd_stride = 1;
  int rev_stride = world_size - 1;
  return std::vector<RingNeighbors>{
      {.prev_rank = (my_rank - fwd_stride + world_size) % world_size,
       .next_rank = (my_rank + fwd_stride) % world_size},
      {.prev_rank = (my_rank - rev_stride + world_size) % world_size,
       .next_rank = (my_rank + rev_stride) % world_size},
  };
}

} // namespace comms::pipes
