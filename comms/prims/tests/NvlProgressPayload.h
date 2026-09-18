// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace comms::prims::test {

/*
 * Position- and key-dependent payload byte for the NVL progress oracles.
 *
 * Shared rather than duplicated per test file on purpose: this is the oracle
 * itself, and two copies that drifted apart would silently weaken whichever
 * side stopped matching its producer.
 *
 * A constant fill cannot tell a correct chunk from one that was replayed,
 * reordered, or written at the wrong offset, and a pattern with a short period
 * cannot tell apart offsets one period apart. This is a splitmix64 finalizer
 * over (rank, peer, index): it avoids the short structured periods a simple
 * arithmetic pattern has, and it is keyed so a chunk from the wrong peer
 * differs from the expected bytes even at the right offset.
 *
 * The output is one byte, so this does not make individual offsets unique --
 * any two unrelated offsets still agree about 1 time in 256. The strength is in
 * checking the whole buffer: a misplaced chunk has to collide at every one of
 * its bytes to pass. A rotated-chunk negative control is what demonstrates
 * that, rather than the per-byte properties of the hash.
 *
 * Channel is deliberately not a key. Channels partition the buffer into
 * disjoint tiles, so a channel swap relocates bytes and the index term already
 * catches it; keying on channel too would mean threading tile geometry through
 * every caller for no added detection.
 */
inline unsigned char
progressPayloadByte(int rank, int peer, std::size_t index) {
  uint64_t h = (static_cast<uint64_t>(rank) << 56) ^
      (static_cast<uint64_t>(peer) << 48) ^ static_cast<uint64_t>(index);
  h ^= h >> 30;
  h *= 0xbf58476d1ce4e5b9ULL;
  h ^= h >> 27;
  h *= 0x94d049bb133111ebULL;
  h ^= h >> 31;
  return static_cast<unsigned char>(h & 0xFFULL);
}

// Fills a host buffer with `rank`'s payload for `peer`, ready to upload.
inline std::vector<char>
makeProgressPayload(int rank, int peer, std::size_t nbytes) {
  std::vector<char> host(nbytes);
  for (std::size_t i = 0; i < nbytes; ++i) {
    host[i] = static_cast<char>(progressPayloadByte(rank, peer, i));
  }
  return host;
}

} // namespace comms::prims::test
