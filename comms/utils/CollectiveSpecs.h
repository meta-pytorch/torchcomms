// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>

/*
 * Names a collective and the algorithm that runs it.
 *
 * Keep this header a leaf -- <cstdint> and nothing more -- so that layers below
 * MCCL can name a collective without depending on the MCCL interface headers,
 * which would close a cycle.
 */
namespace comms::collectives {

/*
 * `Unknown` is zero on both enums so a zero-filled record reads as "not set"
 * rather than as a real collective. It is never selectable: parsing rejects it
 * and dispatch never routes to it.
 *
 * Fixed to uint8_t so a value packs into a telemetry key without narrowing.
 */
enum class Collective : uint8_t {
  Unknown = 0,
  AllReduce,
  AllGather,
  ReduceScatter,
  SendRecv,
};

enum class Algorithm : uint8_t {
  Unknown = 0,
  Direct,
  Ring,
  Tree,
  Max = Tree,
};

} // namespace comms::collectives
