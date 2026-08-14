// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>

/* The vocabulary a collstats key is written in.
 *
 * Owned here rather than borrowed from a producer, because collstats is a sink
 * with more than one of them. MCCL/ctran dispatch names an algorithm NCCL has
 * no word for (`Direct`), NCCL names several MCCL has no word for (`NVLS`,
 * `CollNet*`, `PAT`) and a protocol axis MCCL does not model at all. Neither
 * side's enum is a superset, so keying on either would leave the other unable
 * to say what it did. These are the union; each producer translates into them.
 *
 * A translator is also where drift gets caught: a `switch` over a producer's
 * enum with no `default:` stops compiling when that producer gains a member,
 * whereas sharing its enum would silently admit a value with no bucket here.
 *
 * The numeric values are an in-process detail, not a wire format: the exporter
 * writes `magic_enum::enum_name`, so a consumer reads "AllReduce" rather than 1
 * and renumbering cannot re-label recorded data.
 *
 * `Unknown = 0` on each, so a zero-filled bank slot reads as unset rather than
 * as a real collective -- an untouched slot would otherwise be
 * indistinguishable from a genuine small-message `Direct` `AllReduce`. Sized to
 * u8 to match the `CollStatKey` fields that hold them. */

namespace meta::comms::collstats {

enum class CollStatOp : uint8_t {
  Unknown = 0,
  AllReduce,
  AllGather,
  ReduceScatter,
  Broadcast,
  Reduce,
  AllToAll,
  Scatter,
  Gather,
  SendRecv,
  Send,
  Recv,
};

enum class CollStatAlgo : uint8_t {
  Unknown = 0,
  Direct,
  Ring,
  Tree,
  CollNetDirect,
  CollNetChain,
  NVLS,
  NVLSTree,
  PAT,
};

/* Modelled by NCCL, not by MCCL/ctran, which has no protocol axis: the ctran
 * producer reports Unknown and the field waits for a producer that has one. */
enum class CollStatProto : uint8_t {
  Unknown = 0,
  Simple,
  LL,
  LL128,
};

static_assert(
    sizeof(CollStatOp) == sizeof(uint8_t) &&
        sizeof(CollStatAlgo) == sizeof(uint8_t) &&
        sizeof(CollStatProto) == sizeof(uint8_t),
    "key codes must match the width of the CollStatKey fields holding them");

} // namespace meta::comms::collstats
