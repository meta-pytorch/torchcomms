// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cstddef>
#include <type_traits>
#include <utility>

#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/MemcpyCopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"

namespace comms::prims {
namespace {

// Representative fixed-size policy instantiations, mirroring the concrete
// template arguments used at the production reduce-scatter call sites in
// collectives/*.cu.
using TileReducePolicy = TileReduce<float, SumOp, 16384, 512>;
using TileReduceStagedPolicy = TileReduceStaged<float, SumOp, 24576, 384>;

// Return type of a policy's send()/recv() when invoked with the fixed-size
// argument list every CopyOp policy accepts. Evaluated in an unevaluated
// decltype context, so no device call is emitted.
template <typename Policy>
using send_result_t = decltype(Policy::send(
    std::declval<char*>(),
    std::declval<const char*>(),
    std::declval<std::size_t>(),
    std::declval<ThreadGroup&>(),
    std::declval<std::size_t>()));

template <typename Policy>
using recv_result_t = decltype(Policy::recv(
    std::declval<char*>(),
    std::declval<const char*>(),
    std::declval<std::size_t>(),
    std::declval<ThreadGroup&>(),
    std::declval<std::size_t>(),
    std::declval<const char*>()));

// The wire-bytes contract: a fixed-size CopyOp policy reports the number of
// bytes it produced by returning std::size_t from send()/recv(). Every call
// site discards this value, so without this compile-time guard a policy that
// silently kept a void send()/recv() would still compile clean.
template <typename Policy>
constexpr bool has_fixed_size_wire_contract_v = !Policy::kVariableSize &&
    std::is_same_v<send_result_t<Policy>, std::size_t> &&
    std::is_same_v<recv_result_t<Policy>, std::size_t>;

static_assert(
    has_fixed_size_wire_contract_v<Memcpy>,
    "Memcpy: fixed-size CopyOp send()/recv() must return std::size_t (wire bytes)");
static_assert(
    has_fixed_size_wire_contract_v<TileReducePolicy>,
    "TileReduce: fixed-size CopyOp send()/recv() must return std::size_t (wire bytes)");
static_assert(
    has_fixed_size_wire_contract_v<TileReduceStagedPolicy>,
    "TileReduceStaged: fixed-size CopyOp send()/recv() must return std::size_t (wire bytes)");

} // namespace
} // namespace comms::prims
