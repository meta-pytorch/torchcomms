// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/prims/collectives/ReduceScatterDirectIbCore.cuh"
#include "comms/prims/collectives/ReduceScatterExecution.h"
#include "comms/prims/core/AbortCheck.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/ThreadGroup.cuh"

namespace comms::prims {

template <typename T>
struct DirectIbStridedInput {
  const T* data{nullptr};
  std::size_t chunkStrideBytes{0};

  __host__ __device__ __forceinline__ const T* chunkData(int rank) const {
    return reinterpret_cast<const T*>(
        reinterpret_cast<const char*>(data) +
        static_cast<std::size_t>(rank) * chunkStrideBytes);
  }
};

template <typename T>
struct DirectIbOutput {
  T* data{nullptr};
  ReduceScatterOutputInitialization initialization{
      ReduceScatterOutputInitialization::COPY_OWN_INPUT};
};

template <typename ReduceOp>
struct DirectIbReceiveReduction {
  template <typename T>
  __device__ __forceinline__ void seedSingleRank(
      ThreadGroup& group,
      T* output,
      const T* ownSource,
      std::size_t rangeBytes,
      ReduceScatterOutputInitialization initialization) const {
    if (initialization == ReduceScatterOutputInitialization::COPY_OWN_INPUT) {
      memcpy_vectorized(
          reinterpret_cast<char*>(output),
          reinterpret_cast<const char*>(ownSource),
          rangeBytes,
          group);
    }
  }

  template <typename Transport>
  __device__ __forceinline__ void receive(
      Transport& transport,
      ThreadGroup& group,
      char* output,
      const char* ownSource,
      std::size_t rangeBytes,
      std::size_t signalingBytes,
      const AbortDevice& abort,
      ReduceScatterOutputInitialization initialization,
      int step) const {
    const char* localInput =
        initialization == ReduceScatterOutputInitialization::COPY_OWN_INPUT &&
            step == 0
        ? ownSource
        : output;
    transport.template recv<ReduceOp>(
        group, output, rangeBytes, signalingBytes, abort, localInput);
  }
};

/**
 * Execute one explicit range for an already assigned blocking Direct-IB role.
 *
 * The caller owns role assignment, group geometry, and any phase barriers. The
 * input contains one logical chunk per IB rank and chunkStrideBytes may include
 * padding. Preconditions: 0 <= ibRank < ibSize, range offset plus length fits
 * in every logical input chunk without overflow, pointers have T alignment,
 * and output is disjoint from the requested input ranges unless it is already
 * initialized. peers is indexed in this phase-local IB rank space, returns a
 * copyable transport, and is never accessed for ibRank itself. group identity
 * and block identity are preserved when selecting and invoking peers.
 */
template <
    typename T,
    typename ReductionPolicy,
    bool kStaggerChannels = true,
    typename PeerAccessor>
__device__ __forceinline__ void direct_ib_reduce_scatter_role_range(
    int ibRank,
    int ibSize,
    DirectIbStridedInput<T> input,
    std::size_t rangeOffsetElements,
    DirectIbOutput<T> output,
    std::size_t rangeElements,
    const ReductionPolicy& reduction,
    std::size_t signalingBytes,
    const PeerAccessor& peers,
    ThreadGroup& group,
    DirectIbReduceScatterRole role,
    const AbortDevice& abort) {
  const std::size_t rangeBytes = rangeElements * sizeof(T);
  if (rangeBytes == 0) {
    return;
  }

  char* outputBytes = reinterpret_cast<char*>(output.data);
  if (role == DirectIbReduceScatterRole::RECEIVE) {
    const T* ownSource = input.chunkData(ibRank) + rangeOffsetElements;
    if (ibSize <= 1) {
      reduction.seedSingleRank(
          group, output.data, ownSource, rangeBytes, output.initialization);
      return;
    }

    const int channel = static_cast<int>(group.group_id);
    for (int step = 0; step < ibSize - 1; ++step) {
      const int peer = direct_ib_reduce_scatter_peer_for_step(
          ibRank,
          ibSize,
          channel,
          step,
          DirectIbReduceScatterRole::RECEIVE,
          kStaggerChannels);
      auto transport = peers[peer];
      reduction.receive(
          transport,
          group,
          outputBytes,
          reinterpret_cast<const char*>(ownSource),
          rangeBytes,
          signalingBytes,
          abort,
          output.initialization,
          step);
    }
    return;
  }

  if (ibSize <= 1) {
    return;
  }
  const int channel = static_cast<int>(group.group_id);
  for (int step = 0; step < ibSize - 1; ++step) {
    const int peer = direct_ib_reduce_scatter_peer_for_step(
        ibRank,
        ibSize,
        channel,
        step,
        DirectIbReduceScatterRole::SEND,
        kStaggerChannels);
    const T* sendData = input.chunkData(peer) + rangeOffsetElements;
    auto transport = peers[peer];
    transport.send(
        group,
        reinterpret_cast<const char*>(sendData),
        rangeBytes,
        signalingBytes,
        abort);
  }
}

} // namespace comms::prims
