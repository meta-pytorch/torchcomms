// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdlib>

#include "bootstrap.h"
#include "debug.h"
#include "socket.h"

namespace ncclx {

inline ncclResult_t abortBootstrapState(struct bootstrapState* state) {
  if (state == nullptr) {
    return ncclSuccess;
  }

  ncclResult_t result = ncclSuccess;
  const auto recordCloseResult =
      [&result](const char* endpoint, ncclResult_t closeResult) {
        if (closeResult == ncclSuccess || closeResult == ncclInProgress) {
          return;
        }
        WARN("Failed to close bootstrap %s: %d", endpoint, closeResult);
        if (result == ncclSuccess) {
          result = closeResult;
        }
      };

  if (state->ringUsesOobNet) {
    recordCloseResult(
        "OOB-net send endpoint",
        state->net->closeSend(state->ring.net.sendComm));
    recordCloseResult(
        "OOB-net receive endpoint",
        state->net->closeRecv(state->ring.net.recvComm));
    recordCloseResult(
        "OOB-net listener", state->net->closeListen(state->listen.net.comm));
  } else {
    recordCloseResult(
        "ring send socket", ncclSocketClose(&state->ring.socket.send));
    recordCloseResult(
        "ring receive socket", ncclSocketClose(&state->ring.socket.recv));
    recordCloseResult("ring listener", ncclSocketClose(&state->listen.socket));
  }
  recordCloseResult(
      "peer listener", ncclSocketClose(&state->listen.peerSocket));

  while (state->unexpectedConnections != nullptr) {
    struct unexConn* const connection = state->unexpectedConnections;
    state->unexpectedConnections = connection->next;
    std::free(connection);
  }
  std::free(state->peerProxyAddresses);
  std::free(state->peerProxyAddressesUDS);
  std::free(state->peerP2pAddresses);
  std::free(state);
  return result;
}

inline void abortBootstrapAfterRingAllInfoFailure(
    struct ncclComm* comm,
    struct ncclSocket*& proxySocket) {
  if (proxySocket != nullptr) {
    struct ncclSocket* const socket = proxySocket;
    proxySocket = nullptr;
    const ncclResult_t closeResult = ncclSocketClose(socket);
    if (closeResult != ncclSuccess) {
      WARN("Failed to close bootstrap proxy listener: %d", closeResult);
    }
    std::free(socket);
  }

  void* const bootstrap = comm->bootstrap;
  comm->bootstrap = nullptr;
  if (bootstrap == nullptr) {
    return;
  }

  const ncclResult_t abortResult = bootstrapAbort(bootstrap);
  if (abortResult != ncclSuccess) {
    WARN("Failed to abort bootstrap state: %d", abortResult);
  }
}

} // namespace ncclx
