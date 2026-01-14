// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <fcntl.h>
#include <netdb.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <sys/types.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

#include "meta/tcpstore/Error.h"

using RankType = uint32_t;
using SizeType = uint64_t;

#define SYSCHECK(expr, success_cond)                                       \
  while (true) {                                                           \
    auto __output = (expr);                                                \
    (void)__output;                                                        \
    if (!(success_cond)) {                                                 \
      if (errno == EINTR) {                                                \
        continue;                                                          \
      } else if (errno == EAGAIN || errno == EWOULDBLOCK) {                \
        throw ncclx::tcpstore::detail::NetworkError("Socket Timeout");     \
      } else {                                                             \
        throw ncclx::tcpstore::detail::NetworkError(std::strerror(errno)); \
      }                                                                    \
    } else {                                                               \
      break;                                                               \
    }                                                                      \
  }

#define SYSCHECK_ERR_RETURN_NEG1(expr) SYSCHECK(expr, __output != -1)

namespace ncclx::tcpstore::utils {

template <typename T>
void sendBytes(
    int socket,
    const T* buffer,
    size_t length,
    bool moreData = false) {
  size_t bytesToSend = sizeof(T) * length;
  if (bytesToSend == 0) {
    return;
  }

  auto bytes = reinterpret_cast<const uint8_t*>(buffer);
  uint8_t* currentBytes = const_cast<uint8_t*>(bytes);

  int flags = 0;

#ifdef MSG_MORE
  if (moreData) { // there is more data to send
    flags |= MSG_MORE;
  }
#endif

// Ignore SIGPIPE as the send() return value is always checked for error
#ifdef MSG_NOSIGNAL
  flags |= MSG_NOSIGNAL;
#endif

  while (bytesToSend > 0) {
    ssize_t bytesSent;
    SYSCHECK_ERR_RETURN_NEG1(
        bytesSent =
            ::send(socket, (const char*)currentBytes, bytesToSend, flags))
    if (bytesSent == 0) {
      throw ncclx::tcpstore::detail::NetworkError(
          "failed to send, sent 0 bytes");
    }

    bytesToSend -= bytesSent;
    currentBytes += bytesSent;
  }
}

template <typename T>
void recvBytes(int socket, T* buffer, size_t length) {
  size_t bytesToReceive = sizeof(T) * length;
  if (bytesToReceive == 0) {
    return;
  }

  auto bytes = reinterpret_cast<uint8_t*>(buffer);
  uint8_t* currentBytes = bytes;

  while (bytesToReceive > 0) {
    ssize_t bytesReceived;
    SYSCHECK_ERR_RETURN_NEG1(
        bytesReceived = recv(socket, (char*)currentBytes, bytesToReceive, 0))
    if (bytesReceived == 0) {
      throw ncclx::tcpstore::detail::NetworkError(
          "failed to recv, got 0 bytes");
    }

    bytesToReceive -= bytesReceived;
    currentBytes += bytesReceived;
  }
}

// send a vector's length and data
template <typename T>
void sendVector(int socket, const std::vector<T>& vec, bool moreData = false) {
  SizeType size = vec.size();
  sendBytes<SizeType>(socket, &size, 1, true);
  sendBytes<T>(socket, vec.data(), size, moreData);
}

// receive a vector as sent in sendVector
template <typename T>
std::vector<T> recvVector(int socket) {
  SizeType valueSize;
  recvBytes<SizeType>(socket, &valueSize, 1);
  std::vector<T> value(valueSize);
  recvBytes<T>(socket, value.data(), value.size());
  return value;
}

// this is only for convenience when sending rvalues
template <typename T>
void sendValue(int socket, const T& value, bool moreData = false) {
  sendBytes<T>(socket, &value, 1, moreData);
}

template <typename T>
T recvValue(int socket) {
  T value;
  recvBytes<T>(socket, &value, 1);
  return value;
}

} // namespace ncclx::tcpstore::utils
