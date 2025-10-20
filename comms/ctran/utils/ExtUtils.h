// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef EXT_UTILS_H
#define EXT_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <folly/SocketAddress.h>
#include <folly/container/F14Map.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <deque>

#include "comms/ctran/utils/Checks.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#include "comms/utils/logger/LogUtils.h"

// Convienent functions to dequeue element from front of deque container
template <typename T>
static inline T dequeFront(std::deque<T>& q) {
  T ret = q.front();
  q.pop_front();
  return ret;
}

template <typename T>
static inline std::unique_ptr<T> dequeFront(std::deque<std::unique_ptr<T>>& q) {
  std::unique_ptr<T> ret = std::move(q.front());
  q.pop_front();
  return ret;
}

class SetCudaDevRAII {
 public:
  SetCudaDevRAII(const int cudaDev) {
    FB_CUDACHECKABORT(cudaGetDevice(&devOld_));
    if (devOld_ == cudaDev) {
      devOld_ = -1;
      return;
    }
    FB_CUDACHECKABORT(cudaSetDevice(cudaDev));
  }

  ~SetCudaDevRAII() {
    if (devOld_ != -1) {
      FB_CUDACHECKABORT(cudaSetDevice(devOld_));
    }
  }

 private:
  int devOld_{-1};
};

// Convienent structure to maintain a C-style linked list
template <typename T>
struct CommonList {
  T* head{nullptr};
  T* tail{nullptr};
  int count{0};

  void enqueue(T* elem) {
    if (tail == nullptr) {
      head = tail = elem;
    } else {
      tail->next = elem;
      tail = elem;
    }
    count++;
  }
};

struct SocketServerAddr {
  int port{-1};
  std::string ipv4{""};
  std::string ipv6{""};
  std::string hostname{""};
  std::string ifName{""}; // The interface name that a server binds to

  std::string toString() const {
    return fmt::format(
        "port: {}, ipv4: {}, ipv6: {}, hostname: {}, ifName: {}",
        port,
        ipv4,
        ipv6,
        hostname,
        ifName);
  }
};

inline folly::SocketAddress toSocketAddress(const SocketServerAddr& addr) {
  if (addr.port < 0) {
    throw std::runtime_error(
        "Invalid port number " + std::to_string(addr.port));
  }
  return folly::SocketAddress(
      addr.ipv6.empty() ? addr.ipv4 : addr.ipv6,
      addr.port,
      false /* allowNameLookup*/);
}

inline folly::SocketAddress toSocketAddress(const sockaddr_storage& addr) {
  folly::SocketAddress sockAddr;
  sockAddr.setFromSockaddr((struct sockaddr*)&addr);
  return sockAddr;
}

// let curStream wait on prevStream
inline commResult_t streamWaitStream(
    cudaStream_t curStream,
    cudaStream_t prevStream,
    cudaEvent_t event) {
  FB_CUDACHECK(cudaEventRecord(event, prevStream));
  FB_CUDACHECK(cudaStreamWaitEvent(curStream, event, 0));
  return commSuccess;
}

#endif
