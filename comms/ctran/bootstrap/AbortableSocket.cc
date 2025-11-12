// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AbortableSocket.h"

#include <errno.h>
#include <fcntl.h>
#include <folly/SocketAddress.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <folly/ScopeGuard.h>
#include <folly/logging/xlog.h>
#include "comms/ctran/utils/Exception.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::bootstrap {
namespace {
// Helper to wrap compare_exchange_strong for readability
bool tryCloseFd(std::atomic<int>& fd, int& currentFd) {
  return fd.compare_exchange_strong(currentFd, -1);
}

bool shouldRetry(int errcode) {
  return (
      errcode == ENETDOWN || errcode == EPROTO || errcode == ENOPROTOOPT ||
      errcode == EHOSTDOWN || errcode == ENONET || errcode == EHOSTUNREACH ||
      errcode == EOPNOTSUPP || errcode == ENETUNREACH || errcode == EINTR ||
      errcode == ECONNREFUSED || errcode == EINPROGRESS ||
      errcode == ETIMEDOUT);
}

folly::SocketAddress getSocketAddress(int fd) {
  folly::SocketAddress sa;
  sockaddr_storage localAddr;
  socklen_t localLen = sizeof(localAddr);
  if (::getsockname(fd, (struct sockaddr*)&localAddr, &localLen) == 0) {
    sa.setFromSockaddr((struct sockaddr*)&localAddr, localLen);
  } else {
    XLOGF(
        WARN,
        "Failed to get local socket address for fd={}. errno={}, {}",
        fd,
        errno,
        strerror(errno));
  }
  return sa;
}

} // namespace

//
// AbortableSocket Implementation
//

AbortableSocket::AbortableSocket(
    int sockFd,
    folly::SocketAddress peerAddr,
    std::shared_ptr<Abort> abort)
    : fd_(sockFd), abort_(abort), peerAddr_(std::move(peerAddr)) {
  if (fd_ < 0) {
    throw ctran::utils::Exception(
        "Invalid socket file descriptor", commInvalidArgument);
  }

  if (abort_ == nullptr) {
    throw ctran::utils::Exception(
        "Invalid ctran::utils::Abort object", commInvalidArgument);
  }

  prepareSocket();
  localAddr_ = getSocketAddress(fd_);
}

AbortableSocket::AbortableSocket(AbortableSocket&& other) noexcept {
  *this = std::move(other);
}

AbortableSocket& AbortableSocket::operator=(AbortableSocket&& other) noexcept {
  if (this != &other) {
    close();
    fd_.store(other.fd_.load());
    other.fd_.store(-1);
    peerAddr_ = std::move(other.peerAddr_);
    localAddr_ = std::move(other.localAddr_);
    abort_ = std::move(other.abort_);
  }
  return *this;
}

AbortableSocket::~AbortableSocket() {
  close();
}

int AbortableSocket::connect(
    const folly::SocketAddress& addr,
    const std::string& ifName,
    const std::chrono::milliseconds timeout,
    size_t numRetries,
    bool async) {
  // Create socket
  fd_ = ::socket(addr.getFamily(), SOCK_STREAM, 0);
  if (fd_ < 0) {
    throw std::runtime_error("Failed to create socket");
  }
  prepareSocket();

  // Bind the socket to the specified interface name
  if (!ifName.empty()) {
    if (setsockopt(
            fd_, SOL_SOCKET, SO_BINDTODEVICE, ifName.c_str(), ifName.size()) <
        0) {
      throw std::runtime_error("Failed to bind socket to interface " + ifName);
    }
  }

  // Connect to specified address
  sockaddr_storage sockAddr;
  const auto sockLen = addr.getAddress(&sockAddr);
  size_t retryCount{0};
  do {
    XLOGF(DBG, "Connecting to {} via {}", addr.describe(), ifName);
    if (::connect(fd_, (const struct sockaddr*)&sockAddr, sockLen) == 0) {
      break;
    }
    XLOGF(
        WARN,
        "Failed to connect to {} via {}. errno={}, {}",
        addr.describe(),
        ifName,
        errno,
        strerror(errno));

    // Break the loop on non-retryable errors
    if (!shouldRetry(errno)) {
      XLOGF(ERR, "Connection attempt terminating on non-retryable error");
      break;
    }

    // Break the loop if we've exhausted all retries
    if (retryCount >= numRetries) {
      XLOGF(ERR, "Connection attempt terminating as we exhausted all retries");
      close();
      return errno;
    }

    // Retry after a delay
    const auto retryTimeout = retryCount * timeout;
    XLOGF(INFO, "Will retry connecting in {}ms", retryTimeout.count());
    // Wait for a bit before retrying
    retryCount++;
    std::this_thread::sleep_for(retryCount * timeout);
  } while (true);

  peerAddr_ = addr;
  localAddr_ = getSocketAddress(fd_);

  XLOGF(INFO, "Connected to {} via {}, fd={}", addr.describe(), ifName, fd_);
  return 0;
}

void AbortableSocket::prepareSocket() {
  // Set the default socket buffer size to 1MB
  const int bufSize = 1 << 20;
  if (::setsockopt(fd_, SOL_SOCKET, SO_SNDBUF, &bufSize, sizeof(int)) < 0) {
    throw std::runtime_error("Failed to set socket send buffer size");
  }
  if (::setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &bufSize, sizeof(int)) < 0) {
    throw std::runtime_error("Failed to set socket receive buffer size");
  }
  // Set the socket to blocking mode
  int flags = ::fcntl(fd_, F_GETFL, 0);
  if (flags < 0) {
    throw std::runtime_error("Failed to get socket flags");
  }
  flags = flags | O_NONBLOCK;
  if (::fcntl(fd_, F_SETFL, flags) < 0) {
    throw std::runtime_error("Failed to set socket to non-blocking mode");
  }

  // Set TCP_NODELAY to disable Nagle's algorithm, to improve latency
  const int noDelay = 1;
  if (::setsockopt(
          fd_, IPPROTO_TCP, TCP_NODELAY, (char*)&noDelay, sizeof(int)) < 0) {
    throw std::runtime_error("Failed to set TCP_NODELAY");
  }
}

int AbortableSocket::close() {
  int currentFd = fd_.load();

  // Already closed
  if (currentFd < 0) {
    return EALREADY;
  }

  if (!tryCloseFd(fd_, currentFd)) {
    // Another thread beat us to it
    return EALREADY;
  }

  // We successfully set fd_ to -1, now we own the closing operation
  ::shutdown(currentFd, SHUT_RDWR);

  if (::close(currentFd) < 0) {
    return errno;
  }

  return 0;
}

int AbortableSocket::send(const void* buf, const size_t len) {
  size_t totalSent{0};
  do {
    int sent = ::send(fd_, (uint8_t*)buf + totalSent, len - totalSent, 0);
    if (sent == -1 &&
        (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN)) {
      XLOGF(
          ERR,
          "Failed to write to socket fd={}. errno={}, {}",
          fd_,
          errno,
          strerror(errno));
      return errno;
    }
    if (sent > 0) {
      totalSent += sent;
    }
    // Keep looping until we've sent all the data
  } while (totalSent < len);
  return 0;
}

int AbortableSocket::recv(void* buf, const size_t len) {
  size_t totalRecvd{0};
  do {
    int rcvd = ::recv(fd_, (uint8_t*)buf + totalRecvd, len - totalRecvd, 0);
    if (rcvd == -1 &&
        (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN)) {
      XLOGF(
          ERR,
          "Failed to read from socket fd={}. errno={}, {}",
          fd_,
          errno,
          strerror(errno));
      return errno;
    }
    if (rcvd > 0) {
      totalRecvd += rcvd;
    }
    // Keep looping until we've received all the data
  } while (totalRecvd < len);
  return 0;
}

//
// AbortableServerSocket Implementation
//

AbortableServerSocket::~AbortableServerSocket() {
  shutdown();
}

AbortableServerSocket::AbortableServerSocket(
    AbortableServerSocket&& other) noexcept {
  *this = std::move(other);
}

AbortableServerSocket& AbortableServerSocket::operator=(
    AbortableServerSocket&& other) noexcept {
  if (this != &other) {
    shutdown(); // Close the current socket if open
    isV4_ = other.isV4_;
    fd_ = other.fd_;
    other.fd_ = -1; // Reset the file descriptor of the moved-from object
  }
  return *this;
}

int AbortableServerSocket::bind(
    const folly::SocketAddress& addr,
    const std::string& ifName,
    bool reusePort) {
  XLOGF(
      INFO,
      "Binding AbortableServerSocket to {} via {}",
      addr.describe(),
      ifName);
  // Create socket
  fd_ = ::socket(addr.getFamily(), SOCK_STREAM, 0);
  if (fd_ < 0) {
    XLOGF(ERR, "Failed to create socket. errno={}, {}", errno, strerror(errno));
    return errno;
  }

  // Bind the socket to the specified interface name
  if (!ifName.empty()) {
    if (setsockopt(
            fd_, SOL_SOCKET, SO_BINDTODEVICE, ifName.c_str(), ifName.size()) <
        0) {
      XLOGF(
          ERR,
          "Failed to bind socket to interface {}. errno={}, {}",
          ifName.c_str(),
          errno,
          strerror(errno));
      return -1;
    }
  }

  // Set socket options to reuse address and port
  if (addr.getPort() != 0 || reusePort) {
    int reuse = 1;
    if (setsockopt(
            fd_,
            SOL_SOCKET,
            SO_REUSEADDR | SO_REUSEPORT,
            &reuse,
            sizeof(reuse)) < 0) {
      XLOGF(
          ERR,
          "Failed to set SO_REUSEADDR | SO_REUSEPORT. errno={}, {}",
          errno,
          strerror(errno));
      return -1;
    }
  }

  // Bind the socket to the specified address
  isV4_ = addr.getFamily() == AF_INET;
  sockaddr_storage sockAddr;
  const auto sockLen = addr.getAddress(&sockAddr);
  if (::bind(fd_, reinterpret_cast<sockaddr*>(&sockAddr), sockLen) < 0) {
    XLOGF(
        ERR,
        "Failed to bind socket on {}. errno={}, {}",
        addr.describe(),
        errno,
        strerror(errno));
    return errno;
  }

  // allow user to config TOS options for socket
  if (NCCL_SOCKET_TOS_CONFIG != -1) {
    int setSockRet = -1;
    // referenced D77281608
    if (!isV4_) {
      // For IPv6 set the traffic class field
      setSockRet = setsockopt(
          fd_,
          IPPROTO_IPV6,
          IPV6_TCLASS,
          (char*)&NCCL_SOCKET_TOS_CONFIG,
          sizeof(int));
    } else {
      // For IPv4 set the TOS field
      setSockRet = setsockopt(
          fd_, IPPROTO_IP, IP_TOS, (char*)&NCCL_SOCKET_TOS_CONFIG, sizeof(int));
    }
    if (setSockRet < 0) {
      XLOGF(
          ERR,
          "Failed to set socket TOS. errno={}, {}",
          errno,
          strerror(errno));
      return errno;
    }
  }
  XLOGF(
      INFO,
      "AbortableServerSocket is bound on {} via {}, fd={}",
      getListenAddress()->describe(),
      ifName,
      fd_);
  return 0;
}

int AbortableServerSocket::listen() {
  // Listen for incoming connections
  if (::listen(fd_, SOMAXCONN) < 0) {
    XLOGF(
        ERR,
        "Failed to listen on socket. errno={}, {}",
        errno,
        strerror(errno));
    return errno;
  }

  XLOGF(INFO, "AbortableServerSocket Started listening, fd={}", fd_);
  return 0;
}

int AbortableServerSocket::bindAndListen(
    const folly::SocketAddress& addr,
    const std::string& ifName) {
  int retval = bind(addr, ifName);
  if (retval == 0) {
    retval = listen();
  }
  return retval;
}

folly::Expected<folly::SocketAddress, int>
AbortableServerSocket::getListenAddress() {
  sockaddr_storage sockAddr;
  socklen_t sockLen =
      isV4_ ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
  if (getsockname(fd_, (struct sockaddr*)&sockAddr, &sockLen) == -1) {
    XLOGF(
        ERR, "Failed to get socket name. errno={}, {}", errno, strerror(errno));
    return folly::makeUnexpected(errno);
  }
  folly::SocketAddress addr;
  addr.setFromSockaddr((struct sockaddr*)&sockAddr);
  return addr;
}

folly::Expected<std::unique_ptr<ISocket>, int>
AbortableServerSocket::acceptSocket() {
  int retryCnt = 0;
  XCHECK(acceptRetryCnt_ > 0) << "accept retry count must be positive";
  sockaddr_storage sockAddr;
  socklen_t sockLen =
      isV4_ ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
  int clientFd = -1;
  while (retryCnt < acceptRetryCnt_) {
    clientFd = ::accept(fd_, (struct sockaddr*)&sockAddr, &sockLen);
    if (clientFd >= 0) {
      break;
    }
    if (shouldRetry(errno)) {
      /* per accept's man page, for linux sockets, the following errors might
       * be already pending errors and should be considered as EAGAIN and
       * retried
       */
      ++retryCnt;
      XLOGF(
          WARN,
          "Received {} in attempt {}/{}",
          strerror(errno),
          retryCnt,
          acceptRetryCnt_);
      continue;
    }
    if (errno != EAGAIN && errno != EWOULDBLOCK) {
      if (!hasShutDown_) {
        XLOGF(
            ERR,
            "Failed to accept connection. errno={}, {}",
            errno,
            strerror(errno));
      }
      return folly::makeUnexpected(errno);
    } else {
      XLOGF(INFO, "Received {} and will perform a free retry", strerror(errno));
    }
  }
  folly::SocketAddress addr;
  addr.setFromSockaddr((struct sockaddr*)&sockAddr);
  XLOGF(
      INFO,
      "Accepted a new incoming connection {}, fd={}",
      addr.describe(),
      clientFd);
  return std::make_unique<AbortableSocket>(clientFd, addr);
}

int AbortableServerSocket::shutdown() {
  if (!setShuttingDown(false, true) || hasShutDown_) {
    setShuttingDown(true, false);
    return EALREADY;
  }
  // shutdown fd_ would fail accept on the listen thread. To avoid misleading
  // error logging at accept failure, mark intentional shutdown
  if (fd_ >= 0) {
    XLOGF(
        INFO,
        "AbortableServerSocket is shutting down on {}, fd={}",
        getListenAddress()->describe(),
        fd_);
    if (::shutdown(fd_, SHUT_RDWR) < 0 && errno != ENOTCONN) {
      ::close(fd_);
      fd_ = -1;
      hasShutDown_ = true;
      setShuttingDown(true, false);
      return errno;
    }
    ::close(fd_);
    fd_ = -1;
  }
  hasShutDown_ = true;
  setShuttingDown(true, false);
  return 0;
}

} // namespace ctran::bootstrap
