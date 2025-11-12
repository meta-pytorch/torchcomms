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
#include "comms/utils/cvars/nccl_cvars.h"

using namespace std::literals::chrono_literals;

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
  if (!async) {
    XLOG(
        DBG,
        "AbortableSocket::connect() called with async=false; ignoring flag.");
  }
  // Create socket
  fd_.store(::socket(addr.getFamily(), SOCK_STREAM, 0));
  if (fd_.load() < 0) {
    XLOGF(ERR, "Failed to create socket. errno={}, {}", errno, strerror(errno));
    return errno;
  }
  prepareSocket();

  // Bind the socket to the specified interface name
  if (!ifName.empty()) {
    if (setsockopt(
            fd_.load(),
            SOL_SOCKET,
            SO_BINDTODEVICE,
            ifName.c_str(),
            ifName.size()) < 0) {
      XLOGF(
          ERR,
          "Failed to bind socket to interface {}. errno={}, {}",
          ifName,
          errno,
          strerror(errno));
      return errno;
    }
  }

  sockaddr_storage sockAddr;
  const auto sockLen = addr.getAddress(&sockAddr);
  size_t retryCount{0};

  while (!abort_->Test()) {
    XLOGF(DBG, "Connecting to {} via {}", addr.describe(), ifName);
    int ret = ::connect(fd_.load(), (const struct sockaddr*)&sockAddr, sockLen);

    if (ret == 0) {
      break; // Unlikely (connected immediately).
    }

    if (errno == EINPROGRESS && !waitForConnect(-1ms)) {
      break; // Connected successfully.
    }

    XLOGF(
        WARN,
        "Failed to connect to {} via {}. errno={}, {}",
        addr.describe(),
        ifName,
        errno,
        strerror(errno));

    if (!shouldRetry(errno)) {
      XLOGF(ERR, "Connection attempt terminating on non-retryable error");
      int err = errno;
      close();
      return err;
    }

    const auto sleepInterval = std::chrono::milliseconds(
        static_cast<int64_t>(std::min(100.0 * retryCount, 500.0)));
    retryCount++;
    std::this_thread::sleep_for(sleepInterval);
  }

  CHECK_ABORT_RETURN();

  peerAddr_ = addr;
  localAddr_ = getSocketAddress(fd_.load());

  XLOGF(
      INFO,
      "Connected to {} via {}, fd={}",
      addr.describe(),
      ifName,
      fd_.load());
  return 0;
}

void AbortableSocket::prepareSocket() {
  // Set the default socket buffer size to 1MB
  const int bufSize = 1 << 20;
  int fd = fd_.load();
  if (::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufSize, sizeof(int)) < 0) {
    throw ctran::utils::Exception(
        "Failed to set socket send buffer size", commSystemError);
  }
  if (::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufSize, sizeof(int)) < 0) {
    throw ctran::utils::Exception(
        "Failed to set socket receive buffer size", commSystemError);
  }

  int flags = ::fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    throw ctran::utils::Exception(
        "Failed to get socket flags", commSystemError);
  }
  flags |= O_NONBLOCK;
  if (::fcntl(fd, F_SETFL, flags) < 0) {
    throw ctran::utils::Exception(
        "Failed to set socket to non-blocking mode", commSystemError);
  }

  const int noDelay = 1;
  if (::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char*)&noDelay, sizeof(int)) <
      0) {
    throw ctran::utils::Exception("Failed to set TCP_NODELAY", commSystemError);
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
  auto remaining = abort_->HasTimeout()
      ? abort_->TimeRemaining() + std::chrono::milliseconds(1)
      : std::chrono::milliseconds(-1);
  while (totalSent < len && waitForWritable(remaining)) {
    int sent =
        ::send(fd_.load(), (const uint8_t*)buf + totalSent, len - totalSent, 0);
    if (sent == -1 &&
        (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN)) {
      XLOGF(
          ERR,
          "Failed to write to socket fd={}. errno={}, {}",
          fd_.load(),
          errno,
          strerror(errno));
      return errno;
    }
    if (sent > 0) {
      totalSent += sent;
    }
    if (abort_->HasTimeout() > 0) {
      remaining = abort_->TimeRemaining();
    }
  }

  CHECK_ABORT_RETURN();
  if (totalSent < len) {
    return ETIMEDOUT;
  }
  return 0;
}

int AbortableSocket::recv(void* buf, const size_t len) {
  size_t totalRecvd{0};

  auto remaining = abort_->HasTimeout()
      ? (abort_->TimeRemaining() + std::chrono::milliseconds(1))
      : std::chrono::milliseconds(-1);
  while (totalRecvd < len && waitForReadable(remaining)) {
    int rcvd =
        ::recv(fd_.load(), (uint8_t*)buf + totalRecvd, len - totalRecvd, 0);

    if (rcvd == 0) {
      // Other side closed.
      return ECONNRESET;
    }

    if (rcvd == -1 &&
        (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN)) {
      XLOGF(
          ERR,
          "Failed to read from socket fd={}. errno={}, {}",
          fd_.load(),
          errno,
          strerror(errno));
      return errno;
    }
    if (rcvd > 0) {
      totalRecvd += rcvd;
    }
    if (abort_->HasTimeout() > 0) {
      remaining = abort_->TimeRemaining();
    }
  }

  CHECK_ABORT_RETURN();
  if (totalRecvd < len) {
    return ETIMEDOUT;
  }

  return 0;
}

bool AbortableSocket::waitForReadable(const std::chrono::milliseconds timeout) {
  return waitForEvent(POLLIN, timeout);
}

bool AbortableSocket::waitForWritable(const std::chrono::milliseconds timeout) {
  return waitForEvent(POLLOUT, timeout);
}

bool AbortableSocket::waitForEvent(
    short events,
    const std::chrono::milliseconds timeout) {
  int fd = fd_.load();
  if (fd < 0) {
    XLOG(INFO, "fd_ is < 0");
    return false;
  }

  const auto maxPollTimeout = std::chrono::milliseconds(50);
  auto startTime = std::chrono::steady_clock::now();

  while (!abort_->Test()) {
    struct pollfd pfd{
        .fd = fd,
        .events = events,
        .revents = 0,
    };

    std::chrono::milliseconds remaining = maxPollTimeout;
    if (timeout.count() >= 0) {
      auto elapsed = std::chrono::steady_clock::now() - startTime;
      remaining = timeout -
          std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    }

    if (remaining.count() <= 0) {
      return false;
    }

    // Use short poll timeouts to allow checking abort flag frequently
    if (remaining > maxPollTimeout) {
      remaining = maxPollTimeout;
    }

    int ret = ::poll(&pfd, 1, remaining.count());
    if (ret < 0) {
      XLOGF(
          ERR,
          "poll failed on fd={}. errno={}, {}",
          fd,
          errno,
          strerror(errno));
      return false;
    }

    if (ret == 0) {
      // Timeout, continue loop to check abort flag
      continue;
    }

    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      XLOGF(WARN, "poll returned error events: {}", pfd.revents);
      return false;
    }

    if (pfd.revents & events) {
      return true;
    }
  }

  return false;
}

int AbortableSocket::waitForConnect(const std::chrono::milliseconds timeout) {
  if (!waitForWritable(timeout)) {
    return ETIMEDOUT;
  }

  // Check if the connection succeeded
  int error = 0;
  socklen_t len = sizeof(error);
  if (getsockopt(fd_.load(), SOL_SOCKET, SO_ERROR, &error, &len) < 0) {
    XLOGF(
        ERR,
        "Failed to get socket error status. errno={}, {}",
        errno,
        strerror(errno));
    return errno;
  }

  if (error != 0) {
    XLOGF(
        WARN,
        "Connection failed with error: errno={}, {}",
        error,
        strerror(error));
    return error;
  }

  return 0;
}

//
// AbortableServerSocket Implementation
//

AbortableServerSocket::~AbortableServerSocket() {
  shutdown();
}

AbortableServerSocket::AbortableServerSocket(
    AbortableServerSocket&& other) noexcept
    : isV4_(other.isV4_),
      acceptRetryCnt_(other.acceptRetryCnt_),
      fd_(other.fd_),
      abort_(std::move(other.abort_)),
      shuttingDown_(other.shuttingDown_.load()),
      hasShutDown_(other.hasShutDown_.load()) {
  other.fd_ = -1;
}

AbortableServerSocket& AbortableServerSocket::operator=(
    AbortableServerSocket&& other) noexcept {
  if (this != &other) {
    shutdown(); // Close the current socket if open
    isV4_ = other.isV4_;
    acceptRetryCnt_ = other.acceptRetryCnt_;
    fd_ = other.fd_;
    abort_ = std::move(other.abort_);
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

  prepareSocket();
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
AbortableServerSocket::acceptAsync() {
  int retryCnt = 0;
  sockaddr_storage sockAddr;
  socklen_t sockLen =
      isV4_ ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
  int clientFd = ::accept(fd_, (struct sockaddr*)&sockAddr, &sockLen);
  if (clientFd < 0) {
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
      return folly::makeUnexpected(EAGAIN);
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
      XLOGF(
          INFO,
          "Received error \"{}\" and will perform a free retry",
          strerror(errno));
      return folly::makeUnexpected(EAGAIN); // Treat as EAGAIN to retry
    }
  }

  folly::SocketAddress addr;
  addr.setFromSockaddr((struct sockaddr*)&sockAddr);
  XLOGF(
      INFO,
      "Accepted a new incoming connection {}, fd={}",
      addr.describe(),
      clientFd);
  return std::make_unique<AbortableSocket>(clientFd, addr, abort_);
}

folly::Expected<std::unique_ptr<ISocket>, int>
AbortableServerSocket::acceptSocket() {
  while (!abort_->Test()) {
    auto maybeSocket = acceptAsync();
    if (!maybeSocket.hasError() || maybeSocket.error() != EAGAIN) {
      return maybeSocket;
    }

    struct pollfd pfd;
    pfd.fd = fd_;
    pfd.events = POLLIN;
    pfd.revents = 0;

    auto pollTimeout = std::chrono::milliseconds(10);

    int ret = ::poll(&pfd, 1, pollTimeout.count());

    if (ret < 0) {
      if (!hasShutDown_) {
        XLOGF(
            ERR,
            "poll failed on fd={}. errno={}, {}",
            fd_,
            errno,
            strerror(errno));
      }

      return folly::makeUnexpected(errno);
    }

    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      XLOGF(WARN, "poll returned error events: {}", pfd.revents);
      return folly::makeUnexpected(EIO);
    }
  }

  return folly::makeUnexpected(ECONNABORTED);
}

int AbortableServerSocket::shutdown() {
  if (!setShuttingDown(false, true)) {
    setShuttingDown(true, false);
    return EALREADY;
  }

  if (hasShutDown_.load()) {
    return 0;
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
      XLOGF(
          WARN,
          "Failed to cleanly shutdown AbortableServerSocket. errno={}, {}",
          errno,
          strerror(errno));
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

void AbortableServerSocket::prepareSocket() {
  int flags = ::fcntl(fd_, F_GETFL, 0);
  if (flags < 0) {
    throw ctran::utils::Exception(
        "Failed to get socket flags", commSystemError);
  }
  flags |= O_NONBLOCK;
  if (::fcntl(fd_, F_SETFL, flags) < 0) {
    throw ctran::utils::Exception(
        "Failed to set socket to non-blocking mode", commSystemError);
  }
}

std::unique_ptr<ISocket> AbortableSocketFactory::createClientSocket(
    std::shared_ptr<Abort> abort) {
  return std::make_unique<AbortableSocket>(abort);
}

std::unique_ptr<ISocket> AbortableSocketFactory::createClientSocket(
    int sockFd,
    const folly::SocketAddress& peerAddr,
    std::shared_ptr<Abort> abort) {
  return std::make_unique<AbortableSocket>(sockFd, peerAddr, abort);
}

std::unique_ptr<IServerSocket> AbortableSocketFactory::createServerSocket(
    int acceptRetryCnt,
    std::shared_ptr<ctran::utils::Abort> abort) {
  return std::make_unique<AbortableServerSocket>(acceptRetryCnt, abort);
}

} // namespace ctran::bootstrap
