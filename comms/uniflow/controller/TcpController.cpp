// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/controller/TcpController.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cassert>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <stdexcept>
#include <system_error>
#include <thread>

#include <fmt/core.h>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/executor/EventBase.h"
#include "comms/uniflow/logging/Logger.h"

namespace uniflow::controller {

TcpSocketConfig TcpSocketConfig::osDefaults() {
  TcpSocketConfig cfg;
  cfg.connTimeout = std::nullopt;
  cfg.socketBufSize = std::nullopt;
  cfg.tcpNoDelay = std::nullopt;
  cfg.enableKeepalive = std::nullopt;
  cfg.keepaliveIdle = std::nullopt;
  cfg.keepaliveInterval = std::nullopt;
  cfg.keepaliveCount = std::nullopt;
  cfg.userTimeout = std::nullopt;
  return cfg;
}

Status TcpSocketConfig::validate() const {
  if (connTimeout && connTimeout->count() <= 0) {
    return Err(ErrCode::InvalidArgument, "connTimeout must be positive");
  }
  if (socketBufSize && *socketBufSize <= 0) {
    return Err(ErrCode::InvalidArgument, "socketBufSize must be positive");
  }
  if (keepaliveIdle && keepaliveIdle->count() <= 0) {
    return Err(ErrCode::InvalidArgument, "keepaliveIdle must be positive");
  }
  if (keepaliveInterval && keepaliveInterval->count() <= 0) {
    return Err(ErrCode::InvalidArgument, "keepaliveInterval must be positive");
  }
  if (keepaliveCount && *keepaliveCount <= 0) {
    return Err(ErrCode::InvalidArgument, "keepaliveCount must be positive");
  }
  if (userTimeout && userTimeout->count() <= 0) {
    return Err(ErrCode::InvalidArgument, "userTimeout must be positive");
  }
  if (acceptRetryCnt <= 0) {
    return Err(ErrCode::InvalidArgument, "acceptRetryCnt must be positive");
  }
  if (retryTimeout.count() < 0) {
    return Err(ErrCode::InvalidArgument, "retryTimeout must be non-negative");
  }
  return Ok();
}

namespace {

constexpr uint32_t kMaxMessageSize = 64 << 20;
constexpr int kSocketBufSize = 1 << 20;
constexpr int kAcceptTimeoutSec = 5;
constexpr int kConnectedTimeoutSec = 30;
constexpr int kKeepaliveIdleSec = 60;
constexpr int kKeepaliveIntervalSec = 5;
constexpr int kKeepaliveCount = 3;
constexpr int kUserTimeoutMs = 60000;

// Magic value exchanged during connection handshake to validate that both
// endpoints are uniflow controllers (rejects stray connections).
constexpr uint32_t kMagic = 0x554E4946; // "UNIF" in ASCII

// Accumulates the names of all failed setsockopt calls and produces a single
// Status at the end.
class SockOptSetter {
  int sock_;
  std::string failures_;

 public:
  explicit SockOptSetter(int sock) : sock_(sock) {}

  template <typename T>
  void set(int level, int optname, const T& value, const char* name) {
    if (::setsockopt(sock_, level, optname, &value, sizeof(value)) < 0) {
      if (!failures_.empty()) {
        failures_ += ", ";
      }
      failures_ += name;
    }
  }

  // Best-effort — failure silently ignored.
  template <typename T>
  void trySet(int level, int optname, const T& value) {
    ::setsockopt(sock_, level, optname, &value, sizeof(value));
  }

  Status status() const {
    if (!failures_.empty()) {
      return Err(ErrCode::ConnectionFailed, "setsockopt failed: " + failures_);
    }
    return Ok();
  }
};

// Aligned with ctran/bootstrap/Socket.cc::shouldRetry().
bool shouldRetry(int errcode) {
  return (
      errcode == ENETDOWN || errcode == EPROTO || errcode == ENOPROTOOPT ||
      errcode == EHOSTDOWN || errcode == ENONET || errcode == EHOSTUNREACH ||
      errcode == EOPNOTSUPP || errcode == ENETUNREACH || errcode == EINTR ||
      errcode == ECONNREFUSED || errcode == EINPROGRESS ||
      errcode == ETIMEDOUT);
}

Result<std::pair<std::string, int>> parseHostPort(std::string_view id) {
  auto colonPos = id.rfind(':');
  if (colonPos == std::string_view::npos) {
    return Err(ErrCode::InvalidArgument, "Missing ':' in address");
  }

  auto host = std::string(id.substr(0, colonPos));
  auto portStr = id.substr(colonPos + 1);

  if (portStr.empty()) {
    return Err(ErrCode::InvalidArgument, "Empty port in address");
  }

  int port = 0;
  auto [ptr, ec] =
      std::from_chars(portStr.data(), portStr.data() + portStr.size(), port);
  if (ec != std::errc{} || ptr != portStr.data() + portStr.size()) {
    return Err(
        ErrCode::InvalidArgument, "Invalid port: " + std::string(portStr));
  }

  if (port < 0 || port > 65535) {
    return Err(
        ErrCode::InvalidArgument, "Port out of range: " + std::string(portStr));
  }
  return std::pair{host, port};
}

Result<int> detectAddressFamily(std::string_view host) {
  // Wildcard → IPv6 dual-stack (accepts both v4 and v6 on Linux).
  if (host.empty() || host == "*") {
    return AF_INET6;
  }

  std::string hostStr(host);

  in6_addr addr6{};
  if (::inet_pton(AF_INET6, hostStr.c_str(), &addr6) == 1) {
    return AF_INET6;
  }

  in_addr addr4{};
  if (::inet_pton(AF_INET, hostStr.c_str(), &addr4) == 1) {
    return AF_INET;
  }

  return Err(ErrCode::InvalidArgument, "Invalid host address: " + hostStr);
}

Result<socklen_t> buildSockAddr(
    std::string_view host,
    int port,
    int domain,
    sockaddr_storage& addr) {
  std::memset(&addr, 0, sizeof(addr));

  if (domain == AF_INET6) {
    auto* sa = reinterpret_cast<sockaddr_in6*>(&addr);
    sa->sin6_family = AF_INET6;
    sa->sin6_port = htons(static_cast<uint16_t>(port));

    if (host.empty() || host == "::" || host == "*") {
      sa->sin6_addr = in6addr_any;
    } else if (
        ::inet_pton(AF_INET6, std::string(host).c_str(), &sa->sin6_addr) <= 0) {
      return Err(
          ErrCode::InvalidArgument,
          "Invalid IPv6 address: " + std::string(host));
    }
    return static_cast<socklen_t>(sizeof(sockaddr_in6));
  }

  auto* sa = reinterpret_cast<sockaddr_in*>(&addr);
  sa->sin_family = AF_INET;
  sa->sin_port = htons(static_cast<uint16_t>(port));

  if (host.empty() || host == "0.0.0.0" || host == "*") {
    sa->sin_addr.s_addr = INADDR_ANY;
  } else if (
      ::inet_pton(AF_INET, std::string(host).c_str(), &sa->sin_addr) <= 0) {
    return Err(
        ErrCode::InvalidArgument, "Invalid IPv4 address: " + std::string(host));
  }
  return static_cast<socklen_t>(sizeof(sockaddr_in));
}

std::string formatAddr(const sockaddr_storage& addr) {
  char buf[INET6_ADDRSTRLEN] = {};
  if (addr.ss_family == AF_INET6) {
    auto* sa = reinterpret_cast<const sockaddr_in6*>(&addr);
    ::inet_ntop(AF_INET6, &sa->sin6_addr, buf, sizeof(buf));
    return std::string("[") + buf + "]:" + std::to_string(ntohs(sa->sin6_port));
  }
  auto* sa = reinterpret_cast<const sockaddr_in*>(&addr);
  ::inet_ntop(AF_INET, &sa->sin_addr, buf, sizeof(buf));
  return std::string(buf) + ":" + std::to_string(ntohs(sa->sin_port));
}

Result<int> createListenSocket(int domain) {
  int sock = ::socket(domain, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (sock < 0) {
    return Err(
        ErrCode::ConnectionFailed,
        "socket creation failed: " + std::system_category().message(errno));
  }

  SockOptSetter opt(sock);
  opt.set(SOL_SOCKET, SO_REUSEADDR, 1, "SO_REUSEADDR");
  opt.trySet(SOL_SOCKET, SO_REUSEPORT, 1);

  // Periodic wakeup for shutdown checks during blocking accept()
  struct timeval tv{};
  tv.tv_sec = kAcceptTimeoutSec;
  opt.set(SOL_SOCKET, SO_RCVTIMEO, tv, "SO_RCVTIMEO");

  auto status = opt.status();
  if (!status) {
    ::close(sock);
    return std::move(status).error();
  }
  // @lint-ignore PULSE_RESOURCE_LEAK fd ownership transfers to caller via
  // Result
  return sock;
}

Status configureAcceptedSocket(int sock) {
  SockOptSetter opt(sock);
  opt.set(SOL_SOCKET, SO_SNDBUF, kSocketBufSize, "SO_SNDBUF");
  opt.set(SOL_SOCKET, SO_RCVBUF, kSocketBufSize, "SO_RCVBUF");
  opt.set(IPPROTO_TCP, TCP_NODELAY, 1, "TCP_NODELAY");
  opt.set(SOL_SOCKET, SO_KEEPALIVE, 1, "SO_KEEPALIVE");
  opt.set(IPPROTO_TCP, TCP_KEEPIDLE, kKeepaliveIdleSec, "TCP_KEEPIDLE");
  opt.set(IPPROTO_TCP, TCP_KEEPINTVL, kKeepaliveIntervalSec, "TCP_KEEPINTVL");
  opt.set(IPPROTO_TCP, TCP_KEEPCNT, kKeepaliveCount, "TCP_KEEPCNT");
  opt.set(IPPROTO_TCP, TCP_USER_TIMEOUT, kUserTimeoutMs, "TCP_USER_TIMEOUT");

  struct timeval tv{};
  tv.tv_sec = kConnectedTimeoutSec;
  opt.set(SOL_SOCKET, SO_SNDTIMEO, tv, "SO_SNDTIMEO");
  opt.set(SOL_SOCKET, SO_RCVTIMEO, tv, "SO_RCVTIMEO");

  return opt.status();
}

Status
resolveAndBind(const std::string& host, int port, int domain, int listenFd) {
  sockaddr_storage addr{};
  auto addrLenResult = buildSockAddr(host, port, domain, addr);
  if (!addrLenResult) {
    return std::move(addrLenResult).error();
  }

  if (::bind(
          listenFd, reinterpret_cast<sockaddr*>(&addr), addrLenResult.value()) <
      0) {
    return Err(
        ErrCode::ConnectionFailed,
        "bind failed: " + std::system_category().message(errno));
  }

  return Ok();
}

// 500ms bound prevents the 8-byte magic exchange from blocking the loop
// thread indefinitely. Caller must close the socket if this fails.
Status setHandshakeTimeout(int sock) {
  constexpr int kHandshakeTimeoutMs = 500;
  struct timeval tv{};
  tv.tv_sec = 0;
  tv.tv_usec = kHandshakeTimeoutMs * 1000;
  SockOptSetter opt(sock);
  opt.set(SOL_SOCKET, SO_SNDTIMEO, tv, "SO_SNDTIMEO");
  opt.set(SOL_SOCKET, SO_RCVTIMEO, tv, "SO_RCVTIMEO");
  return opt.status();
}

} // namespace

bool TcpConn::sendAll(const void* buf, size_t len) {
  auto* ptr = static_cast<const uint8_t*>(buf);
  size_t remaining = len;
  while (remaining > 0) {
    ssize_t n = ::send(sock_, ptr, remaining, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      int savedErrno = errno;
      UNIFLOW_LOG_ERROR(
          "sendAll failed: fd={} errno={} ({})",
          sock_,
          savedErrno,
          std::system_category().message(savedErrno));
      errno = savedErrno;
      return false;
    }
    ptr += n;
    remaining -= static_cast<size_t>(n);
  }
  return true;
}

bool TcpConn::recvAll(void* buf, size_t len) {
  auto* ptr = static_cast<uint8_t*>(buf);
  size_t remaining = len;
  while (remaining > 0) {
    ssize_t n = ::recv(sock_, ptr, remaining, 0);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      int savedErrno = errno;
      UNIFLOW_LOG_ERROR(
          "recvAll failed: fd={} errno={} ({})",
          sock_,
          savedErrno,
          std::system_category().message(savedErrno));
      errno = savedErrno;
      return false;
    }
    if (n == 0) {
      UNIFLOW_LOG_WARN("recvAll: peer closed connection, fd={}", sock_);
      errno = ECONNRESET;
      return false;
    }
    ptr += n;
    remaining -= static_cast<size_t>(n);
  }
  return true;
}

bool TcpConn::exchangeMagic() {
  uint32_t magic = htonl(kMagic);
  if (!sendAll(&magic, sizeof(magic))) {
    UNIFLOW_LOG_ERROR("magic exchange: send failed, fd={}", sock_);
    return false;
  }

  uint32_t peerMagic = 0;
  if (!recvAll(&peerMagic, sizeof(peerMagic))) {
    UNIFLOW_LOG_ERROR("magic exchange: recv failed, fd={}", sock_);
    return false;
  }

  if (ntohl(peerMagic) != kMagic) {
    UNIFLOW_LOG_ERROR(
        "magic exchange: mismatch fd={} expected={:#x} got={:#x}",
        sock_,
        kMagic,
        ntohl(peerMagic));
    return false;
  }
  return true;
}

std::unique_ptr<TcpConn> TcpConn::create(int sock) {
  UNIFLOW_LOG_DEBUG("TcpConn: handshake starting, fd={}", sock);
  auto conn = std::unique_ptr<TcpConn>(new TcpConn(sock));
  if (!conn->exchangeMagic()) {
    UNIFLOW_LOG_ERROR("TcpConn: handshake failed, fd={}", sock);
    return nullptr;
  }
  UNIFLOW_LOG_INFO("TcpConn: established, fd={}", sock);
  return conn;
}

Result<size_t> TcpConn::send(std::span<const uint8_t> data) {
  if (sock_ < 0) {
    return Err(ErrCode::NotConnected, "Socket is not connected");
  }

  if (data.size() > kMaxMessageSize) {
    return Err(
        ErrCode::InvalidArgument,
        "message size " + std::to_string(data.size()) + " exceeds maximum " +
            std::to_string(kMaxMessageSize));
  }

  UNIFLOW_LOG_DEBUG("TcpConn::send: fd={} bytes={}", sock_, data.size());

  // Length-prefixed framing: 4-byte network-order size header + payload
  uint32_t len = htonl(static_cast<uint32_t>(data.size()));
  if (!sendAll(&len, sizeof(len))) {
    return Err(
        ErrCode::ConnectionFailed,
        "send header failed: " + std::system_category().message(errno));
  }

  if (!data.empty() && !sendAll(data.data(), data.size())) {
    return Err(
        ErrCode::ConnectionFailed,
        "send payload failed: " + std::system_category().message(errno));
  }

  return data.size();
}

Result<size_t> TcpConn::recv(std::vector<uint8_t>& data) {
  if (sock_ < 0) {
    return Err(ErrCode::NotConnected, "Socket is not connected");
  }

  uint32_t rawLen = 0;
  if (!recvAll(&rawLen, sizeof(rawLen))) {
    return Err(
        ErrCode::ConnectionFailed,
        "recv header failed: " + std::system_category().message(errno));
  }

  uint32_t len = ntohl(rawLen);
  if (len > kMaxMessageSize) {
    return Err(
        ErrCode::InvalidArgument,
        "message size " + std::to_string(len) + " exceeds maximum " +
            std::to_string(kMaxMessageSize));
  }

  data.resize(len);
  if (len != 0 && !recvAll(data.data(), len)) {
    return Err(
        ErrCode::ConnectionFailed,
        "recv payload failed: " + std::system_category().message(errno));
  }

  UNIFLOW_LOG_DEBUG("TcpConn::recv: fd={} bytes={}", sock_, len);
  return len;
}

void TcpConn::close() {
  if (sock_ >= 0) {
    UNIFLOW_LOG_DEBUG("TcpConn: close, fd={}", sock_);
    ::shutdown(sock_, SHUT_RDWR);
    ::close(sock_);
    sock_ = -1;
  }
}

TcpConn::~TcpConn() {
  close();
}

// ---------------------------------------------------------------------------
// SyncAccept
// ---------------------------------------------------------------------------

std::future<std::unique_ptr<Conn>> SyncAccept::accept(
    std::atomic<int>& listenSock,
    int acceptRetryCnt,
    const std::string& id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (listenSock.load() < 0) {
    return make_ready_future(std::unique_ptr<Conn>(nullptr));
  }

  UNIFLOW_LOG_DEBUG("TcpServer: waiting for connection on {}", id);

  sockaddr_storage clientAddr{};

  // EAGAIN from SO_RCVTIMEO loops back for shutdown checks without
  // counting as a retry. Transient errors retry up to acceptRetryCnt.
  int retryCnt = 0;
  while (retryCnt < acceptRetryCnt) {
    socklen_t clientLen = sizeof(clientAddr);
    int clientSock = ::accept4(
        listenSock.load(),
        reinterpret_cast<sockaddr*>(&clientAddr),
        &clientLen,
        SOCK_CLOEXEC);
    if (clientSock >= 0) {
      UNIFLOW_LOG_INFO(
          "TcpServer: accepted fd={} from {}",
          clientSock,
          formatAddr(clientAddr));
      auto status = configureAcceptedSocket(clientSock);
      if (!status) {
        UNIFLOW_LOG_ERROR(
            "TcpServer: socket config failed fd={}: {}",
            clientSock,
            status.error().toString());
        ::close(clientSock);
        return make_ready_future(std::unique_ptr<Conn>(nullptr));
      }
      auto conn = TcpConn::create(clientSock);
      if (conn) {
        return make_ready_future(std::unique_ptr<Conn>(std::move(conn)));
      }
      // Handshake failed (non-uniflow client) — keep accepting.
      // Socket was already closed by TcpConn destructor inside create().
      UNIFLOW_LOG_WARN(
          "TcpServer: rejecting non-uniflow client, fd={}", clientSock);
      if (listenSock.load() < 0) {
        UNIFLOW_LOG_INFO("TcpServer: accept interrupted by shutdown");
        return make_ready_future(std::unique_ptr<Conn>(nullptr));
      }
      continue;
    }

    int savedErrno = errno;
    if (savedErrno == EAGAIN || savedErrno == EWOULDBLOCK) {
      if (listenSock.load() < 0) {
        UNIFLOW_LOG_INFO("TcpServer: accept interrupted by shutdown");
        return make_ready_future(std::unique_ptr<Conn>(nullptr));
      }
      continue; // don't count timeouts as retries
    }

    if (!shouldRetry(savedErrno)) {
      UNIFLOW_LOG_ERROR(
          "TcpServer: accept failed (non-retryable): errno={} ({})",
          savedErrno,
          std::system_category().message(savedErrno));
      return make_ready_future(std::unique_ptr<Conn>(nullptr));
    }

    ++retryCnt;
    UNIFLOW_LOG_WARN(
        "TcpServer: accept retry {}/{}: errno={} ({})",
        retryCnt,
        acceptRetryCnt,
        savedErrno,
        std::system_category().message(savedErrno));
  }

  UNIFLOW_LOG_ERROR(
      "TcpServer: accept exhausted {} retries on {}", acceptRetryCnt, id);
  return make_ready_future(std::unique_ptr<Conn>(nullptr));
}

void SyncAccept::shutdown(std::atomic<int>& listenSock, const std::string& id) {
  // SHUT_RDWR unblocks any thread blocked in accept(). The mutex
  // ensures we wait for accept() to return before closing the fd,
  // preventing use-after-free on listenSock after destruction.
  int fd = listenSock.exchange(-1);
  if (fd >= 0) {
    UNIFLOW_LOG_INFO("TcpServer: shutting down {}, fd={}", id, fd);
    ::shutdown(fd, SHUT_RDWR);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (fd >= 0) {
    ::close(fd);
  }
}

// ---------------------------------------------------------------------------
// AsyncAccept
// ---------------------------------------------------------------------------

void AsyncAccept::teardown(int fd) {
  accepting_ = false;
  evb_.unregisterFd(fd);
  // Deliver any pre-accepted connections to waiting consumers.
  while (!readyConns_.empty() && !pendingPromises_.empty()) {
    pendingPromises_.front().set_value(std::move(readyConns_.front()));
    readyConns_.pop();
    pendingPromises_.pop();
  }
  while (!pendingPromises_.empty()) {
    pendingPromises_.front().set_value(nullptr);
    pendingPromises_.pop();
  }
  readyConns_ = {};
}

void AsyncAccept::acceptPendingConnections(std::atomic<int>& listenSock) {
  if (!accepting_) {
    return;
  }

  while (!readyConns_.empty() && !pendingPromises_.empty()) {
    pendingPromises_.front().set_value(std::move(readyConns_.front()));
    readyConns_.pop();
    pendingPromises_.pop();
  }

  int listenFd = listenSock.load();
  if (listenFd < 0) {
    return;
  }

  // Drain all pending connections from the non-blocking listen socket.
  for (;;) {
    sockaddr_storage addr{};
    socklen_t len = sizeof(addr);
    int clientSock = accept4(
        listenFd, reinterpret_cast<sockaddr*>(&addr), &len, SOCK_CLOEXEC);
    if (clientSock < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        break;
      }
      if (errno == EINTR) {
        continue;
      }
      UNIFLOW_LOG_ERROR(
          "TcpServer: async accept4 failed: errno={} ({})",
          errno,
          std::system_category().message(errno));
      break;
    }

    UNIFLOW_LOG_INFO(
        "TcpServer: async accepted fd={} from {}",
        clientSock,
        formatAddr(addr));

    auto timeoutStatus = setHandshakeTimeout(clientSock);
    if (!timeoutStatus) {
      UNIFLOW_LOG_WARN(
          "TcpServer: handshake timeout failed, fd={}: {}",
          clientSock,
          timeoutStatus.error().toString());
      ::close(clientSock);
      continue;
    }

    auto conn = TcpConn::create(clientSock);
    if (!conn) {
      continue;
    }

    // configureAcceptedSocket sets 30s timeouts — must come after the
    // 500ms handshake timeout to avoid overriding it.
    auto status = configureAcceptedSocket(conn->getFd());
    if (!status) {
      UNIFLOW_LOG_ERROR(
          "TcpServer: async socket config failed fd={}: {}",
          conn->getFd(),
          status.error().toString());
      continue;
    }

    if (!pendingPromises_.empty()) {
      pendingPromises_.front().set_value(std::move(conn));
      pendingPromises_.pop();
    } else {
      readyConns_.push(std::move(conn));
    }
  }
}

std::future<std::unique_ptr<Conn>> AsyncAccept::accept(
    std::atomic<int>& listenSock,
    int /*acceptRetryCnt*/,
    const std::string& /*id*/) {
  if (listenSock.load() < 0) {
    return make_ready_future(std::unique_ptr<Conn>(nullptr));
  }

  std::promise<std::unique_ptr<Conn>> promise;
  auto future = promise.get_future();

  evb_.dispatch([this, &listenSock, p = std::move(promise)]() mutable noexcept {
    int sock = listenSock.load();
    if (sock < 0) {
      p.set_value(nullptr);
      return;
    }

    // Lazy setup: fcntl + registerFd both on the loop thread.
    if (!accepting_) {
      int flags = fcntl(sock, F_GETFL);
      if (flags < 0 || fcntl(sock, F_SETFL, flags | O_NONBLOCK) < 0) {
        UNIFLOW_LOG_ERROR(
            "TcpServer: fcntl O_NONBLOCK failed, fd={}: {}",
            sock,
            std::system_category().message(errno));
        p.set_value(nullptr);
        return;
      }
      evb_.registerFd(sock, EPOLLIN, [this, &listenSock](uint32_t /*events*/) {
        acceptPendingConnections(listenSock);
      });
      accepting_ = true;
    }

    pendingPromises_.push(std::move(p));

    // Deliver any pre-accepted connections waiting in readyConns_.
    while (!readyConns_.empty() && !pendingPromises_.empty()) {
      pendingPromises_.front().set_value(std::move(readyConns_.front()));
      readyConns_.pop();
      pendingPromises_.pop();
    }
  });

  return future;
}

void AsyncAccept::shutdown(
    std::atomic<int>& listenSock,
    const std::string& id) {
  int fd = listenSock.exchange(-1);
  if (fd < 0) {
    return;
  }

  auto closeFd = [fd, &id]() {
    UNIFLOW_LOG_INFO("TcpServer: shutting down {}, fd={}", id, fd);
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
  };

  if (evb_.inLoopThread()) {
    teardown(fd);
    closeFd();
  } else if (evb_.isLoopRunning()) {
    evb_.dispatchAndWait([this, fd]() noexcept { teardown(fd); });
    // Drain: unregisterFd inside teardown is deferred —
    // wait for it to complete before closing the fd.
    evb_.dispatchAndWait([]() noexcept {});
    closeFd();
  } else {
    // Loop stopped — closing the fd auto-removes it from epoll.
    teardown(fd);
    closeFd();
  }
}

// ---------------------------------------------------------------------------
// BasicTcpServer<AcceptPolicy> — template methods
// ---------------------------------------------------------------------------

template <typename AcceptPolicy>
void BasicTcpServer<AcceptPolicy>::parseId() {
  auto result = parseHostPort(id_);
  if (!result) {
    throw std::invalid_argument(
        "Invalid address format: " + result.error().toString());
  }
  auto [host, port] = result.value();
  host_ = host;
  port_ = port;

  auto status = config_.validate();
  if (!status) {
    throw std::invalid_argument(
        "Invalid socket config: " + status.error().toString());
  }
}

template <typename AcceptPolicy>
Status BasicTcpServer<AcceptPolicy>::init() {
  if (listenSock_ >= 0) {
    return Err(ErrCode::InvalidArgument, "Server already initialized");
  }

  UNIFLOW_LOG_INFO("TcpServer: initializing on {}", id_);

  auto domainResult = detectAddressFamily(host_);
  if (!domainResult) {
    return std::move(domainResult).error();
  }
  int domain = domainResult.value();

  auto sockResult = createListenSocket(domain);
  if (!sockResult) {
    UNIFLOW_LOG_ERROR("TcpServer: socket creation failed on {}", id_);
    return std::move(sockResult).error();
  }
  listenSock_ = sockResult.value();

  auto bindStatus = resolveAndBind(host_, port_, domain, listenSock_.load());
  if (!bindStatus) {
    UNIFLOW_LOG_ERROR(
        "TcpServer: bind failed on {}: {}", id_, bindStatus.error().toString());
    ::close(listenSock_);
    listenSock_ = -1;
    return bindStatus;
  }

  sockaddr_storage boundAddr{};
  socklen_t boundLen = sizeof(boundAddr);
  if (::getsockname(
          listenSock_, reinterpret_cast<sockaddr*>(&boundAddr), &boundLen) ==
      0) {
    if (domain == AF_INET6) {
      port_ = ntohs(reinterpret_cast<sockaddr_in6*>(&boundAddr)->sin6_port);
    } else {
      port_ = ntohs(reinterpret_cast<sockaddr_in*>(&boundAddr)->sin_port);
    }
  }

  if (::listen(listenSock_, SOMAXCONN) < 0) {
    int savedErrno = errno;
    UNIFLOW_LOG_ERROR(
        "TcpServer: listen failed on {}: {}",
        id_,
        std::system_category().message(savedErrno));
    ::close(listenSock_);
    listenSock_ = -1;
    return Err(
        ErrCode::ConnectionFailed,
        "listen failed: " + std::system_category().message(savedErrno));
  }

  // Resolve wildcard host to connectable loopback address so that getId()
  // returns a directly connectable "ip:port" string.
  if (host_.empty() || host_ == "*" || host_ == "0.0.0.0" || host_ == "::") {
    host_ = "127.0.0.1";
  }

  id_ = fmt::format("{}:{}", host_, port_);
  UNIFLOW_LOG_INFO("TcpServer: listening on {} fd={}", id_, listenSock_.load());
  return Ok();
}

template class BasicTcpServer<SyncAccept>;
template class BasicTcpServer<AsyncAccept>;

// ---------------------------------------------------------------------------
// Shared client helpers
// ---------------------------------------------------------------------------

namespace {

Status configureClientSocket(int sock, const TcpSocketConfig& config) {
  SockOptSetter opt(sock);

  if (config.socketBufSize) {
    opt.set(SOL_SOCKET, SO_SNDBUF, *config.socketBufSize, "SO_SNDBUF");
    opt.set(SOL_SOCKET, SO_RCVBUF, *config.socketBufSize, "SO_RCVBUF");
  }
  if (config.tcpNoDelay) {
    int val = *config.tcpNoDelay ? 1 : 0;
    opt.set(IPPROTO_TCP, TCP_NODELAY, val, "TCP_NODELAY");
  }
  if (config.enableKeepalive) {
    int val = *config.enableKeepalive ? 1 : 0;
    opt.set(SOL_SOCKET, SO_KEEPALIVE, val, "SO_KEEPALIVE");
  }
  if (config.enableKeepalive && *config.enableKeepalive) {
    if (config.keepaliveIdle) {
      int val = static_cast<int>(config.keepaliveIdle->count());
      opt.set(IPPROTO_TCP, TCP_KEEPIDLE, val, "TCP_KEEPIDLE");
    }
    if (config.keepaliveInterval) {
      int val = static_cast<int>(config.keepaliveInterval->count());
      opt.set(IPPROTO_TCP, TCP_KEEPINTVL, val, "TCP_KEEPINTVL");
    }
    if (config.keepaliveCount) {
      opt.set(IPPROTO_TCP, TCP_KEEPCNT, *config.keepaliveCount, "TCP_KEEPCNT");
    }
  }
  if (config.userTimeout) {
    int val = static_cast<int>(config.userTimeout->count());
    opt.set(IPPROTO_TCP, TCP_USER_TIMEOUT, val, "TCP_USER_TIMEOUT");
  }
  if (config.connTimeout) {
    struct timeval tv{};
    tv.tv_sec = config.connTimeout->count();
    opt.set(SOL_SOCKET, SO_SNDTIMEO, tv, "SO_SNDTIMEO");
    opt.set(SOL_SOCKET, SO_RCVTIMEO, tv, "SO_RCVTIMEO");
  }

  return opt.status();
}

struct ResolvedAddr {
  int domain;
  sockaddr_storage addr;
  socklen_t addrLen;
};

Result<ResolvedAddr> resolveConnectAddr(const std::string& id) {
  auto result = parseHostPort(id);
  if (!result) {
    return Err(ErrCode::InvalidArgument, "Invalid address: " + id);
  }
  auto [host, port] = std::move(result).value();

  auto domainResult = detectAddressFamily(host);
  if (!domainResult) {
    return Err(ErrCode::InvalidArgument, "Invalid host: " + host);
  }

  ResolvedAddr resolved{};
  resolved.domain = domainResult.value();

  auto addrLenResult =
      buildSockAddr(host, port, resolved.domain, resolved.addr);
  if (!addrLenResult) {
    return std::move(addrLenResult).error();
  }
  resolved.addrLen = addrLenResult.value();
  return resolved;
}

std::unique_ptr<Conn> finishConnect(int sock, const TcpSocketConfig& config) {
  auto status = configureClientSocket(sock, config);
  if (!status) {
    UNIFLOW_LOG_ERROR(
        "TcpClient: socket config failed fd={}: {}",
        sock,
        status.error().toString());
    ::close(sock);
    return nullptr;
  }
  return TcpConn::create(sock);
}

std::unique_ptr<Conn> completeAsyncConnect(
    int fd,
    const TcpSocketConfig& config) {
  int flags = fcntl(fd, F_GETFL);
  if (flags < 0 || fcntl(fd, F_SETFL, flags & ~O_NONBLOCK) < 0) {
    UNIFLOW_LOG_ERROR(
        "TcpClient: fcntl O_NONBLOCK clear failed fd={}: {}",
        fd,
        std::system_category().message(errno));
    ::close(fd);
    return nullptr;
  }

  auto timeoutStatus = setHandshakeTimeout(fd);
  if (!timeoutStatus) {
    UNIFLOW_LOG_ERROR(
        "TcpClient: handshake timeout failed fd={}: {}",
        fd,
        timeoutStatus.error().toString());
    ::close(fd);
    return nullptr;
  }

  // Handshake first with 500ms timeout, then configure production timeouts.
  // configureClientSocket sets 30s SO_SNDTIMEO/SO_RCVTIMEO which would
  // override the handshake timeout if called before TcpConn::create.
  auto conn = TcpConn::create(fd);
  if (!conn) {
    return nullptr;
  }

  auto status = configureClientSocket(conn->getFd(), config);
  if (!status) {
    UNIFLOW_LOG_ERROR(
        "TcpClient: socket config failed fd={}: {}",
        conn->getFd(),
        status.error().toString());
    return nullptr;
  }
  return conn;
}

} // namespace

// ---------------------------------------------------------------------------
// SyncConnect
// ---------------------------------------------------------------------------

std::future<std::unique_ptr<Conn>> SyncConnect::connect(
    const std::string& id,
    const TcpSocketConfig& config) {
  auto resolved = resolveConnectAddr(id);
  if (!resolved) {
    UNIFLOW_LOG_ERROR("TcpClient: {}", resolved.error().toString());
    return make_ready_future(std::unique_ptr<Conn>(nullptr));
  }

  UNIFLOW_LOG_INFO("TcpClient: connecting to {}", id);

  for (size_t attempt = 0; attempt <= config.connectRetries; ++attempt) {
    if (attempt > 0) {
      UNIFLOW_LOG_WARN(
          "TcpClient: retry {}/{} to {} (backoff {}ms)",
          attempt,
          config.connectRetries,
          id,
          (attempt * config.retryTimeout).count());
      // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
      std::this_thread::sleep_for(attempt * config.retryTimeout);
    }

    // Fresh socket per attempt — connect on a failed socket is UB on Linux
    int sock = ::socket(resolved->domain, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (sock < 0) {
      UNIFLOW_LOG_ERROR(
          "TcpClient: socket creation failed: {}",
          std::system_category().message(errno));
      return make_ready_future(std::unique_ptr<Conn>(nullptr));
    }

    if (::connect(
            sock,
            reinterpret_cast<sockaddr*>(&resolved->addr),
            resolved->addrLen) == 0) {
      // @lint-ignore PULSE_RESOURCE_LEAK fd ownership transfers to
      // finishConnect which closes on failure or passes to TcpConn
      return make_ready_future(finishConnect(sock, config));
    }

    int savedErrno = errno;
    ::close(sock);

    if (!shouldRetry(savedErrno)) {
      UNIFLOW_LOG_ERROR(
          "TcpClient: connect to {} failed (non-retryable): {} ({})",
          id,
          savedErrno,
          std::system_category().message(savedErrno));
      return make_ready_future(std::unique_ptr<Conn>(nullptr));
    }
  }

  UNIFLOW_LOG_ERROR(
      "TcpClient: connect to {} failed after {} retries",
      id,
      config.connectRetries);
  return make_ready_future(std::unique_ptr<Conn>(nullptr));
}

// ---------------------------------------------------------------------------
// AsyncConnect
// ---------------------------------------------------------------------------

std::future<std::unique_ptr<Conn>> AsyncConnect::connect(
    const std::string& id,
    const TcpSocketConfig& config) {
  auto resolved = resolveConnectAddr(id);
  if (!resolved) {
    UNIFLOW_LOG_ERROR("TcpClient: {}", resolved.error().toString());
    return make_ready_future(std::unique_ptr<Conn>(nullptr));
  }

  int sock =
      ::socket(resolved->domain, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
  if (sock < 0) {
    UNIFLOW_LOG_ERROR(
        "TcpClient: socket creation failed: {}",
        std::system_category().message(errno));
    return make_ready_future(std::unique_ptr<Conn>(nullptr));
  }

  int rc = ::connect(
      sock, reinterpret_cast<sockaddr*>(&resolved->addr), resolved->addrLen);
  if (rc == 0) {
    return make_ready_future(completeAsyncConnect(sock, config));
  }
  if (errno != EINPROGRESS) {
    UNIFLOW_LOG_ERROR(
        "TcpClient: connect failed: {}", std::system_category().message(errno));
    ::close(sock);
    return make_ready_future(std::unique_ptr<Conn>(nullptr));
  }

  auto promise = std::make_shared<std::promise<std::unique_ptr<Conn>>>();
  auto future = promise->get_future();

  evb_.registerFd(
      sock,
      EPOLLOUT | EPOLLONESHOT,
      [&evb = evb_, fd = sock, promise = std::move(promise), config = config](
          uint32_t) mutable {
        int err = 0;
        socklen_t errLen = sizeof(err);
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &errLen) < 0) {
          UNIFLOW_LOG_ERROR(
              "TcpClient: getsockopt SO_ERROR failed fd={}: {} ({})",
              fd,
              errno,
              std::system_category().message(errno));
          ::close(fd);
          promise->set_value(nullptr);
        } else if (err != 0) {
          UNIFLOW_LOG_ERROR(
              "TcpClient: async connect failed fd={}: {} ({})",
              fd,
              err,
              std::system_category().message(err));
          ::close(fd);
          promise->set_value(nullptr);
        } else {
          auto conn = completeAsyncConnect(fd, config);
          promise->set_value(std::move(conn));
        }
        // Clean up the ioEntries_ entry to release captured state.
        // The fd may already be closed — unregisterFd tolerates this.
        evb.unregisterFd(fd);
      });

  return future;
}

template class BasicTcpClient<SyncConnect>;
template class BasicTcpClient<AsyncConnect>;

} // namespace uniflow::controller
