// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/controller/TcpController.h"

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <stdexcept>
#include <system_error>
#include <thread>

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

constexpr uint32_t kMaxMessageSize = 64 << 20; // 64MB max message size
constexpr int kAcceptTimeoutSec = 5; // accept() wakeup interval

// Magic value exchanged during connection handshake to validate that both
// endpoints are uniflow controllers (rejects stray connections).
constexpr uint32_t kMagic = 0x554E4946; // "UNIF" in ASCII

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

  // Best-effort, failure silently ignored.
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
      errno = ECONNRESET; // peer closed
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
// TcpServer
// ---------------------------------------------------------------------------

Result<int> TcpServer::createListenSocket(int domain) {
  int sock = ::socket(domain, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (sock < 0) {
    return Err(
        ErrCode::ConnectionFailed,
        "socket creation failed: " + std::system_category().message(errno));
  }

  SockOptSetter opt(sock);
  opt.set(SOL_SOCKET, SO_REUSEADDR, 1, "SO_REUSEADDR");
  opt.trySet(SOL_SOCKET, SO_REUSEPORT, 1);

  // Set accept timeout so accept() wakes up periodically, allowing
  // shutdown checks instead of blocking indefinitely
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

Status TcpServer::configureAcceptedSocket(int sock) {
  SockOptSetter opt(sock);

  if (config_.socketBufSize) {
    opt.set(SOL_SOCKET, SO_SNDBUF, *config_.socketBufSize, "SO_SNDBUF");
    opt.set(SOL_SOCKET, SO_RCVBUF, *config_.socketBufSize, "SO_RCVBUF");
  }
  if (config_.tcpNoDelay) {
    int val = *config_.tcpNoDelay ? 1 : 0;
    opt.set(IPPROTO_TCP, TCP_NODELAY, val, "TCP_NODELAY");
  }
  if (config_.enableKeepalive) {
    int val = *config_.enableKeepalive ? 1 : 0;
    opt.set(SOL_SOCKET, SO_KEEPALIVE, val, "SO_KEEPALIVE");
  }
  if (config_.enableKeepalive && *config_.enableKeepalive) {
    if (config_.keepaliveIdle) {
      int val = static_cast<int>(config_.keepaliveIdle->count());
      opt.set(IPPROTO_TCP, TCP_KEEPIDLE, val, "TCP_KEEPIDLE");
    }
    if (config_.keepaliveInterval) {
      int val = static_cast<int>(config_.keepaliveInterval->count());
      opt.set(IPPROTO_TCP, TCP_KEEPINTVL, val, "TCP_KEEPINTVL");
    }
    if (config_.keepaliveCount) {
      opt.set(IPPROTO_TCP, TCP_KEEPCNT, *config_.keepaliveCount, "TCP_KEEPCNT");
    }
  }
  if (config_.userTimeout) {
    int val = static_cast<int>(config_.userTimeout->count());
    opt.set(IPPROTO_TCP, TCP_USER_TIMEOUT, val, "TCP_USER_TIMEOUT");
  }
  if (config_.connTimeout) {
    struct timeval tv{};
    tv.tv_sec = config_.connTimeout->count();
    opt.set(SOL_SOCKET, SO_SNDTIMEO, tv, "SO_SNDTIMEO");
    opt.set(SOL_SOCKET, SO_RCVTIMEO, tv, "SO_RCVTIMEO");
  }

  return opt.status();
}

Status TcpServer::resolveAndBind(int domain) {
  sockaddr_storage addr{};
  auto addrLenResult = buildSockAddr(host_, port_, domain, addr);
  if (!addrLenResult) {
    return std::move(addrLenResult).error();
  }

  if (::bind(
          listenSock_,
          reinterpret_cast<sockaddr*>(&addr),
          addrLenResult.value()) < 0) {
    return Err(
        ErrCode::ConnectionFailed,
        "bind failed: " + std::system_category().message(errno));
  }

  return Ok();
}

TcpServer::TcpServer(std::string id, TcpSocketConfig config)
    : id_(std::move(id)), config_(std::move(config)) {
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

TcpServer::~TcpServer() {
  shutdown();
}

const std::string& TcpServer::getId() const {
  return id_;
}

void TcpServer::shutdown() {
  int sock = listenSock_.exchange(-1);
  if (sock >= 0) {
    UNIFLOW_LOG_INFO("TcpServer: shutting down {}, fd={}", id_, sock);
    // SHUT_RDWR unblocks any thread blocked in accept()
    ::shutdown(sock, SHUT_RDWR);
    ::close(sock);
  }
}

Status TcpServer::init() {
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

  auto bindStatus = resolveAndBind(domain);
  if (!bindStatus) {
    UNIFLOW_LOG_ERROR(
        "TcpServer: bind failed on {}: {}", id_, bindStatus.error().toString());
    ::close(listenSock_);
    listenSock_ = -1;
    return bindStatus;
  }

  // Retrieve actual bound port (needed when binding to port 0).
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

std::unique_ptr<Conn> TcpServer::accept() {
  if (listenSock_ < 0) {
    return nullptr;
  }

  UNIFLOW_LOG_DEBUG("TcpServer: waiting for connection on {}", id_);

  sockaddr_storage clientAddr{};
  socklen_t clientLen = sizeof(clientAddr);

  int retryCnt = 0;
  while (retryCnt < config_.acceptRetryCnt) {
    clientLen = sizeof(clientAddr);
    int clientSock = ::accept4(
        listenSock_,
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
        return nullptr;
      }
      auto conn = TcpConn::create(clientSock);
      if (conn) {
        return conn;
      }
      // Non-uniflow client — socket closed by TcpConn dtor, keep accepting.
      UNIFLOW_LOG_WARN(
          "TcpServer: rejecting non-uniflow client, fd={}", clientSock);
      continue;
    }

    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      if (listenSock_ < 0) {
        UNIFLOW_LOG_INFO("TcpServer: accept interrupted by shutdown");
        return nullptr;
      }
      continue;
    }

    int savedErrno = errno;
    if (!shouldRetry(savedErrno)) {
      UNIFLOW_LOG_ERROR(
          "TcpServer: accept failed (non-retryable): errno={} ({})",
          savedErrno,
          std::system_category().message(savedErrno));
      return nullptr;
    }

    ++retryCnt;
    UNIFLOW_LOG_WARN(
        "TcpServer: accept retry {}/{}: errno={} ({})",
        retryCnt,
        config_.acceptRetryCnt,
        savedErrno,
        std::system_category().message(savedErrno));
  }

  UNIFLOW_LOG_ERROR(
      "TcpServer: accept exhausted {} retries on {}",
      config_.acceptRetryCnt,
      id_);
  return nullptr;
}

TcpClient::TcpClient(TcpSocketConfig config) : config_(std::move(config)) {
  auto status = config_.validate();
  if (!status) {
    throw std::invalid_argument(
        "Invalid socket config: " + status.error().toString());
  }
}

Status TcpClient::configureClientSocket(int sock) {
  SockOptSetter opt(sock);

  if (config_.socketBufSize) {
    opt.set(SOL_SOCKET, SO_SNDBUF, *config_.socketBufSize, "SO_SNDBUF");
    opt.set(SOL_SOCKET, SO_RCVBUF, *config_.socketBufSize, "SO_RCVBUF");
  }
  if (config_.tcpNoDelay) {
    int val = *config_.tcpNoDelay ? 1 : 0;
    opt.set(IPPROTO_TCP, TCP_NODELAY, val, "TCP_NODELAY");
  }
  if (config_.enableKeepalive) {
    int val = *config_.enableKeepalive ? 1 : 0;
    opt.set(SOL_SOCKET, SO_KEEPALIVE, val, "SO_KEEPALIVE");
  }
  if (config_.enableKeepalive && *config_.enableKeepalive) {
    if (config_.keepaliveIdle) {
      int val = static_cast<int>(config_.keepaliveIdle->count());
      opt.set(IPPROTO_TCP, TCP_KEEPIDLE, val, "TCP_KEEPIDLE");
    }
    if (config_.keepaliveInterval) {
      int val = static_cast<int>(config_.keepaliveInterval->count());
      opt.set(IPPROTO_TCP, TCP_KEEPINTVL, val, "TCP_KEEPINTVL");
    }
    if (config_.keepaliveCount) {
      opt.set(IPPROTO_TCP, TCP_KEEPCNT, *config_.keepaliveCount, "TCP_KEEPCNT");
    }
  }
  if (config_.userTimeout) {
    int val = static_cast<int>(config_.userTimeout->count());
    opt.set(IPPROTO_TCP, TCP_USER_TIMEOUT, val, "TCP_USER_TIMEOUT");
  }
  if (config_.connTimeout) {
    struct timeval tv{};
    tv.tv_sec = config_.connTimeout->count();
    opt.set(SOL_SOCKET, SO_SNDTIMEO, tv, "SO_SNDTIMEO");
    opt.set(SOL_SOCKET, SO_RCVTIMEO, tv, "SO_RCVTIMEO");
  }

  return opt.status();
}

std::unique_ptr<Conn> TcpClient::connect(std::string id) {
  auto result = parseHostPort(id);
  if (!result) {
    UNIFLOW_LOG_ERROR("TcpClient: invalid address: {}", id);
    return nullptr;
  }
  auto [host, port] = std::move(result).value();

  auto domainResult = detectAddressFamily(host);
  if (!domainResult) {
    UNIFLOW_LOG_ERROR("TcpClient: invalid host: {}", host);
    return nullptr;
  }
  int domain = domainResult.value();

  sockaddr_storage serverAddr{};
  auto addrLenResult = buildSockAddr(host, port, domain, serverAddr);
  if (!addrLenResult) {
    return nullptr;
  }
  socklen_t addrLen = addrLenResult.value();

  UNIFLOW_LOG_INFO("TcpClient: connecting to {}", id);

  for (size_t attempt = 0; attempt <= config_.connectRetries; ++attempt) {
    if (attempt > 0) {
      UNIFLOW_LOG_WARN(
          "TcpClient: retry {}/{} to {} (backoff {}ms)",
          attempt,
          config_.connectRetries,
          id,
          (attempt * config_.retryTimeout).count());
      // Intentional linear backoff between connect retries
      // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
      std::this_thread::sleep_for(attempt * config_.retryTimeout);
    }

    // Create a fresh socket for each attempt (connect on a failed socket
    // is undefined behavior on Linux)
    int sock = ::socket(domain, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (sock < 0) {
      UNIFLOW_LOG_ERROR(
          "TcpClient: socket creation failed: {}",
          std::system_category().message(errno));
      return nullptr;
    }

    if (::connect(sock, reinterpret_cast<sockaddr*>(&serverAddr), addrLen) ==
        0) {
      auto status = configureClientSocket(sock);
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

    int savedErrno = errno;
    ::close(sock);

    if (!shouldRetry(savedErrno)) {
      UNIFLOW_LOG_ERROR(
          "TcpClient: connect to {} failed (non-retryable): {} ({})",
          id,
          savedErrno,
          std::system_category().message(savedErrno));
      return nullptr;
    }
  }

  UNIFLOW_LOG_ERROR(
      "TcpClient: connect to {} failed after {} retries",
      id,
      config_.connectRetries);
  return nullptr;
}

} // namespace uniflow::controller
