// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/controller/TcpController.h"

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <system_error>
#include <thread>

#include "comms/uniflow/logging/Logger.h"

namespace uniflow::controller {

namespace {

constexpr int kUserTimeoutMs = 60000;
constexpr int kKeepaliveIdleSec = 60;
constexpr int kKeepaliveIntervalSec = 5;
constexpr int kKeepaliveCount = 3;
constexpr int kSocketBufSize = 1 << 20; // 1MB send/recv buffer
constexpr uint32_t kMaxMessageSize = 64 << 20; // 64MB max message size
constexpr int kAcceptTimeoutSec = 5; // accept() wakeup interval
constexpr int kConnectedTimeoutSec =
    30; // send/recv timeout on connected sockets

// Magic value exchanged during connection handshake to validate that both
// endpoints are uniflow controllers (rejects stray connections).
constexpr uint32_t kMagic = 0x554E4946; // "UNIF" in ASCII

// Helper for setting socket options with error collection. Accumulates the
// names of all failed options and produces a single Status at the end.
class SockOptSetter {
  int sock_;
  std::string failures_;

 public:
  explicit SockOptSetter(int sock) : sock_(sock) {}

  // Checked — collects failure name if setsockopt fails.
  template <typename T>
  void set(int level, int optname, const T& value, const char* name) {
    if (::setsockopt(sock_, level, optname, &value, sizeof(value)) < 0) {
      if (!failures_.empty()) {
        failures_ += ", ";
      }
      failures_ += name;
    }
  }

  // Optional — best-effort, failure is silently ignored.
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

// Returns true if the given errno indicates a transient error that should
// be retried. Aligned with ctran/bootstrap/Socket.cc::shouldRetry().
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
  try {
    port = std::stoi(std::string(portStr));
  } catch (const std::exception&) {
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
  // Empty host and "*" are wildcards — use IPv6 dual-stack (AF_INET6 with
  // in6addr_any accepts both IPv4 and IPv6 when IPV6_V6ONLY is not set,
  // which is the Linux default)
  if (host.empty() || host == "*") {
    return AF_INET6;
  }

  std::string hostStr(host);

  // Try IPv6 first
  in6_addr addr6{};
  if (::inet_pton(AF_INET6, hostStr.c_str(), &addr6) == 1) {
    return AF_INET6;
  }

  // Try IPv4
  in_addr addr4{};
  if (::inet_pton(AF_INET, hostStr.c_str(), &addr4) == 1) {
    return AF_INET;
  }

  return Err(ErrCode::InvalidArgument, "Invalid host address: " + hostStr);
}

// Build a sockaddr for the given host/port/domain, handling wildcard addresses
// for bind. Returns the sockaddr size.
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

// Format a sockaddr as "host:port" for logging.
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

// ---------------------------------------------------------------------------
// TcpConn
// ---------------------------------------------------------------------------

bool TcpConn::sendAll(const void* buf, size_t len) {
  auto* ptr = static_cast<const uint8_t*>(buf);
  size_t remaining = len;
  while (remaining > 0) {
    ssize_t n = ::send(sock_, ptr, remaining, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      UNIFLOW_LOG_ERROR(
          "sendAll failed: fd={} errno={} ({})",
          sock_,
          errno,
          std::system_category().message(errno));
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
      UNIFLOW_LOG_ERROR(
          "recvAll failed: fd={} errno={} ({})",
          sock_,
          errno,
          std::system_category().message(errno));
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

  // Length-prefixed framing: send 4-byte size header in network byte order
  // then payload
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

  // Length-prefixed framing: read 4-byte size header in network byte order
  // then payload
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
  opt.set(SOL_SOCKET, SO_REUSEPORT, 1, "SO_REUSEPORT"); // optional

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

TcpServer::TcpServer(std::string id, int acceptRetryCnt)
    : id_(std::move(id)), acceptRetryCnt_(acceptRetryCnt) {
  auto result = parseHostPort(id_);
  if (!result) {
    throw std::invalid_argument(
        "Invalid address format: " + result.error().toString());
  }
  auto [host, port] = result.value();
  host_ = host;
  port_ = port;
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

  // Retrieve the actual bound port (needed when binding to port 0)
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
    UNIFLOW_LOG_ERROR(
        "TcpServer: listen failed on {}: {}",
        id_,
        std::system_category().message(errno));
    ::close(listenSock_);
    listenSock_ = -1;
    return Err(
        ErrCode::ConnectionFailed,
        "listen failed: " + std::system_category().message(errno));
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

  // Retry accept on transient errors, aligned with ctran
  // ServerSocket::accept(). EAGAIN/EWOULDBLOCK from SO_RCVTIMEO timeout are
  // handled separately — they loop back to allow periodic shutdown checks
  // without counting as a transient failure.
  int retryCnt = 0;
  while (retryCnt < acceptRetryCnt_) {
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
      // Sockopt failure is a system-level issue — return immediately
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
      // Magic handshake failed (non-uniflow client) — keep accepting.
      // Socket was already closed by TcpConn destructor inside create().
      UNIFLOW_LOG_WARN(
          "TcpServer: rejecting non-uniflow client, fd={}", clientSock);
      continue;
    }

    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      // Timeout fired — check if shutdown was called
      if (listenSock_ < 0) {
        UNIFLOW_LOG_INFO("TcpServer: accept interrupted by shutdown");
        return nullptr;
      }
      continue; // don't count timeouts as retries
    }

    if (!shouldRetry(errno)) {
      UNIFLOW_LOG_ERROR(
          "TcpServer: accept failed (non-retryable): errno={} ({})",
          errno,
          std::system_category().message(errno));
      return nullptr;
    }

    ++retryCnt;
    UNIFLOW_LOG_WARN(
        "TcpServer: accept retry {}/{}: errno={} ({})",
        retryCnt,
        acceptRetryCnt_,
        errno,
        std::system_category().message(errno));
  }

  UNIFLOW_LOG_ERROR(
      "TcpServer: accept exhausted {} retries on {}", acceptRetryCnt_, id_);
  return nullptr;
}

// ---------------------------------------------------------------------------
// TcpClient
// ---------------------------------------------------------------------------

Status TcpClient::configureClientSocket(int sock) {
  SockOptSetter opt(sock);
  opt.set(SOL_SOCKET, SO_SNDBUF, kSocketBufSize, "SO_SNDBUF");
  opt.set(SOL_SOCKET, SO_RCVBUF, kSocketBufSize, "SO_RCVBUF");
  opt.set(IPPROTO_TCP, TCP_NODELAY, 1, "TCP_NODELAY");

  struct timeval tv{};
  tv.tv_sec = kConnectedTimeoutSec;
  opt.set(SOL_SOCKET, SO_SNDTIMEO, tv, "SO_SNDTIMEO");
  opt.set(SOL_SOCKET, SO_RCVTIMEO, tv, "SO_RCVTIMEO");

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

  // Build the destination sockaddr once
  sockaddr_storage serverAddr{};
  auto addrLenResult = buildSockAddr(host, port, domain, serverAddr);
  if (!addrLenResult) {
    return nullptr;
  }
  socklen_t addrLen = addrLenResult.value();

  UNIFLOW_LOG_INFO("TcpClient: connecting to {}", id);

  // Retry connect on transient errors with linear backoff,
  // aligned with ctran Socket::connect()
  for (size_t attempt = 0; attempt <= numRetries_; ++attempt) {
    if (attempt > 0) {
      UNIFLOW_LOG_WARN(
          "TcpClient: retry {}/{} to {} (backoff {}ms)",
          attempt,
          numRetries_,
          id,
          (attempt * retryTimeout_).count());
      // Intentional linear backoff between connect retries
      // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
      std::this_thread::sleep_for(attempt * retryTimeout_);
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
      "TcpClient: connect to {} failed after {} retries", id, numRetries_);
  return nullptr;
}

} // namespace uniflow::controller
