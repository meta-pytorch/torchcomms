// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "meta/tcpstore/TCPSocket.h"

#include <cstring>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include "debug.h"

#include "meta/tcpstore/Error.h"

namespace ncclx::tcpstore::detail {
namespace {
const auto pollFd = ::poll;

const auto getSocketOption = ::getsockopt;
const auto setSocketOption = ::setsockopt;

inline std::error_code getSocketError() noexcept {
  return lastError();
}

inline void setSocketError(int val) noexcept {
  errno = val;
}

// Suspends the current thread for the specified duration.
void delay(std::chrono::milliseconds d) {
  ::timespec req{};
  auto ms = d.count();
  req.tv_sec = ms / 1000;
  req.tv_nsec = (ms % 1000) * 1000000;

  // The C++ Standard does not specify whether `sleep_for()` should be signal-
  // aware; therefore, we use the `nanosleep()` syscall.
  if (::nanosleep(&req, nullptr) != 0) {
    std::error_code err = getSocketError();
    // We don't care about error conditions other than EINTR since a failure
    // here is not critical.
    if (err == std::errc::interrupted) {
      throw NetworkError(std::strerror(err.value()));
    }
  }
}

// class SocketListenOp;
class SocketConnectOp;
} // namespace

class SocketImpl {
  friend class SocketListenOp;
  friend class SocketConnectOp;

 public:
  using Handle = int;

  static constexpr Handle invalid_socket = -1;

  explicit SocketImpl(Handle hnd) noexcept : hnd_{hnd} {}

  SocketImpl(const SocketImpl& other) = delete;

  SocketImpl& operator=(const SocketImpl& other) = delete;

  SocketImpl(SocketImpl&& other) noexcept = delete;

  SocketImpl& operator=(SocketImpl&& other) noexcept = delete;

  ~SocketImpl();

  void closeOnExec() noexcept;

  void enableNonBlocking();

  void disableNonBlocking();

  bool enableNoDelay() noexcept;

  bool enableDualStack() noexcept;

  Handle handle() const noexcept {
    return hnd_;
  }

  bool waitForInput(std::chrono::milliseconds timeout);

 private:
  bool setSocketFlag(int level, int optname, bool value) noexcept;

  Handle hnd_;
};

SocketImpl::~SocketImpl() {
  ::close(hnd_);
}

void SocketImpl::closeOnExec() noexcept {
  ::fcntl(hnd_, F_SETFD, FD_CLOEXEC);
}

void SocketImpl::enableNonBlocking() {
  int flg = ::fcntl(hnd_, F_GETFL);
  if (flg != -1) {
    if (::fcntl(hnd_, F_SETFL, flg | O_NONBLOCK) == 0) {
      return;
    }
  }
  throw std::runtime_error("enableNonBlocking() failed");
}

// TODO: Remove once we migrate everything to non-blocking mode.
void SocketImpl::disableNonBlocking() {
  int flg = ::fcntl(hnd_, F_GETFL);
  if (flg != -1) {
    if (::fcntl(hnd_, F_SETFL, flg & ~O_NONBLOCK) == 0) {
      return;
    }
  }
  throw std::runtime_error("The socket cannot be switched to blocking mode.");
}

bool SocketImpl::enableNoDelay() noexcept {
  return setSocketFlag(IPPROTO_TCP, TCP_NODELAY, true);
}

bool SocketImpl::setSocketFlag(int level, int optname, bool value) noexcept {
  auto buf = value ? 1 : 0;
  return setSocketOption(hnd_, level, optname, &buf, sizeof(buf)) == 0;
}

bool SocketImpl::waitForInput(std::chrono::milliseconds timeout) {
  using Clock = std::chrono::steady_clock;

  auto deadline = Clock::now() + timeout;
  do {
    ::pollfd pfd{};
    pfd.fd = hnd_;
    pfd.events = POLLIN;

    int res = pollFd(&pfd, 1, static_cast<int>(timeout.count()));
    if (res > 0) {
      return true;
    }
    std::error_code err = getSocketError();

    if (err == std::errc::operation_in_progress) {
      bool timedout = Clock::now() >= deadline;
      if (timedout) {
        return false;
      }
      WARN(
          "pollFD for socket returned operation_in_progress before a timeout.");
    } else if (err != std::errc::interrupted) {
      WARN(
          "%s",
          fmt::format(
              "waitForInput: poll for socket failed with res={}, err={}.",
              res,
              err)
              .c_str());
      return false;
    }
  } while (Clock::now() < deadline);
  return false;
}

namespace {

struct addrinfo_delete {
  void operator()(::addrinfo* addr) const noexcept {
    ::freeaddrinfo(addr);
  }
};

using addrinfo_ptr = std::unique_ptr<::addrinfo, addrinfo_delete>;

class SocketConnectOp {
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::steady_clock::duration;
  using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

  enum class ConnectResult { Success, Error, Retry };

 public:
  SocketConnectOp(
      const std::string& host,
      std::uint16_t port,
      const SocketOptions& opts);

  std::unique_ptr<SocketImpl> run();

 private:
  bool tryConnect(int family);

  ConnectResult tryConnect(const ::addrinfo& addr);

  ConnectResult tryConnectCore(const ::addrinfo& addr);

  [[noreturn]] void throwTimeoutError() const;

  const char* host_;
  std::string port_;
  const SocketOptions* opts_;
  TimePoint deadline_{};
  std::unique_ptr<SocketImpl> socket_{};
};

SocketConnectOp::SocketConnectOp(
    const std::string& host,
    std::uint16_t port,
    const SocketOptions& opts)
    : host_{host.c_str()}, port_{std::to_string(port)}, opts_{&opts} {}

std::unique_ptr<SocketImpl> SocketConnectOp::run() {
  if (opts_->prefer_ipv6()) {
    INFO(
        NCCL_INIT,
        "The client socket will attempt to connect to an IPv6 address of (%s, %s).",
        host_,
        port_.c_str());

    if (tryConnect(AF_INET6)) {
      return std::move(socket_);
    }

    INFO(
        NCCL_INIT,
        "The client socket will attempt to connect to an IPv4 address of (%s, %s).",
        host_,
        port_.c_str());

    if (tryConnect(AF_INET)) {
      return std::move(socket_);
    }
  } else {
    INFO(
        NCCL_INIT,
        "The client socket will attempt to connect to an IPv4 or IPv6 address of (%s, %s).",
        host_,
        port_.c_str());

    if (tryConnect(AF_UNSPEC)) {
      return std::move(socket_);
    }
  }

  auto msg = fmt::format(
      "The client socket has failed to connect to any network address of ({}, {}).",
      host_,
      port_);

  ERR("%s", msg.c_str());
  throw std::runtime_error(msg);
}

bool SocketConnectOp::tryConnect(int family) {
  ::addrinfo hints{};
  hints.ai_flags = AI_V4MAPPED | AI_ALL | AI_NUMERICSERV;
  hints.ai_family = family;
  hints.ai_socktype = SOCK_STREAM;

  deadline_ = Clock::now() + opts_->connect_timeout();

  bool retry; // NOLINT(cppcoreguidelines-init-variables)
  do {
    retry = false;

    ::addrinfo* naked_result = nullptr;
    // patternlint-disable cpp-dns-deps
    int r = ::getaddrinfo(host_, port_.c_str(), &hints, &naked_result);
    if (r != 0) {
      const char* gai_err = ::gai_strerror(r);

      WARN(
          "The %s network addresses of (%s, %s) cannot be retrieved (gai error: %d - %s).",
          family == AF_INET        ? "IPv4"
              : family == AF_INET6 ? "IPv6"
                                   : "",
          host_,
          port_.c_str(),
          r,
          gai_err);

      retry = true;
    } else {
      addrinfo_ptr result{naked_result};

      for (::addrinfo* addr = naked_result; addr != nullptr;
           addr = addr->ai_next) {
        ConnectResult cr = tryConnect(*addr);
        if (cr == ConnectResult::Success) {
          return true;
        }

        if (cr == ConnectResult::Retry) {
          retry = true;
        }
      }
    }

    if (retry) {
      auto connectBackoff = opts_->connect_backoff();
      auto delayDuration = connectBackoff->nextBackoff();

      if (Clock::now() < deadline_ - delayDuration) {
        // Prevent our log output to be too noisy, warn only every 1 second.
        static auto lastLog = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if ((now - lastLog) >= std::chrono::seconds(1)) {
          INFO(
              NCCL_INIT,
              "No socket on (%s, %s) is listening yet, will retry.",
              host_,
              port_.c_str());
          lastLog = now;
        }
        // Wait to avoid choking the server.
        delay(delayDuration);
      } else {
        throwTimeoutError();
      }
    }
  } while (retry);

  return false;
}

SocketConnectOp::ConnectResult SocketConnectOp::tryConnect(
    const ::addrinfo& addr) {
  if (Clock::now() >= deadline_) {
    throwTimeoutError();
  }

  SocketImpl::Handle hnd =
      ::socket(addr.ai_family, addr.ai_socktype, addr.ai_protocol);
  if (hnd == SocketImpl::invalid_socket) {
    WARN(
        "%s",
        fmt::format(
            "The client socket cannot be initialized to connect, error: {}.",
            getSocketError())
            .c_str());
    return ConnectResult::Error;
  }

  socket_ = std::make_unique<SocketImpl>(hnd);

  socket_->enableNonBlocking();

  ConnectResult cr = tryConnectCore(addr);
  if (cr == ConnectResult::Error) {
    std::error_code err = getSocketError();
    if (err == std::errc::interrupted) {
      // C10_THROW_ERROR(DistNetworkError, std::strerror(err.value()));
      throw NetworkError(std::strerror(err.value()));
    }

    // Retry if the server is not yet listening or if its backlog is exhausted.
    if (err == std::errc::connection_refused ||
        err == std::errc::connection_reset) {
      return ConnectResult::Retry;
    } else if (err == std::errc::timed_out) {
      WARN("Socket connection has timed out, will retry.");
      return ConnectResult::Retry;
    } else {
      WARN(
          "%s",
          fmt::format(
              "The client socket has failed to connect, error: {}.", err)
              .c_str());
      return ConnectResult::Error;
    }
  }

  socket_->closeOnExec();

  // TODO: Remove once we fully migrate to non-blocking mode.
  socket_->disableNonBlocking();

  INFO(NCCL_INIT, "The client socket has connected to server");

  if (!socket_->enableNoDelay()) {
    WARN("The no-delay option cannot be enabled for the client socket.");
  }

  return ConnectResult::Success;
}

SocketConnectOp::ConnectResult SocketConnectOp::tryConnectCore(
    const ::addrinfo& addr) {
  int r = ::connect(socket_->handle(), addr.ai_addr, addr.ai_addrlen);
  if (r == 0) {
    return ConnectResult::Success;
  }

  std::error_code err = getSocketError();
  if (err == std::errc::already_connected) {
    return ConnectResult::Success;
  }

  if (err != std::errc::operation_in_progress &&
      err != std::errc::operation_would_block) {
    return ConnectResult::Error;
  }

  Duration remaining = deadline_ - Clock::now();
  if (remaining <= Duration::zero()) {
    throwTimeoutError();
  }

  ::pollfd pfd{};
  pfd.fd = socket_->handle();
  pfd.events = POLLOUT;

  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(remaining);

  r = pollFd(&pfd, 1, static_cast<int>(ms.count()));
  if (r == 0) {
    throwTimeoutError();
  }
  if (r == -1) {
    return ConnectResult::Error;
  }

  int err_code = 0;

  ::socklen_t err_len = sizeof(int);

  r = getSocketOption(
      socket_->handle(), SOL_SOCKET, SO_ERROR, &err_code, &err_len);
  if (r != 0) {
    return ConnectResult::Error;
  }

  if (err_code != 0) {
    setSocketError(err_code);

    return ConnectResult::Error;
  } else {
    return ConnectResult::Success;
  }
}

void SocketConnectOp::throwTimeoutError() const {
  auto msg = fmt::format(
      "The client socket has timed out after {} while trying to connect to ({}, {}).",
      opts_->connect_timeout(),
      host_,
      port_);

  ERR("%s", msg.c_str());
  throw std::runtime_error(msg);
}

} // namespace

void Socket::initialize() {}

Socket Socket::connect(
    const std::string& host,
    std::uint16_t port,
    const SocketOptions& opts) {
  SocketConnectOp op{host, port, opts};

  return Socket{op.run()};
}

Socket::Socket(Socket&& other) noexcept = default;

Socket& Socket::operator=(Socket&& other) noexcept = default;

Socket::~Socket() = default;

int Socket::handle() const noexcept {
  if (impl_) {
    return impl_->handle();
  }
  return SocketImpl::invalid_socket;
}

Socket::Socket(std::unique_ptr<SocketImpl>&& impl) noexcept
    : impl_{std::move(impl)} {}

bool Socket::waitForInput(std::chrono::milliseconds timeout) {
  return impl_->waitForInput(timeout);
}

} // namespace ncclx::tcpstore::detail
