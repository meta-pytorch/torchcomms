// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/controller/TcpController.h"

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "comms/uniflow/executor/ScopedEventBaseThread.h"

using namespace uniflow;
using namespace uniflow::controller;

// ---------------------------------------------------------------------------
// Parameterized fixture: async accept tests over both IPv4 and IPv6.
// ---------------------------------------------------------------------------

struct AddrFamily {
  std::string serverAddr;
  std::string clientHost;
  // Stated rather than derived from clientHost: comparing against the
  // "127.0.0.1" literal silently defaults every other spelling -- "localhost",
  // "[::1]", a resolvable hostname -- to AF_INET6.
  int family;
};

class TcpAsyncAcceptTest : public ::testing::TestWithParam<AddrFamily> {
 protected:
  std::string clientAddr(int port) const {
    return GetParam().clientHost + ":" + std::to_string(port);
  }
};

namespace {

// Polls `flag` until it is set, or gives up. Used where the thing under test is
// "does this call return at all", so a hang has to become a failed assertion
// rather than a hung test binary.
bool waitForFlag(
    const std::atomic<bool>& flag,
    std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (flag.load(std::memory_order_acquire)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return flag.load(std::memory_order_acquire);
}

int getRcvBuf(int fd) {
  int val = 0;
  socklen_t len = sizeof(val);
  EXPECT_EQ(::getsockopt(fd, SOL_SOCKET, SO_RCVBUF, &val, &len), 0);
  return val;
}

// SO_RCVBUF on a socket nobody has configured. Used as the control for "the
// kernel is still in charge" so the assertions do not depend on the host's
// net.core.rmem_max / tcp_rmem values. Takes the family from the caller because
// this fixture runs over both IPv4 and IPv6, and a probe from the wrong family
// is not guaranteed to report the same default.
//
// SOCK_STREAM matters: tcp_init_sock() overwrites the sock_init_data() default
// with tcp_rmem[1], which is the same value an accepted TCP socket starts from,
// so net.core.rmem_default never enters into it for either socket.
int kernelDefaultRcvBuf(int family) {
  int probe = ::socket(family, SOCK_STREAM, 0);
  if (probe < 0) {
    return -1;
  }
  int val = getRcvBuf(probe);
  ::close(probe);
  return val;
}

// SO_RCVBUF as this kernel stores it for a given request, measured on a
// throwaway socket of the same family. Requesting a size is not the same as
// getting it: the kernel clamps the request to net.core.rmem_max, doubles it
// for skb overhead, and floors it at SOCK_MIN_RCVBUF. Measuring that instead of
// reproducing the arithmetic keeps the expectation correct on any host tuning.
int rcvBufForRequest(int family, int request) {
  int probe = ::socket(family, SOCK_STREAM, 0);
  if (probe < 0) {
    return -1;
  }
  if (::setsockopt(probe, SOL_SOCKET, SO_RCVBUF, &request, sizeof(request)) !=
      0) {
    ::close(probe);
    return -1;
  }
  int val = getRcvBuf(probe);
  ::close(probe);
  return val;
}

// Accepted connections are TcpConn<SyncIO>; only that type exposes the fd.
int acceptedFd(const std::unique_ptr<Conn>& conn) {
  auto* tcp = dynamic_cast<TcpConn<SyncIO>*>(conn.get());
  return tcp == nullptr ? -1 : tcp->getFd();
}

int getSockOptInt(int fd, int level, int optname) {
  int val = 0;
  socklen_t len = sizeof(val);
  EXPECT_EQ(::getsockopt(fd, level, optname, &val, &len), 0);
  return val;
}

// The value an untouched socket of this family reports, so "the kernel is still
// in charge" is measured rather than assumed. Hardcoding 0 would encode a
// current Linux default instead of the property under test.
int probeSockOptInt(int family, int level, int optname) {
  int probe = ::socket(family, SOCK_STREAM, 0);
  if (probe < 0) {
    return -1;
  }
  int val = getSockOptInt(probe, level, optname);
  ::close(probe);
  return val;
}

} // namespace

TEST_P(TcpAsyncAcceptTest, SingleAsyncAccept) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  auto future = server.accept();

  TcpClient client;
  auto clientConn = client.connect(clientAddr(port)).get();
  ASSERT_NE(clientConn, nullptr) << "Client failed to connect";

  auto conn = future.get();
  EXPECT_NE(conn, nullptr);
}

// Leaving socketBufSize unset must leave the kernel's own buffer sizing alone.
// An explicit SO_RCVBUF disables receive-window autotuning, so the buffer can
// no longer grow to absorb a stall while the reader is busy with a multi-MiB
// copy. Before the config was threaded through the accept policy,
// configureAcceptedSocket hardcoded 1 MiB, so a caller had no way to opt out.
TEST_P(TcpAsyncAcceptTest, UnsetSocketBufSizeLeavesKernelAutotuning) {
  const int kernelDefault = kernelDefaultRcvBuf(GetParam().family);
  ASSERT_GT(kernelDefault, 0);

  TcpSocketConfig cfg;
  cfg.socketBufSize = std::nullopt;

  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, cfg, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }

  auto future = server.accept();
  TcpClient client;
  auto clientConn = client.connect(clientAddr(server.getPort())).get();
  ASSERT_NE(clientConn, nullptr) << "Client failed to connect";
  auto conn = future.get();
  ASSERT_NE(conn, nullptr);

  const int fd = acceptedFd(conn);
  ASSERT_GE(fd, 0);
  // Read once: with autotuning active the kernel may adjust sk_rcvbuf between
  // calls, and both bounds should describe the same observation.
  const int rcvBuf = getRcvBuf(fd);
  // A lower bound rather than equality: this socket has completed a handshake,
  // and with autotuning left on (which is the thing being asserted) the kernel
  // is allowed to grow sk_rcvbuf above the initial tcp_rmem[1].
  EXPECT_GE(rcvBuf, kernelDefault);
  // The upper bound carries the regression. It is scaled off the probe rather
  // than a 1 MiB literal so that a host tuned with a large tcp_rmem[1] cannot
  // make the two bounds mutually unsatisfiable. Restoring the hardcode sets
  // SO_RCVBUF explicitly, which the kernel doubles, so it lands well outside
  // this bound; autotuning alone does not double the buffer on a connection
  // that has carried no payload.
  EXPECT_LT(rcvBuf, 2 * kernelDefault);
}

// The configured value reaches the accepted socket. The assertion is pinned to
// the exact value the kernel stores for the request, because a range wide
// enough to contain the unconfigured default would also be satisfied if
// socketBufSize were dropped again.
TEST_P(TcpAsyncAcceptTest, AcceptedSocketAppliesConfiguredSocketBufSize) {
  const int kernelDefault = kernelDefaultRcvBuf(GetParam().family);
  ASSERT_GT(kernelDefault, 0);

  // Scaled off the probe rather than a literal. A quarter of the default
  // doubles to half the default, which cannot coincide with the default itself;
  // a fixed 64 KiB doubles to exactly the stock tcp_rmem[1] of 131072, which
  // would make a correctly configured socket indistinguishable from an
  // untouched one on any stock host. Staying below the default also keeps the
  // request clear of net.core.rmem_max, so it is not silently clamped.
  const int requested = kernelDefault / 4;
  const int expected = rcvBufForRequest(GetParam().family, requested);
  ASSERT_GT(expected, 0);

  // Whether the configured size is observably different from the default is a
  // property of the host, not of the code under test, so skip rather than fail:
  // where the two coincide this test can prove nothing either way.
  if (expected == kernelDefault) {
    GTEST_SKIP() << "configured size is indistinguishable from the kernel "
                    "default ("
                 << kernelDefault << ") on this host";
  }

  TcpSocketConfig cfg;
  cfg.socketBufSize = requested;

  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, cfg, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }

  auto future = server.accept();
  TcpClient client;
  auto clientConn = client.connect(clientAddr(server.getPort())).get();
  ASSERT_NE(clientConn, nullptr) << "Client failed to connect";
  auto conn = future.get();
  ASSERT_NE(conn, nullptr);

  const int fd = acceptedFd(conn);
  ASSERT_GE(fd, 0);
  EXPECT_EQ(getRcvBuf(fd), expected);
}

// osDefaults() means "leave the OS alone", and that has to hold for accepted
// sockets too. Before configureAcceptedSocket applied the config, it hardcoded
// TCP_NODELAY=1 and SO_KEEPALIVE=1 regardless, so this configuration was
// silently ignored on the server side while the client honoured it.
TEST_P(TcpAsyncAcceptTest, AcceptedSocketLeavesOsDefaultsUntouched) {
  const int family = GetParam().family;
  const int probeNoDelay = probeSockOptInt(family, IPPROTO_TCP, TCP_NODELAY);
  const int probeKeepalive = probeSockOptInt(family, SOL_SOCKET, SO_KEEPALIVE);
  ASSERT_GE(probeNoDelay, 0);
  ASSERT_GE(probeKeepalive, 0);

  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(
      GetParam().serverAddr,
      TcpSocketConfig::osDefaults(),
      *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }

  auto future = server.accept();
  TcpClient client;
  auto clientConn = client.connect(clientAddr(server.getPort())).get();
  ASSERT_NE(clientConn, nullptr) << "Client failed to connect";
  auto conn = future.get();
  ASSERT_NE(conn, nullptr);

  const int fd = acceptedFd(conn);
  ASSERT_GE(fd, 0);
  EXPECT_EQ(getSockOptInt(fd, IPPROTO_TCP, TCP_NODELAY), probeNoDelay);
  EXPECT_EQ(getSockOptInt(fd, SOL_SOCKET, SO_KEEPALIVE), probeKeepalive);
}

// A server that turns keepalive off must actually get it off. This is the case
// that used to be a no-op with no compiler error and no failing test.
TEST_P(TcpAsyncAcceptTest, AcceptedSocketHonoursKeepaliveDisable) {
  TcpSocketConfig cfg;
  cfg.enableKeepalive = false;

  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, cfg, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }

  auto future = server.accept();
  TcpClient client;
  auto clientConn = client.connect(clientAddr(server.getPort())).get();
  ASSERT_NE(clientConn, nullptr) << "Client failed to connect";
  auto conn = future.get();
  ASSERT_NE(conn, nullptr);

  const int fd = acceptedFd(conn);
  ASSERT_GE(fd, 0);
  EXPECT_EQ(getSockOptInt(fd, SOL_SOCKET, SO_KEEPALIVE), 0);
}

TEST_P(TcpAsyncAcceptTest, MultipleAsyncAccepts) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  constexpr int kNumClients = 3;

  std::vector<std::future<std::unique_ptr<Conn>>> futures;
  futures.reserve(kNumClients);
  for (int i = 0; i < kNumClients; ++i) {
    futures.push_back(server.accept());
  }

  TcpClient client;
  std::vector<std::unique_ptr<Conn>> clientConns;
  for (int i = 0; i < kNumClients; ++i) {
    auto c = client.connect(clientAddr(port)).get();
    ASSERT_NE(c, nullptr) << "Client " << i << " failed to connect";
    clientConns.push_back(std::move(c));
  }

  for (int i = 0; i < kNumClients; ++i) {
    auto conn = futures[i].get();
    EXPECT_NE(conn, nullptr) << "Async accept " << i << " returned nullptr";
  }
}

TEST_P(TcpAsyncAcceptTest, ConnectionQueueing) {
  ScopedEventBaseThread evbThread("async-accept");
  auto* evb = evbThread.getEventBase();
  AsyncTcpServer server(GetParam().serverAddr, {}, *evb);
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  // First accept triggers fd registration setup
  auto future1 = server.accept();

  TcpClient client;
  auto c1 = client.connect(clientAddr(port)).get();
  auto c2 = client.connect(clientAddr(port)).get();
  auto c3 = client.connect(clientAddr(port)).get();
  ASSERT_NE(c1, nullptr);
  ASSERT_NE(c2, nullptr);
  ASSERT_NE(c3, nullptr);

  auto conn1 = future1.get();
  ASSERT_NE(conn1, nullptr);

  // Wait for IO callback to process remaining connections into readyConns_
  evb->dispatchAndWait([]() noexcept {});

  auto conn2 = server.accept().get();
  auto conn3 = server.accept().get();
  EXPECT_NE(conn2, nullptr);
  EXPECT_NE(conn3, nullptr);
}

TEST_P(TcpAsyncAcceptTest, PromiseQueueing) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  auto future1 = server.accept();
  auto future2 = server.accept();

  TcpClient client;
  auto c1 = client.connect(clientAddr(port)).get();
  auto c2 = client.connect(clientAddr(port)).get();
  ASSERT_NE(c1, nullptr);
  ASSERT_NE(c2, nullptr);

  auto conn1 = future1.get();
  auto conn2 = future2.get();
  EXPECT_NE(conn1, nullptr);
  EXPECT_NE(conn2, nullptr);
}

TEST_P(TcpAsyncAcceptTest, ShutdownResolvesPromises) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }

  auto future1 = server.accept();
  auto future2 = server.accept();

  server.shutdown();

  auto conn1 = future1.get();
  auto conn2 = future2.get();
  EXPECT_EQ(conn1, nullptr);
  EXPECT_EQ(conn2, nullptr);
}

TEST_P(TcpAsyncAcceptTest, ShutdownDuringAsyncAccept) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }

  auto future = server.accept();

  // Shutdown from another thread while future.get() is blocking
  std::thread shutdownThread([&]() {
    // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    server.shutdown();
  });

  auto conn = future.get();
  EXPECT_EQ(conn, nullptr);

  shutdownThread.join();
}

TEST_P(TcpAsyncAcceptTest, AsyncAcceptRejectsNonUniflowClient) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  auto future = server.accept();

  {
    int sock = ::socket(GetParam().family, SOCK_STREAM | SOCK_CLOEXEC, 0);
    ASSERT_GE(sock, 0);

    sockaddr_storage addr{};
    if (GetParam().family == AF_INET) {
      auto* sa = reinterpret_cast<sockaddr_in*>(&addr);
      sa->sin_family = AF_INET;
      sa->sin_port = htons(static_cast<uint16_t>(port));
      ::inet_pton(AF_INET, "127.0.0.1", &sa->sin_addr);
      ::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(sockaddr_in));
    } else {
      auto* sa = reinterpret_cast<sockaddr_in6*>(&addr);
      sa->sin6_family = AF_INET6;
      sa->sin6_port = htons(static_cast<uint16_t>(port));
      ::inet_pton(AF_INET6, "::1", &sa->sin6_addr);
      ::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(sockaddr_in6));
    }
    uint32_t garbage = 0xDEADBEEF;
    ::send(sock, &garbage, sizeof(garbage), 0);
    ::close(sock);
  }

  // Now connect a valid uniflow client — should still be accepted
  TcpClient client;
  auto clientConn = client.connect(clientAddr(port)).get();
  ASSERT_NE(clientConn, nullptr) << "Valid client failed after bad client";

  auto conn = future.get();
  EXPECT_NE(conn, nullptr);
}

TEST_P(TcpAsyncAcceptTest, SendRecvAfterAsyncAccept) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  auto future = server.accept();

  TcpClient client;
  auto clientConn = client.connect(clientAddr(port)).get();
  ASSERT_NE(clientConn, nullptr);

  auto serverConn = future.get();
  ASSERT_NE(serverConn, nullptr);

  const std::vector<uint8_t> msg = {0x01, 0x02, 0x03, 0x04};
  auto sendResult = clientConn->send(msg).get();
  ASSERT_TRUE(sendResult.hasValue()) << sendResult.error().toString();

  std::vector<uint8_t> buf;
  auto recvResult = serverConn->recv(buf).get();
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(buf, msg);

  const std::vector<uint8_t> reply = {0x05, 0x06};
  sendResult = serverConn->send(reply).get();
  ASSERT_TRUE(sendResult.hasValue()) << sendResult.error().toString();

  buf.clear();
  recvResult = clientConn->recv(buf).get();
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(buf, reply);
}

TEST_P(TcpAsyncAcceptTest, ShutdownCleanupBeforeEventBaseDestroy) {
  auto evbThread = std::make_unique<ScopedEventBaseThread>("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread->getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }

  auto future = server.accept();

  server.shutdown();

  auto conn = future.get();
  EXPECT_EQ(conn, nullptr);

  // Destroying evbThread should exit cleanly — no dangling fd registrations
  evbThread.reset();
}

// ---------------------------------------------------------------------------
// Non-parameterized tests: error cases that are not address-family-specific.
// ---------------------------------------------------------------------------

class TcpAsyncAcceptMiscTest : public ::testing::Test {};

TEST_F(TcpAsyncAcceptMiscTest, AsyncAcceptBeforeInit) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server("127.0.0.1:0", {}, *evbThread.getEventBase());
  // Do NOT call init()

  // Before init, listenSock_ < 0, so accept returns nullptr
  auto conn = server.accept().get();
  EXPECT_EQ(conn, nullptr);
}

// shutdown() must not park forever because the EventBase happens to be busy.
//
// AsyncAccept::shutdown() reaches the loop through two dispatchAndWait() calls,
// and dispatchAndWait() has no timeout: it waits on a condition variable that
// only the loop can notify. So a loop occupied by any other callback holds
// shutdown() for as long as that callback runs, and a loop that stops in the
// window between the isLoopRunning() check and the dispatch holds it forever.
//
// This is not hypothetical load. The TCP transport runs its staged-read-reply
// and H2D poll loops on this same EventBase, and those re-dispatch themselves
// while a copy is outstanding, so teardown competes with the data path for the
// one loop thread. A responder was observed wedged here for 600s during an
// 8-GPU run, having logged the first of its two listener shutdowns and never
// reaching the second.
//
// The occupying callback here stands in for that: it is what a busy loop looks
// like from shutdown()'s perspective, and it makes the wedge deterministic
// rather than a race to lose.
TEST_F(TcpAsyncAcceptMiscTest, ShutdownDoesNotBlockOnABusyEventBase) {
  ScopedEventBaseThread evbThread("async-accept-busy");
  AsyncTcpServer server("127.0.0.1:0", {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }

  // Shared with the queued callback through a shared_ptr rather than captured
  // by reference, for the same reason runOnLoopBounded() does it: the callback
  // can still be unwinding after this frame is gone. evbThread is declared
  // first and so destroyed last, meaning it joins the loop thread only after
  // these locals would already have died -- capturing them by reference is a
  // stack-use-after-scope, which ASAN reports on the wait predicate below.
  struct Occupier {
    std::mutex mu;
    std::condition_variable released;
    bool release{false};
    std::atomic<bool> occupying{false};
  };
  auto occupier = std::make_shared<Occupier>();

  // Occupy the loop thread until this test lets it go.
  evbThread.getEventBase()->dispatch([occupier]() noexcept {
    occupier->occupying.store(true, std::memory_order_release);
    std::unique_lock<std::mutex> lk(occupier->mu);
    occupier->released.wait(lk, [&occupier]() { return occupier->release; });
  });
  ASSERT_TRUE(waitForFlag(occupier->occupying))
      << "the loop never picked up the callback";

  std::atomic<bool> returned{false};
  std::thread shutdownThread([&]() {
    server.shutdown();
    returned.store(true, std::memory_order_release);
  });

  // Must outlast AsyncAccept::kLoopTeardownTimeout (5000ms), not merely match
  // it. With the loop occupied, shutdown() returns only once its first bounded
  // wait expires, so it returns at ~kLoopTeardownTimeout + scheduling overhead.
  // Waiting exactly as long as the code under test makes this a dead heat that
  // fails under ASAN and coin-flips otherwise -- and it fails with a message
  // claiming teardown blocked indefinitely, which is the opposite of what
  // happened. The point of the assertion is "shutdown() returns at all", so the
  // deadline only has to be comfortably clear of the production timeout.
  const bool finished = waitForFlag(returned, std::chrono::seconds(30));

  // Let the loop go regardless, so the test can always join and tear down.
  {
    std::lock_guard<std::mutex> lk(occupier->mu);
    occupier->release = true;
  }
  occupier->released.notify_all();
  shutdownThread.join();

  EXPECT_TRUE(finished)
      << "shutdown() blocked while the EventBase was busy: it waits on the loop "
         "with no timeout, so any other callback -- including the transport's "
         "own poll loops -- can hold teardown indefinitely";
}

INSTANTIATE_TEST_SUITE_P(
    AddrFamilies,
    TcpAsyncAcceptTest,
    ::testing::Values(
        AddrFamily{"127.0.0.1:0", "127.0.0.1", AF_INET},
        AddrFamily{":::0", "::1", AF_INET6}),
    [](const ::testing::TestParamInfo<AddrFamily>& info) {
      return info.param.family == AF_INET ? "IPv4" : "IPv6";
    });
