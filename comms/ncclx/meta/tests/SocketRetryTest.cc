// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <net/if.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <string>
#include <tuple>

#include "nccl.h"
#include "os.h"
#include "socket.h"

namespace {

class SocketRetryTest
    : public testing::TestWithParam<std::tuple<int, bool, int>> {
 protected:
  void SetUp() override {
    const auto [family, async, trafficClass] = GetParam();
    socket_.asyncFlag = async;
    if (family == AF_INET) {
      socket_.addr.sin.sin_family = AF_INET;
      socket_.addr.sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      socket_.salen = sizeof(sockaddr_in);
    } else {
      socket_.addr.sin6.sin6_family = AF_INET6;
      socket_.addr.sin6.sin6_addr = in6addr_loopback;
      socket_.salen = sizeof(sockaddr_in6);
    }
    ASSERT_EQ(ncclOsSocketResetFd(&socket_), ncclSuccess);
    if (trafficClass != 0) {
      ASSERT_EQ(
          setsockopt(
              socket_.socketDescriptor,
              family == AF_INET ? IPPROTO_IP : IPPROTO_IPV6,
              family == AF_INET ? IP_TOS : IPV6_TCLASS,
              &trafficClass,
              sizeof(trafficClass)),
          0);
    }
  }

  void TearDown() override {
    for (const int fd : {socket_.socketDescriptor, peerFd_, acceptedFd_}) {
      if (fd != NCCL_INVALID_SOCKET) {
        close(fd);
      }
    }
  }

  void bindInterface() {
    constexpr char kInterface[] = "lo";
    ASSERT_EQ(
        setsockopt(
            socket_.socketDescriptor,
            SOL_SOCKET,
            SO_BINDTODEVICE,
            kInterface,
            sizeof(kInterface)),
        0);
  }

  void expectOptions(const char* const interface) {
    char actualInterface[IFNAMSIZ] = {};
    socklen_t length = sizeof(actualInterface);
    ASSERT_EQ(
        getsockopt(
            socket_.socketDescriptor,
            SOL_SOCKET,
            SO_BINDTODEVICE,
            actualInterface,
            &length),
        0);
    EXPECT_STREQ(actualInterface, interface);

    int trafficClass = 0;
    length = sizeof(trafficClass);
    const int family = socket_.addr.sa.sa_family;
    ASSERT_EQ(
        getsockopt(
            socket_.socketDescriptor,
            family == AF_INET ? IPPROTO_IP : IPPROTO_IPV6,
            family == AF_INET ? IP_TOS : IPV6_TCLASS,
            &trafficClass,
            &length),
        0);
    // The kernel owns the ECN bits on each connection.
    EXPECT_EQ(trafficClass & 0xfc, std::get<2>(GetParam()));

    const int flags = fcntl(socket_.socketDescriptor, F_GETFL);
    ASSERT_NE(flags, -1);
    EXPECT_EQ(flags & O_NONBLOCK, socket_.asyncFlag ? O_NONBLOCK : 0);
    int noDelay = 0;
    length = sizeof(noDelay);
    ASSERT_EQ(
        getsockopt(
            socket_.socketDescriptor,
            IPPROTO_TCP,
            TCP_NODELAY,
            &noDelay,
            &length),
        0);
    EXPECT_EQ(noDelay, 1);
  }

  static void expectSameFile(const int fd, const struct stat& originalStat) {
    struct stat currentStat{};
    ASSERT_EQ(fstat(fd, &currentStat), 0);
    EXPECT_EQ(currentStat.st_dev, originalStat.st_dev);
    EXPECT_EQ(currentStat.st_ino, originalStat.st_ino);
    EXPECT_EQ(currentStat.st_mode, originalStat.st_mode);
  }

  void reservePeer() {
    // Reserve the port without listening, so connect is refused without a
    // free-port race. The same socket can start listening after retries.
    peerFd_ =
        ::socket(socket_.addr.sa.sa_family, SOCK_STREAM | SOCK_NONBLOCK, 0);
    ASSERT_NE(peerFd_, -1);
    ASSERT_EQ(bind(peerFd_, &socket_.addr.sa, socket_.salen), 0);
    socklen_t length = socket_.salen;
    ASSERT_EQ(getsockname(peerFd_, &socket_.addr.sa, &length), 0);
  }

  ncclResult_t connectAttempt() {
    socket_.state = ncclSocketStateConnecting;
    ncclResult_t result = ncclOsSocketStartConnect(&socket_);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (result == ncclSuccess &&
           socket_.state == ncclSocketStateConnectPolling &&
           std::chrono::steady_clock::now() < deadline) {
      result = ncclOsSocketPollConnect(&socket_);
    }
    EXPECT_NE(socket_.state, ncclSocketStateConnectPolling);
    return result;
  }

  ncclSocket socket_{.socketDescriptor = NCCL_INVALID_SOCKET};
  int peerFd_{NCCL_INVALID_SOCKET};
  int acceptedFd_{NCCL_INVALID_SOCKET};
};

TEST_P(SocketRetryTest, ResetPreservesInterfaceAndFlags) {
  ASSERT_NO_FATAL_FAILURE(bindInterface());
  const int originalFd = socket_.socketDescriptor;
  for (int retry = 0; retry < 3; retry++) {
    ASSERT_EQ(ncclOsSocketResetFd(&socket_), ncclSuccess);
    EXPECT_EQ(socket_.socketDescriptor, originalFd);
    ASSERT_NO_FATAL_FAILURE(expectOptions("lo"));
  }
}

TEST_P(SocketRetryTest, ResetKeepsUnboundSocketUnbound) {
  const int originalFd = socket_.socketDescriptor;
  ASSERT_EQ(ncclOsSocketResetFd(&socket_), ncclSuccess);
  EXPECT_EQ(socket_.socketDescriptor, originalFd);
  ASSERT_NO_FATAL_FAILURE(expectOptions(""));
}

TEST_P(SocketRetryTest, RefusedPeerRecoversWithoutLosingInterface) {
  ASSERT_NO_FATAL_FAILURE(bindInterface());
  ASSERT_NO_FATAL_FAILURE(reservePeer());
  const int originalFd = socket_.socketDescriptor;
  for (int retry = 1; retry <= 3; retry++) {
    ASSERT_EQ(connectAttempt(), ncclSuccess);
    ASSERT_EQ(socket_.state, ncclSocketStateConnecting);
    EXPECT_EQ(socket_.errorRetries, retry);
    EXPECT_EQ(socket_.socketDescriptor, originalFd);
    ASSERT_NO_FATAL_FAILURE(expectOptions("lo"));
  }

  ASSERT_EQ(listen(peerFd_, 1), 0);
  ASSERT_EQ(connectAttempt(), ncclSuccess);
  ASSERT_EQ(socket_.state, ncclSocketStateConnected);
  ASSERT_NO_FATAL_FAILURE(expectOptions("lo"));
  acceptedFd_ = accept(peerFd_, nullptr, nullptr);
  ASSERT_NE(acceptedFd_, -1);

  constexpr char kPayload[] = "connected after retries";
  ASSERT_EQ(
      send(socket_.socketDescriptor, kPayload, sizeof(kPayload), MSG_NOSIGNAL),
      sizeof(kPayload));
  char received[sizeof(kPayload)] = {};
  ASSERT_EQ(
      recv(acceptedFd_, received, sizeof(received), MSG_WAITALL),
      sizeof(received));
  EXPECT_STREQ(received, kPayload);
}

TEST_P(SocketRetryTest, ThreeRetriesExhaustAfterFourAttempts) {
  ASSERT_NO_FATAL_FAILURE(bindInterface());
  ASSERT_NO_FATAL_FAILURE(reservePeer());
  const auto start = std::chrono::steady_clock::now();
  for (int retry = 1; retry <= 3; retry++) {
    ASSERT_EQ(connectAttempt(), ncclSuccess);
    ASSERT_EQ(socket_.state, ncclSocketStateConnecting);
    EXPECT_EQ(socket_.errorRetries, retry);
    ASSERT_NO_FATAL_FAILURE(expectOptions("lo"));
  }
  EXPECT_EQ(connectAttempt(), ncclRemoteError);
  EXPECT_EQ(socket_.state, ncclSocketStateError);
  EXPECT_EQ(socket_.errorRetries, 4);
  // Test targets set 1 ms backoff: the three retries sleep 1 + 2 + 3 ms.
  // Refused loopback peers do not model a blackholed peer's kernel SYN timeout.
  EXPECT_GE(
      std::chrono::steady_clock::now() - start, std::chrono::milliseconds(6));
}

TEST_P(SocketRetryTest, FailedOptionReadLeavesOriginalDescriptorIntact) {
  ASSERT_EQ(close(socket_.socketDescriptor), 0);
  socket_.socketDescriptor = open("/dev/null", O_RDONLY);
  ASSERT_NE(socket_.socketDescriptor, -1);
  const int originalFd = socket_.socketDescriptor;
  struct stat originalStat{};
  ASSERT_EQ(fstat(originalFd, &originalStat), 0);

  EXPECT_EQ(ncclOsSocketResetFd(&socket_), ncclSystemError);
  EXPECT_EQ(socket_.socketDescriptor, originalFd);
  ASSERT_NO_FATAL_FAILURE(expectSameFile(originalFd, originalStat));
}

TEST_P(SocketRetryTest, FailedTrafficClassReadLeavesOriginalDescriptorIntact) {
  ASSERT_EQ(close(socket_.socketDescriptor), 0);
  socket_.socketDescriptor = ::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_NE(socket_.socketDescriptor, -1);
  const int originalFd = socket_.socketDescriptor;
  struct stat originalStat{};
  ASSERT_EQ(fstat(originalFd, &originalStat), 0);

  // The interface read succeeds; the IP traffic-class read must fail.
  char interface[IFNAMSIZ] = {};
  socklen_t length = sizeof(interface);
  ASSERT_EQ(
      getsockopt(originalFd, SOL_SOCKET, SO_BINDTODEVICE, interface, &length),
      0);
  EXPECT_EQ(ncclOsSocketResetFd(&socket_), ncclSystemError);
  EXPECT_EQ(socket_.socketDescriptor, originalFd);
  ASSERT_NO_FATAL_FAILURE(expectSameFile(originalFd, originalStat));
}

INSTANTIATE_TEST_SUITE_P(
    Loopback,
    SocketRetryTest,
    testing::Combine(
        testing::Values(AF_INET, AF_INET6),
        testing::Bool(),
        testing::Values(0, 0xb8)),
    [](const testing::TestParamInfo<SocketRetryTest::ParamType>& info) {
      return std::string(std::get<0>(info.param) == AF_INET ? "IPv4" : "IPv6") +
          (std::get<1>(info.param) ? "Nonblocking" : "Blocking") +
          (std::get<2>(info.param) ? "ConfiguredTrafficClass"
                                   : "DefaultTrafficClass");
    });

} // namespace
