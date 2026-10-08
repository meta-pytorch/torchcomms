// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <cctype>
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "nccl.h"
#include "os.h"
#include "socket.h"

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 32, 0) && defined(NCCL_OS_LINUX)
#include <net/if.h>
#endif

class SocketSetTosTest : public ::testing::Test {
 public:
  std::string testName;

  SocketSetTosTest() = default;

 protected:
  void SetUp() override {
    const ::testing::TestInfo* test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    testName =
        fmt::format("{}.{}", test_info->test_case_name(), test_info->name());
  }
};

TEST_F(SocketSetTosTest, TestOverrideTOS) {
  const int kExpectedTos = 96;
  SysEnvRAII tosConfigGuard(
      "NCCL_SOCKET_TOS_CONFIG", std::to_string(kExpectedTos));
  ncclCvarInit();
  struct ncclSocket sock{};
  union ncclSocketAddress addr{};
  char bootstrapNetIfName[MAX_IF_NAME_SIZE + 1];
  int numIfs = 0;
  EXPECT_EQ(
      ncclFindInterfaces(
          bootstrapNetIfName, &addr, MAX_IF_NAME_SIZE, 1, &numIfs),
      ncclSuccess);
  EXPECT_GT(numIfs, 0);
  EXPECT_EQ(
      ncclSocketInit(&sock, &addr, 0 /* magic */, ncclSocketTypeBootstrap),
      ncclSuccess);
  int family = sock.addr.sa.sa_family;
  int fd;
  EXPECT_EQ(ncclSocketGetFd(&sock, &fd), ncclSuccess);
  int socketTos = 0;
  socklen_t rlen = sizeof(int);
  if (family == AF_INET6) {
    ASSERT_EQ(getsockopt(fd, IPPROTO_IPV6, IPV6_TCLASS, &socketTos, &rlen), 0);
  } else {
    ASSERT_EQ(getsockopt(fd, IPPROTO_IP, IP_TOS, &socketTos, &rlen), 0);
  }
  EXPECT_EQ(socketTos, kExpectedTos);

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 32, 0)
  ASSERT_EQ(ncclOsSocketResetFd(&sock), ncclSuccess);
  socketTos = 0;
  rlen = sizeof(int);
  if (family == AF_INET6) {
    ASSERT_EQ(getsockopt(fd, IPPROTO_IPV6, IPV6_TCLASS, &socketTos, &rlen), 0);
  } else {
    ASSERT_EQ(getsockopt(fd, IPPROTO_IP, IP_TOS, &socketTos, &rlen), 0);
  }
  EXPECT_EQ(socketTos, kExpectedTos);
#endif

  EXPECT_EQ(ncclSocketClose(&sock), ncclSuccess);
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 32, 0) && defined(NCCL_OS_LINUX)
TEST_F(SocketSetTosTest, TestIpAddressPrefixFiltersInterfaces) {
  EnvRAII<std::string> prefixGuard(NCCL_SOCKET_IPADDR_PREFIX, std::string());
  char ifName[MAX_IF_NAME_SIZE + 1] = {0};
  union ncclSocketAddress addr{};
  int numIfs = 0;
  ASSERT_EQ(
      ncclFindInterfaces(ifName, &addr, MAX_IF_NAME_SIZE, 1, &numIfs),
      ncclSuccess);
  ASSERT_GT(numIfs, 0);

  char numericHost[NI_MAXHOST] = {0};
  const void* address = addr.sa.sa_family == AF_INET
      ? static_cast<const void*>(&addr.sin.sin_addr)
      : static_cast<const void*>(&addr.sin6.sin6_addr);
  ASSERT_NE(
      inet_ntop(addr.sa.sa_family, address, numericHost, sizeof(numericHost)),
      nullptr);

  NCCL_SOCKET_IPADDR_PREFIX = numericHost;
  numIfs = 0;
  EXPECT_EQ(
      ncclFindInterfaces(ifName, &addr, MAX_IF_NAME_SIZE, 1, &numIfs),
      ncclSuccess);
  EXPECT_GT(numIfs, 0);

  NCCL_SOCKET_IPADDR_PREFIX = "not-an-address-prefix";
  numIfs = 0;
  EXPECT_EQ(
      ncclFindInterfaces(ifName, &addr, MAX_IF_NAME_SIZE, 1, &numIfs),
      ncclSuccess);
  EXPECT_EQ(numIfs, 0);
}

TEST_F(SocketSetTosTest, TestIpAddressPrefixRequiresIpv4OctetBoundary) {
  EnvRAII<std::string> prefixGuard(NCCL_SOCKET_IPADDR_PREFIX, std::string());
  char ifNames[MAX_IFS * (MAX_IF_NAME_SIZE + 1)] = {0};
  union ncclSocketAddress addrs[MAX_IFS]{};
  int numIfs = 0;
  ASSERT_EQ(
      ncclOsFindInterfaces(
          "", ifNames, addrs, AF_INET, MAX_IF_NAME_SIZE + 1, MAX_IFS, &numIfs),
      ncclSuccess);
  ASSERT_GT(numIfs, 0);

  char numericHosts[MAX_IFS][NI_MAXHOST] = {};
  for (int i = 0; i < numIfs; ++i) {
    ASSERT_NE(
        inet_ntop(
            AF_INET,
            &addrs[i].sin.sin_addr,
            numericHosts[i],
            sizeof(numericHosts[i])),
        nullptr);
  }

  std::string ambiguousPrefix;
  for (int i = 0; i < numIfs && ambiguousPrefix.empty(); ++i) {
    const std::string host = numericHosts[i];
    for (size_t end = 1; end < host.size(); ++end) {
      if (!std::isdigit(static_cast<unsigned char>(host[end - 1])) ||
          !std::isdigit(static_cast<unsigned char>(host[end]))) {
        continue;
      }
      const std::string candidate = host.substr(0, end);
      bool hasBoundaryMatch = false;
      for (int j = 0; j < numIfs; ++j) {
        const std::string otherHost = numericHosts[j];
        if (otherHost.compare(0, candidate.size(), candidate) == 0 &&
            (otherHost.size() == candidate.size() ||
             otherHost[candidate.size()] == '.')) {
          hasBoundaryMatch = true;
          break;
        }
      }
      if (!hasBoundaryMatch) {
        ambiguousPrefix = candidate;
        break;
      }
    }
  }
  if (ambiguousPrefix.empty()) {
    GTEST_SKIP()
        << "No IPv4 interface exposes an ambiguous partial-octet prefix";
  }

  NCCL_SOCKET_IPADDR_PREFIX = ambiguousPrefix;
  numIfs = 0;
  EXPECT_EQ(
      ncclOsFindInterfaces(
          "", ifNames, addrs, AF_INET, MAX_IF_NAME_SIZE + 1, MAX_IFS, &numIfs),
      ncclSuccess);
  EXPECT_EQ(numIfs, 0);
}

TEST_F(SocketSetTosTest, TestInterfaceBindingSurvivesSocketReset) {
  EnvRAII<std::string> prefixGuard(NCCL_SOCKET_IPADDR_PREFIX, std::string());
  EnvRAII<int> tosGuard(NCCL_SOCKET_TOS_CONFIG, -1);
  char ifName[MAX_IF_NAME_SIZE + 1] = {0};
  union ncclSocketAddress addr{};
  int numIfs = 0;
  ASSERT_EQ(
      ncclFindInterfaces(ifName, &addr, MAX_IF_NAME_SIZE, 1, &numIfs),
      ncclSuccess);
  ASSERT_GT(numIfs, 0);
  EnvRAII<std::string> ifNameGuard(NCCL_CLIENT_SOCKET_IFNAME, ifName);

  if (addr.sa.sa_family == AF_INET6) {
    addr.sin6.sin6_port = htons(1);
  } else {
    addr.sin.sin_port = htons(1);
  }

  struct ncclSocket sock{};
  ASSERT_EQ(
      ncclSocketInit(&sock, &addr, 0, ncclSocketTypeBootstrap, nullptr, 1, 1),
      ncclSuccess);
  ASSERT_EQ(ncclSocketConnect(&sock), ncclSuccess);

  int fd = NCCL_INVALID_SOCKET;
  ASSERT_EQ(ncclSocketGetFd(&sock, &fd), ncclSuccess);
  char boundIfName[IFNAMSIZ] = {0};
  socklen_t len = sizeof(boundIfName);
  ASSERT_EQ(getsockopt(fd, SOL_SOCKET, SO_BINDTODEVICE, boundIfName, &len), 0);
  EXPECT_STREQ(boundIfName, ifName);

  ASSERT_EQ(ncclOsSocketResetFd(&sock), ncclSuccess);
  memset(boundIfName, 0, sizeof(boundIfName));
  len = sizeof(boundIfName);
  ASSERT_EQ(getsockopt(fd, SOL_SOCKET, SO_BINDTODEVICE, boundIfName, &len), 0);
  EXPECT_STREQ(boundIfName, ifName);
  EXPECT_EQ(ncclSocketClose(&sock), ncclSuccess);
}

#endif
