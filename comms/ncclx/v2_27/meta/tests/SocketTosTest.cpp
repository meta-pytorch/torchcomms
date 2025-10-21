// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <thread>
#include "comms/testinfra/TestUtils.h"
#include "nccl.h"

class SocketSetTosTest : public ::testing::Test {
 public:
  std::string testName;

  SocketSetTosTest() = default;

 protected:
  void SetUp() override {
    const ::testing::TestInfo* test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    testName =
        folly::sformat("{}.{}", test_info->test_case_name(), test_info->name());
  }
};

TEST_F(SocketSetTosTest, TestOverrideTOS) {
  const int kExpectedTos = 96;
  EnvRAII<int> tosConfigGuard(NCCL_SOCKET_TOS_CONFIG, kExpectedTos);
  struct ncclSocket sock {};
  union ncclSocketAddress addr {};
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
  int socketTos = 0;
  socklen_t rlen = sizeof(int);
  if (family == AF_INET6) {
    getsockopt(sock.fd, IPPROTO_IPV6, IPV6_TCLASS, &socketTos, &rlen);
  } else {
    getsockopt(sock.fd, IPPROTO_IP, IP_TOS, &socketTos, &rlen);
  }
  EXPECT_EQ(socketTos, kExpectedTos);
  EXPECT_EQ(ncclSocketClose(&sock), ncclSuccess);
}
