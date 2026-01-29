// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/analyzer/NCCLXCommsTracingServiceUtil.h"

#include <folly/concurrency/AtomicSharedPtr.h>
#include <folly/io/async/Liburing.h>
#include <folly/io/async/SSLOptions.h>
#include <folly/synchronization/CallOnce.h>
#include <thrift/lib/cpp2/server/ThriftServer.h>
#include <thrift/lib/cpp2/util/ScopedServerThread.h>
#include <wangle/ssl/SSLContextConfig.h>

#include "comms/utils/RankUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/analyzer/NCCLXCommsTracingServiceHandler.h"
#include "nccl.h"

namespace ncclx {
namespace {

// Note: We use a Meyers singleton (local static) instead of folly::Singleton
// because library_object_internal gets statically linked into multiple shared
// libraries (e.g., ncclx-api.so, libtorch_cuda.so). When both DSOs are loaded,
// folly's singleton registration happens twice, causing a "Double registration"
// error. The Meyers singleton pattern with a local static variable is
// guaranteed by C++11 to be initialized exactly once in a thread-safe manner.
folly::atomic_shared_ptr<apache::thrift::util::ScopedServerThread>&
getCommsTracingServicePtr() {
  static folly::atomic_shared_ptr<apache::thrift::util::ScopedServerThread>
      instance;
  return instance;
}

bool shouldEnableCommsTracingService() {
  return NCCL_COMM_TRACING_SERVICE_ENABLE;
}

int getCommsTracingServicePort() {
  auto localRank = RankUtils::getLocalRank();
  XCHECK(localRank.has_value()) << "Unable to get local rank info";
  XCHECK(NCCL_COMMS_TRACING_SERVICE_PORTS.size() > localRank.value())
      << "NCCL_COMMS_TRACING_SERVICE_PORTS env var is not enough to cover all ranks";
  return folly::to<int>(NCCL_COMMS_TRACING_SERVICE_PORTS[localRank.value()]);
}

std::unique_ptr<apache::thrift::util::ScopedServerThread> startAndGetService(
    int port) {
  auto server = std::make_unique<apache::thrift::ThriftServer>();
  auto handler = std::make_shared<NCCLXCommsTracingServiceHandler>();
  server->setPort(port);
  server->setInterface(handler);

// Check whether io_uring is available. Please note this does not 100%
// guarantee that io_uring is available in thrift (which is built separately)
// but it is a good indicator.
#ifndef FOLLY_HAS_LIBURING
  static_assert(
      false,
      "FOLLY_HAS_LIBURING is not defined! liburing is required for CommsTracingService to work");
#endif
  // Workaround to avoid using libevent backend for EventBase. Currently
  // folly has bug with libevent backend with libevent 2.1.12.
  server->setPreferIoUring(true);
  server->setUseDefaultIoUringExecutor(true);

  // Use fewer resources to run faster
  server->setNumIOWorkerThreads(1);
  server->setNumCPUWorkerThreads(1);

  // Enable SSL
  auto sslConfig = std::make_shared<wangle::SSLContextConfig>();
  sslConfig->setCertificate(
      "/var/facebook/x509_identities/server.pem",
      "/var/facebook/x509_identities/server.pem",
      "");
  sslConfig->clientCAFiles =
      std::vector<std::string>{"/var/facebook/rootcanal/ca.pem"};
  sslConfig->sslVersion = folly::SSLContext::SSLVersion::TLSv1_3;
  sslConfig->sessionContext = "thrift";
  sslConfig->setNextProtocols(
      **apache::thrift::ThriftServer::defaultNextProtocols());
  sslConfig->sslCiphers =
      folly::join(":", folly::ssl::SSLOptions2021::ciphers());
  server->setSSLConfig(std::move(sslConfig));
  server->setSSLPolicy(apache::thrift::SSLPolicy::PERMITTED);
  // Allow another socket to bind to [::]:port
  // so long as the remote address is different.
  server->setReusePort(true);

  // Allows a socket to bind to [::]:port
  // when a previous connection is in TIME_WAIT state
  // (not likely to happen as this is generally at the very beginning).
  folly::SocketOptionMap options = {{{SOL_SOCKET, SO_REUSEADDR}, 1}};
  server->setSocketOptions(options);

  XLOG(INFO) << handler->getName() << " is requested to run on port: " << port;
  auto serverThread =
      std::make_unique<apache::thrift::util::ScopedServerThread>();
  try {
    serverThread->start(std::move(server));
  } catch (const std::exception& e) {
    // we may still fail even when we set SO_REUSEPORT and SO_REUSEADDR
    // as other sockets may not do so,
    // in which case, we will just check if we should crash.
    const std::string msg(e.what());
    if (msg.find("Address already in use") != std::string::npos &&
        NCCL_COMM_TRACING_SERVICE_WARN_ON_PORT_CONFLICT) {
      XLOG(WARNING) << "Port " << port << " is in use. "
                    << "Tracing service cannot start. We will move on as "
                    << "NCCL_COMM_TRACING_SERVICE_WARN_ON_PORT_CONFLICT is set "
                    << "but analyzer may behave incorrectly.";
      return nullptr;
    }
    throw;
  }
  serverThread->setServeThreadName("CommsTracingService");
  int actualPort = serverThread->getAddress()->getPort();
  XLOG(INFO) << handler->getName() << " running on port: " << actualPort;

  return serverThread;
}

} // namespace

void NCCLXCommsTracingServiceUtil::startService() {
  if (!shouldEnableCommsTracingService()) {
    return;
  }

  auto& servicePtr = getCommsTracingServicePtr();

  // startService() should have been guarded by init logic already
  // to be invoked only once per process https://fburl.com/code/yeb1p8nd
  // but just in case
  if (servicePtr.load()) {
    XLOG(FATAL) << "Already started. Call stopService() before restarting.";
    return;
  }

  auto port = getCommsTracingServicePort();
  auto service = startAndGetService(port);
  servicePtr.store(std::move(service));
}

void NCCLXCommsTracingServiceUtil::stopService() {
  auto& servicePtr = getCommsTracingServicePtr();
  auto service = servicePtr.load();
  if (!service) {
    XLOG(INFO) << "Service is not running";
    return;
  }
  service->stop();
  servicePtr.store(nullptr);
}

int NCCLXCommsTracingServiceUtil::getPort() {
  auto& servicePtr = getCommsTracingServicePtr();
  auto service = servicePtr.load();
  if (!service) {
    XLOG(ERR) << "Service is not running";
    return -1;
  }
  return service->getAddress()->getPort();
}

} // namespace ncclx

__attribute__((visibility("default"))) ncclResult_t
ncclCommsTracingServicePort(int& port) {
  port = ncclx::NCCLXCommsTracingServiceUtil::getPort();
  if (port == -1) {
    return ncclInternalError;
  }

  return ncclSuccess;
}
