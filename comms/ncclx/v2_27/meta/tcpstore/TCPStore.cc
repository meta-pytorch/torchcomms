// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "meta/tcpstore/TCPStore.h"
#include "meta/tcpstore/Backoff.h"

#include <fmt/ranges.h>
#include <stdexcept>
#include <thread>
#include "debug.h"

#include "meta/tcpstore/Error.h"
#include "meta/tcpstore/TCPSocket.h"
#include "meta/tcpstore/TCPUtils.h"

namespace ncclx::tcpstore {

namespace detail {

class TCPServer {};

class TCPClient {
 public:
  static std::unique_ptr<TCPClient> connect(
      const SocketAddress& addr,
      const TCPStoreOptions& opts,
      std::shared_ptr<Backoff> backoff);

  void sendRaw(uint8_t* data, size_t length) {
    try {
      utils::sendBytes(socket_.handle(), data, length);
    } catch (const std::exception& e) {
      WARN("sendBytes failed: %s", e.what());
      throw;
    }
  }

  std::vector<std::uint8_t> receiveBits() {
    try {
      return utils::recvVector<std::uint8_t>(socket_.handle());
    } catch (const std::exception& e) {
      WARN("recvVector failed: %s", e.what());
      throw;
    }
  }

  template <typename T>
  T receiveValue() {
    try {
      return utils::recvValue<T>(socket_.handle());
    } catch (const std::exception& e) {
      WARN("recvValue failed on: %s", e.what());
      throw;
    }
  }

  template <typename T>
  bool receiveValueWithTimeout(T& t, std::chrono::milliseconds timeout) {
    if (!socket_.waitForInput(timeout)) {
      return false;
    }
    t = utils::recvValue<T>(socket_.handle());
    return true;
  }

  explicit TCPClient(Socket&& socket) : socket_{std::move(socket)} {}

 private:
  Socket socket_;
};

std::unique_ptr<TCPClient> TCPClient::connect(
    const SocketAddress& addr,
    const TCPStoreOptions& opts,
    std::shared_ptr<Backoff> backoff) {
  Socket socket = Socket::connect(
      addr.host,
      addr.port,
      SocketOptions{}
          .connect_timeout(opts.timeout)
          .connect_backoff(std::move(backoff)));

  return std::make_unique<TCPClient>(std::move(socket));
}

class SendBuffer {
  // ethernet mtu 1500 - 40 (ip v6 header) - 20 (tcp header)
  const size_t FLUSH_WATERMARK = 1440;
  std::vector<uint8_t> buffer;
  detail::TCPClient& client;

  void maybeFlush() {
    if (buffer.size() >= FLUSH_WATERMARK) {
      flush();
    }
  }

 public:
  SendBuffer(detail::TCPClient& client, detail::QueryType cmd)
      : client(client) {
    buffer.reserve(32); // enough for most commands
    buffer.push_back((uint8_t)cmd);
  }

  void appendString(const std::string& str) {
    appendValue<uint64_t>(str.size());
    buffer.insert(buffer.end(), str.begin(), str.end());
    maybeFlush();
  }

  void appendBytes(const std::vector<uint8_t>& vec) {
    appendValue<uint64_t>(vec.size());
    buffer.insert(buffer.end(), vec.begin(), vec.end());
    maybeFlush();
  }

  template <typename T>
  void appendValue(T value) {
    uint8_t* begin = (uint8_t*)&value;
    buffer.insert(buffer.end(), begin, begin + sizeof(T));
    maybeFlush();
  }

  void flush() {
    if (buffer.size() > 0) {
      client.sendRaw(buffer.data(), buffer.size());
      buffer.clear();
    }
  }
};

} // namespace detail

TCPStore::TCPStore(
    const std::string& masterAddr,
    std::uint16_t masterPort,
    std::optional<int> numWorkers,
    bool isServer,
    const std::chrono::milliseconds& timeout,
    bool waitWorkers)
    : TCPStore{
          masterAddr,
          TCPStoreOptions{
              masterPort,
              isServer,
              numWorkers ? std::optional<std::size_t>(*numWorkers)
                         : std::nullopt,
              waitWorkers,
              timeout}} {}

TCPStore::TCPStore(std::string host, const TCPStoreOptions& opts)
    : timeout_(opts.timeout),
      addr_{std::move(host)},
      numWorkers_{opts.numWorkers},
      usingLibUv_{opts.useLibUV} {
  detail::Socket::initialize();

  addr_.port = opts.port;

  // Try connecting several times -- if the server listen backlog is full it may
  // fail on the first send in validate.
  auto deadline = std::chrono::steady_clock::now() + opts.timeout;
  auto backoff = std::make_shared<ExponentialBackoffWithJitter>();

  auto retry = 0;
  do {
    try {
      std::string msg = fmt::format("host {}:{}", addr_.host, addr_.port);

      auto client = detail::TCPClient::connect(addr_, opts, backoff);
      syncClient_ = folly::Synchronized<std::unique_ptr<detail::TCPClient>>(
          std::move(client));
      // TCP connection established
      INFO(NCCL_INIT, "TCP client connected to host %s", msg.c_str());

      // client's first query for validation
      validate();

      ping();

      INFO(NCCL_INIT, "TCP client validation complete");

      // success
      break;
    } catch (const detail::NetworkError& ex) {
      if (deadline < std::chrono::steady_clock::now()) {
        throw std::runtime_error(fmt::format(
            "TCP client failed to connect/validate to host {}:{} - timed out (try={}, timeout={}ms): {}",
            addr_.host,
            addr_.port,
            retry,
            opts.timeout.count(),
            ex.what()));
      }

      auto delayDuration = backoff->nextBackoff();

      std::string msg = fmt::format(
          "TCP client failed to connect/validate to host {}:{} - retrying (try={}, timeout={}ms, delay={}ms): {}",
          addr_.host,
          addr_.port,
          retry,
          opts.timeout.count(),
          delayDuration.count(),
          ex.what());

      INFO(NCCL_INIT, "%s", msg.c_str());

      std::this_thread::sleep_for(delayDuration);
      retry += 1;
    } catch (const std::exception&) {
      throw;
    }
  } while (true);

  if (opts.waitWorkers) {
    waitForWorkers();
  }
}

TCPStore::~TCPStore() = default;

void TCPStore::set(const std::string& key, const std::vector<uint8_t>& data) {
  //   detail::timing_guard tguard(clientCounters_["set"]);
  auto client = syncClient_.wlock();
  detail::SendBuffer buffer(**client, detail::QueryType::SET);
  buffer.appendString(keyPrefix_ + key);
  buffer.appendBytes(data);
  buffer.flush();
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  //   detail::timing_guard tguard(clientCounters_["get"]);
  auto client = syncClient_.wlock();
  return doGet(&(**client), keyPrefix_ + key);
}

int64_t TCPStore::incrementValueBy(const std::string& key, int64_t delta) {
  auto client = syncClient_.wlock();
  detail::SendBuffer buff(**client, detail::QueryType::ADD);
  buff.appendString(key);
  buff.appendValue<std::int64_t>(delta);
  buff.flush();

  return (*client)->receiveValue<std::int64_t>();
}

void TCPStore::doWait(
    detail::TCPClient* client,
    const std::vector<std::string>& keys,
    std::chrono::milliseconds timeout) {
  {
    detail::SendBuffer buffer(*client, detail::QueryType::WAIT);
    buffer.appendValue(keys.size());
    for (const std::string& key : keys) {
      buffer.appendString(key);
    }
    buffer.flush();
  }

  detail::WaitResponseType response;
  if (client->receiveValueWithTimeout<detail::WaitResponseType>(
          response, timeout)) {
    if (response != detail::WaitResponseType::STOP_WAITING) {
      throw std::runtime_error("Stop_waiting response is expected");
    }
    return;
  }
  // this is the cancel wait timeout, once here we expect the server to respond
  // in a timely fashion
  {
    detail::SendBuffer buffer(*client, detail::QueryType::CANCEL_WAIT);
    buffer.flush();
  }

  response = client->receiveValue<detail::WaitResponseType>();
  // this can happen if the server responds before we cancel, just ignore it
  if (response != detail::WaitResponseType::WAIT_CANCELED) {
    if (response != detail::WaitResponseType::STOP_WAITING) {
      throw std::runtime_error("Stop_waiting response is expected");
    }

    response = client->receiveValue<detail::WaitResponseType>(); // ignore
    if (response != detail::WaitResponseType::WAIT_CANCELED) {
      throw std::runtime_error("wait_canceled response is expected");
    }
  }
  throw std::runtime_error(fmt::format(
      "doWait timeout after {}ms, keys: {}",
      timeout.count(),
      fmt::join(keys, ", ")));
}

std::vector<uint8_t> TCPStore::doGet(
    detail::TCPClient* client,
    const std::string& key) {
  doWait(client, {key}, timeout_);
  detail::SendBuffer buffer(*client, detail::QueryType::GET);
  buffer.appendString(key);
  buffer.flush();

  return client->receiveBits();
}

void TCPStore::waitForWorkers() {
  if (numWorkers_ == std::nullopt) {
    return;
  }

  incrementValueBy(initKey_, 1);
}

void TCPStore::validate() {
  auto client = syncClient_.wlock();
  detail::SendBuffer buffer(**client, detail::QueryType::VALIDATE);
  buffer.appendValue<std::uint32_t>(detail::validationMagicNumber);
  buffer.flush();
}

void TCPStore::ping() {
  auto client = syncClient_.wlock();
  detail::SendBuffer buffer(**client, detail::QueryType::PING);

  uint32_t nonce = getpid();
  buffer.appendValue<std::uint32_t>(nonce);
  buffer.flush();

  uint32_t returnedNonce = (*client)->receiveValue<std::uint32_t>();
  if (nonce != returnedNonce) {
    INFO(NCCL_INIT, "Ping failed, invalid nonce returned");
    throw std::runtime_error("Ping failed, invalid nonce returned");
  }
}

} // namespace ncclx::tcpstore
