// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/pipes/MultimemHandler.h"
#include "comms/pipes/MultimemNvlTransportDevice.cuh"

namespace comms::pipes {

struct MultimemNvlTransportConfig {
  // Size in bytes of the multicast staging data window.
  std::size_t dataBufferSize{0};

  // Signal slots exposed through signal(), read_signal(), and
  // wait_signal_until().
  uint32_t userSignalCount{1};

  // Signal slots reserved for transport-internal protocols. These are not
  // addressable through the public signal API.
  uint32_t internalSignalCount{0};
};

/**
 * Host-side owner for a copy-based NVL multimem transport.
 *
 * This transport is group-scoped rather than peer-scoped: every rank in the
 * NVL team maps the same multicast object and a private local backing VA.
 * Writers use `multimemData` to broadcast into every rank's `localData` at the
 * same offset; readers consume from their local backing memory after observing
 * a signal. The numeric `multimemData` pointer is a process-local CUDA VA and
 * is not part of the cross-rank protocol.
 *
 * The caller must set the current CUDA device before construction. The
 * transport records that device ordinal and passes it to MultimemHandler; it
 * does not switch CUDA devices internally.
 *
 * `bootstrap` is the global communicator bootstrap. `commRank` is this
 * process's rank in that global communicator. `nvlRankToCommRank` maps each
 * NVLink-local rank to its global communicator rank; the transport uses
 * NVLink-domain bootstrap collectives internally.
 */
class MultimemNvlTransport {
 public:
  MultimemNvlTransport(
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      int commRank,
      std::vector<int> nvlRankToCommRank,
      const MultimemNvlTransportConfig& config);

  // Compatibility constructor for callers that already wrapped `bootstrap` in
  // an NVLink-local adapter. Prefer the global-bootstrap constructor above for
  // new code.
  MultimemNvlTransport(
      int nvlRank,
      int nvlRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultimemNvlTransportConfig& config);

  ~MultimemNvlTransport() = default;

  MultimemNvlTransport(const MultimemNvlTransport&) = delete;
  MultimemNvlTransport& operator=(const MultimemNvlTransport&) = delete;
  MultimemNvlTransport(MultimemNvlTransport&&) = delete;
  MultimemNvlTransport& operator=(MultimemNvlTransport&&) = delete;

  void exchange();

  MultimemNvlTransportDevice getDeviceTransport() const;

  std::size_t getAllocatedDataBufferSize() const;
  std::size_t getAllocatedSignalBufferSize() const;

  // Mirrors NCCL/NVLS enablement: multicast must be supported on cudaDevice
  // and useful only for teams larger than two ranks. The caller should already
  // have made cudaDevice current.
  static bool isEligible(int nRanks, int cudaDevice) {
    return nRanks > 2 && MultimemHandler::isMultimemSupported(cudaDevice);
  }

 private:
  const int commRank_{-1};
  const int nvlRanks_{-1};
  const std::vector<int> nvlRankToCommRank_;
  const int cudaDevice_{-1};
  const MultimemNvlTransportConfig config_;
  bool exchanged_{false};

  std::unique_ptr<MultimemHandler> dataBufferHandler_;
  std::unique_ptr<MultimemHandler> signalBufferHandler_;
};

} // namespace comms::pipes
