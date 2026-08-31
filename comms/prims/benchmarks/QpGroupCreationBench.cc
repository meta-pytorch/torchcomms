// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/*
 * Cost of the DOCA GPU verbs object creation behind createPeerQps(), which is
 * ~75% of per-peer materialization (0.86s of 1.15s measured on GB300). Creates
 * the same shape createPeerQps() does -- one QP group plus one loopback
 * companion QP per channel slot -- and reports per-slot cost.
 *
 * WARNING on --unsafe_threads: creating DOCA objects concurrently against one
 * doca_gpu context is NOT SAFE and this mode exists only to measure what a
 * batched materialization hook could win if it ever became safe. doca_gpu holds
 * an unsynchronized std::unordered_map (doca_gpunetio.hpp, `mtable`) that
 * doca_gpu_mem_alloc inserts into and doca_gpu_mem_free finds/erases from, so
 * concurrent creation corrupts the heap. Reproduced under ASAN as a
 * heap-use-after-free (allocated on one thread, freed and re-read on another)
 * and independently as a SIGSEGV inside jemalloc's own metadata. Serial runs
 * are clean at every scale tested. The threaded numbers are therefore
 * measurements of an unshippable configuration, not a recommendation -- see
 * P2458066730. The default sweep is serial for this reason.
 */

// Include order mirrors MultipeerIbgdaTransport.cc: the transport header pulls
// in Ibvcore.h before <doca_gpunetio_host.h>, which the ibverbx headers below
// need in order to see a complete ibv_context.
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/prims/transport/rdma/NicDiscovery.h"

DEFINE_int32(
    slots,
    0,
    "Measure this many channel slots as a single configuration. 0 runs the "
    "default serial sweep instead.");
DEFINE_int32(
    unsafe_threads,
    0,
    "Create slots across this many threads. UNSAFE: concurrent DOCA object "
    "creation on one doca_gpu context corrupts the heap (see file header). "
    "0 keeps creation serial.");
DEFINE_int32(warmup, 1, "Discard a small first allocation before measuring.");
DEFINE_int32(
    cycles,
    1,
    "Repeat the --slots configuration this many create/destroy rounds, "
    "reporting the running total.");

namespace {

constexpr int kQpDepth = 1024;
constexpr int kLoopbackQpDepth = 8;

struct Slot {
  doca_gpu_verbs_qp_group_hl* group{nullptr};
  doca_gpu_verbs_qp_hl* loopback{nullptr};
};

// One channel slot, matching createPeerQps().
void createSlot(
    doca_gpu_verbs_qp_init_attr_hl mainAttr,
    doca_gpu_verbs_qp_init_attr_hl loopbackAttr,
    Slot& slot) {
  doca_error_t err = doca_gpu_verbs_create_qp_group_hl(&mainAttr, &slot.group);
  if (err != DOCA_SUCCESS) {
    throw std::runtime_error("doca_gpu_verbs_create_qp_group_hl failed");
  }
  err = doca_gpu_verbs_create_qp_hl(&loopbackAttr, &slot.loopback);
  if (err != DOCA_SUCCESS) {
    throw std::runtime_error("doca_gpu_verbs_create_qp_hl failed");
  }
}

void destroySlots(std::vector<Slot>& slots) {
  for (auto& slot : slots) {
    if (slot.group != nullptr) {
      doca_gpu_verbs_destroy_qp_group_hl(slot.group);
      slot.group = nullptr;
    }
    if (slot.loopback != nullptr) {
      doca_gpu_verbs_destroy_qp_hl(slot.loopback);
      slot.loopback = nullptr;
    }
  }
}

double runSerial(
    doca_gpu_verbs_qp_init_attr_hl mainAttr,
    doca_gpu_verbs_qp_init_attr_hl loopbackAttr,
    int totalSlots) {
  std::vector<Slot> slots(totalSlots);
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < totalSlots; ++i) {
    createSlot(mainAttr, loopbackAttr, slots[i]);
  }
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
  destroySlots(slots);
  return ms;
}

double runThreaded(
    doca_gpu_verbs_qp_init_attr_hl mainAttr,
    doca_gpu_verbs_qp_init_attr_hl loopbackAttr,
    int totalSlots,
    int numThreads) {
  std::vector<Slot> slots(totalSlots);
  std::vector<std::thread> workers;
  workers.reserve(numThreads);
  const auto start = std::chrono::steady_clock::now();
  for (int t = 0; t < numThreads; ++t) {
    workers.emplace_back([&, t]() {
      // Each DOCA call needs the CUDA context of the target device.
      const cudaError_t err = cudaSetDevice(0);
      if (err != cudaSuccess) {
        LOG(FATAL) << "cudaSetDevice(0) failed in worker: "
                   << cudaGetErrorString(err);
      }
      for (int i = t; i < totalSlots; i += numThreads) {
        createSlot(mainAttr, loopbackAttr, slots[i]);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
  destroySlots(slots);
  return ms;
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  // Unbuffered: a hang must still show how far it got.
  setvbuf(stdout, nullptr, _IONBF, 0);

  if (cudaSetDevice(0) != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice(0) failed\n");
    return 1;
  }

  const std::string pciBusId =
      comms::prims::GpuNicDiscovery::getCudaPciBusId(0);
  printf("GPU 0 PCIe %s\n", pciBusId.c_str());

  comms::prims::DocaGpu* docaGpu = nullptr;
  doca_error_t err = doca_gpu_create(pciBusId.c_str(), &docaGpu);
  if (err != DOCA_SUCCESS) {
    fprintf(
        stderr,
        "doca_gpu_create failed (err=%d) -- DOCA GPUNetIO is not\n",
        err);
    fprintf(
        stderr, "usable on this host; run this benchmark on GB200/GB300.\n");
    return 2;
  }
  CHECK(docaGpu != nullptr);
  printf("DOCA GPU context created\n");

  auto initResult = ibverbx::ibvInit();
  if (!initResult) {
    fprintf(
        stderr,
        "failed to initialize ibverbx: %s\n",
        initResult.error().errStr.c_str());
    return 3;
  }
  auto& symbols = ibverbx::ibvSymbols;

  int numDevices = 0;
  ibverbx::ibv_device** devices =
      symbols.ibv_internal_get_device_list(&numDevices);
  if (devices == nullptr || numDevices == 0) {
    fprintf(stderr, "no IB devices found\n");
    return 4;
  }
  ibverbx::ibv_context* ctx = symbols.ibv_internal_open_device(devices[0]);
  symbols.ibv_internal_free_device_list(devices);
  if (ctx == nullptr) {
    fprintf(stderr, "ibv_open_device failed\n");
    return 5;
  }
  ibverbx::ibv_pd* pd = symbols.ibv_internal_alloc_pd(ctx);
  if (pd == nullptr) {
    fprintf(stderr, "ibv_alloc_pd failed\n");
    return 6;
  }
  printf("IB device opened (%d available), PD allocated\n", numDevices);

  doca_dev_t* netDev = nullptr;
  err = doca_verbs_dev_open(reinterpret_cast<ibv_pd*>(pd), &netDev);
  if (err != DOCA_SUCCESS || netDev == nullptr) {
    fprintf(stderr, "doca_verbs_dev_open failed (err=%d)\n", err);
    symbols.ibv_internal_dealloc_pd(pd);
    symbols.ibv_internal_close_device(ctx);
    doca_gpu_destroy(docaGpu);
    return 7;
  }

  doca_gpu_verbs_qp_init_attr_hl mainAttr{};
  mainAttr.gpu_dev = docaGpu;
  mainAttr.net_dev = netDev;
  mainAttr.ibpd = reinterpret_cast<decltype(mainAttr.ibpd)>(pd);
  mainAttr.sq_nwqe = kQpDepth;
  mainAttr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
  mainAttr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;

  doca_gpu_verbs_qp_init_attr_hl loopbackAttr = mainAttr;
  loopbackAttr.sq_nwqe = kLoopbackQpDepth;

  if (FLAGS_warmup != 0) {
    // First creation pays one-off driver/context warmup; discard it.
    printf("warmup...\n");
    runSerial(mainAttr, loopbackAttr, 32);
    printf("warmup done\n");
  }

  // Single explicit configuration, for isolating one data point. With
  // --cycles N the same create/destroy round is repeated N times, printing the
  // running total so a stall can be attributed to cumulative churn rather than
  // to the number of simultaneously live objects.
  if (FLAGS_slots > 0) {
    if (FLAGS_unsafe_threads > 0) {
      printf(
          "WARNING: concurrent DOCA object creation on one doca_gpu context is\n"
          "unsafe (unsynchronized mtable); expect heap corruption. Measurement\n"
          "only -- do not treat these numbers as shippable.\n");
    }
    printf(
        "slots=%d threads=%d warmup=%d cycles=%d\n",
        FLAGS_slots,
        FLAGS_unsafe_threads,
        FLAGS_warmup,
        FLAGS_cycles);
    long cumulative = 0;
    for (int cycle = 0; cycle < FLAGS_cycles; ++cycle) {
      const double ms = (FLAGS_unsafe_threads <= 0)
          ? runSerial(mainAttr, loopbackAttr, FLAGS_slots)
          : runThreaded(
                mainAttr, loopbackAttr, FLAGS_slots, FLAGS_unsafe_threads);
      cumulative += FLAGS_slots;
      printf(
          "  cycle %3d  %8.1f ms  (%.2f ms/slot)  cumulative slots=%ld objects=%ld\n",
          cycle,
          ms,
          ms / FLAGS_slots,
          cumulative,
          cumulative * 2);
    }
    doca_verbs_dev_close(netDev);
    symbols.ibv_internal_dealloc_pd(pd);
    symbols.ibv_internal_close_device(ctx);
    doca_gpu_destroy(docaGpu);
    return 0;
  }

  // Default sweep is serial only: the threaded path corrupts the heap (see the
  // file header), so running it must be an explicit opt-in via
  // --unsafe_threads.
  // 64 slots is one peer on one NIC at the default MCCL_MAX_NCHANNELS=32
  // (32 channels x 2 directions), so 128 slots is one full peer.
  printf("\nserial sweep (pass --slots to measure one configuration)\n");
  for (const int slotsPerPeer : {64}) {
    for (const int peers : {1, 2, 4, 8}) {
      const int total = slotsPerPeer * peers;
      const double serialMs = runSerial(mainAttr, loopbackAttr, total);
      printf(
          "  %d x %d = %4d slots  %8.1f ms  (%.2f ms/slot)\n",
          peers,
          slotsPerPeer,
          total,
          serialMs,
          serialMs / total);
    }
  }

  doca_verbs_dev_close(netDev);
  symbols.ibv_internal_dealloc_pd(pd);
  symbols.ibv_internal_close_device(ctx);
  doca_gpu_destroy(docaGpu);
  return 0;
}
