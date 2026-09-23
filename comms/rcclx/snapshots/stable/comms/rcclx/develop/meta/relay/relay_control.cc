/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "meta/relay/relay_control.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "bootstrap.h"
#include "comm.h"
#include "debug.h"
#include "param.h"

namespace rcclx::relay {

namespace {

/**
 * Segment header. Fixed 64 bytes; every field is read by peers, so this is a
 * wire format and its layout is pinned by the static_assert below.
 */
struct Header {
  uint32_t magic;
  uint32_t version;
  uint32_t nRanks;
  uint32_t nActive; // diagnostic only; 0 when not known at init
  uint64_t commHash;
  uint32_t abortReason;
  uint32_t abortRank;
  uint32_t ringDepth;
  uint32_t maxCalls;
  uint32_t creatorPid;
  uint32_t hwmCalls;
  // Stamped last by the creator, with release ordering, so an attacher that
  // sees 1 also sees a fully written header. Without it a rank that wins the
  // race to shm_open an existing-but-unstamped segment reads zeroes and reports
  // a bogus magic mismatch instead of simply waiting.
  uint32_t ready;
  // Which rank owns publishing. There is ONE ring per communicator, so two
  // ranks publishing the same epoch would race the seqlock and drive it
  // backwards -- even though their plans are byte-identical, because both know
  // the same token count. Claimed on first publish and enforced thereafter, so
  // the misuse surfaces as an error instead of as corruption.
  uint32_t publisherRank;
  // Start time of creatorPid, so a RECYCLED pid can be told from the process
  // that recorded it. Without this a stale segment whose pid has been reused by
  // an unrelated process looks live forever, and create() then refuses to build
  // a segment for that commHash until someone removes the shm object by hand.
  // 0 means "could not be determined", in which case the pid alone decides.
  uint64_t creatorStartTime;
};
static_assert(
    sizeof(Header) == 64,
    "Header is a wire format; its size is part of the segment layout");

constexpr size_t kSlotAlign = 64;

// The per-slot prologue: the seqlock word plus the fixed plan record. Counts
// follow it, `maxCalls` of them, which is why the slot stride is a runtime
// value rather than a type's size.
constexpr size_t kSlotPrologue = sizeof(uint64_t) + sizeof(RelayPlanInfo);

size_t alignUp(size_t v, size_t a) {
  return (v + a - 1) & ~(a - 1);
}

size_t slotStride(uint32_t maxCalls) {
  return alignUp(
      kSlotPrologue + static_cast<size_t>(maxCalls) * sizeof(uint64_t),
      kSlotAlign);
}

size_t consumedBytes(uint32_t nRanks) {
  return alignUp(static_cast<size_t>(nRanks) * sizeof(uint64_t), kSlotAlign);
}

int64_t nowNs() {
  struct timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL +
      static_cast<int64_t>(ts.tv_nsec);
}

inline void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield" ::: "memory");
#else
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

// Enough pauses to cover a publisher that is a few hundred nanoseconds ahead of
// us, and no more. In steady state the publisher runs a whole forward ahead, so
// the plan is already there on the first load and none of this budget is spent;
// it exists for cold start and for the first forward after a shape change.
constexpr int kSpinBudget = 256;
constexpr int64_t kBackoffStartNs = 10 * 1000; // 10 us
constexpr int64_t kBackoffMaxNs = 1000 * 1000; // 1 ms

// publisherRank's "nobody has claimed it" value. Not 0, because 0 is a real
// rank and in practice the likely publisher.
constexpr uint32_t kNoPublisher = 0xFFFFFFFFu;

// abortRank before any abort has been attributed. Distinct from a real rank so
// a reader cannot mistake "not yet recorded" for rank 0.
constexpr uint32_t kRelayAbortRankUnknown = 0xFFFFFFFFu;

/**
 * Per-rank consumer progress, encoded in one word so it can be published with a
 * single release store.
 *
 *   0      rank has never entered consume(): NOT a consumer of this segment,
 *          and must never be waited on -- the active ranks attach too, and
 *          waiting on them would deadlock immediately.
 *   1      rank is consuming but has not completed an epoch yet.
 *   k + 2  rank has completed every epoch through k.
 *
 * Registration happens on ENTRY to consume(), not on completion. Completion
 * would leave a consumer that has started but not yet finished epoch 0
 * unprotected, letting the publisher run a full ring ahead and overwrite the
 * slot that consumer is about to read -- which is a realistic race, because a
 * helper does work between consumes.
 */
constexpr uint64_t kConsumerUnregistered = 0;
constexpr uint64_t kConsumerRegistered = 1;

uint64_t consumedEncode(uint64_t epoch) {
  return epoch + 2;
}

// What a consumer must have reached before the slot for `epoch` may be reused.
// The slot was last written by epoch - ringDepth, so that is what must be done.
uint64_t consumedNeededFor(uint64_t epoch, uint32_t ringDepth) {
  return (epoch - ringDepth) + 2;
}

/**
 * Spin briefly, then sleep with exponential backoff, then give up.
 *
 * The give-up is the point. The store this replaces had a wait() timeout, which
 * is the property that kept a helper loop from wedging silently; removing the
 * store removes that, so it has to be put back here.
 */
class BoundedWait {
 public:
  explicit BoundedWait(int64_t timeoutNs)
      : start_(nowNs()), deadline_(start_ + (timeoutNs > 0 ? timeoutNs : 0)) {}

  // False once the budget is exhausted.
  bool step() {
    if (spins_ < kSpinBudget) {
      ++spins_;
      cpuRelax();
      return true;
    }
    const int64_t now = nowNs();
    if (now >= deadline_) {
      return false;
    }
    int64_t sleepNs = std::min(backoffNs_, deadline_ - now);
    struct timespec ts{
        static_cast<time_t>(sleepNs / 1000000000LL),
        static_cast<long>(sleepNs % 1000000000LL)};
    nanosleep(&ts, nullptr);
    backoffNs_ = std::min(backoffNs_ * 2, kBackoffMaxNs);
    return true;
  }

  int64_t elapsedNs() const {
    return nowNs() - start_;
  }

 private:
  int64_t start_;
  int64_t deadline_;
  int64_t backoffNs_{kBackoffStartNs};
  int spins_{0};
};

// Start time of a process in clock ticks since boot, from field 22 of
// /proc/<pid>/stat. 0 if it cannot be read.
//
// Parsed from the LAST ')' rather than by splitting on spaces: field 2 is the
// executable name in parentheses and may itself contain spaces and parens, so
// counting fields from the front misreads for any such process.
uint64_t processStartTime(pid_t pid) {
  char path[64];
  snprintf(path, sizeof(path), "/proc/%d/stat", static_cast<int>(pid));
  FILE* f = fopen(path, "re");
  if (f == nullptr) {
    return 0;
  }
  char buf[1024];
  const size_t got = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  if (got == 0) {
    return 0;
  }
  buf[got] = '\0';
  const char* close = strrchr(buf, ')');
  if (close == nullptr) {
    return 0;
  }
  // Fields 3.. follow the ')'. starttime is field 22, so it is the 20th value
  // after it.
  const char* p = close + 1;
  for (int field = 3; field < 22; field++) {
    while (*p == ' ') {
      p++;
    }
    while (*p != '\0' && *p != ' ') {
      p++;
    }
    if (*p == '\0') {
      return 0;
    }
  }
  while (*p == ' ') {
    p++;
  }
  return strtoull(p, nullptr, 10);
}

// seq is 2*epoch+1 while a slot is in flux and 2*epoch+2 once it is stable, so
// which epoch a seq value refers to depends on its parity. Getting that wrong
// misreports the epoch in exactly the desync diagnostics this is here to make
// legible.
uint64_t epochFromSeq(uint64_t seq) {
  if (seq < 2) {
    return 0;
  }
  return (seq % 2 == 0) ? ((seq - 2) / 2) : ((seq - 1) / 2);
}

// True if a process with this pid exists AND is the same process that recorded
// startTime. Used to decide whether a segment left behind by an earlier run is
// garbage or belongs to something still running -- the difference between
// safely reclaiming it and destroying a live job's state.
//
// The start time is what makes this unambiguous. pids are recycled, so
// kill(pid, 0) alone reports an unrelated process as the live creator.
bool creatorAlive(uint32_t pid, uint64_t startTime) {
  if (pid == 0) {
    return false;
  }
  if (kill(static_cast<pid_t>(pid), 0) != 0 && errno != EPERM) {
    return false;
  }
  if (startTime == 0) {
    return true; // creator could not record one; fall back to pid alone
  }
  const uint64_t current = processStartTime(static_cast<pid_t>(pid));
  // Unreadable (permissions, or it exited between the two checks): treat the
  // pid as authoritative rather than reclaim a segment that may be live.
  return current == 0 || current == startTime;
}

} // namespace

// Defined in sharded_relay_oneshot.cc. Reused rather than redefined so a relay
// user sets one variable, not two, and so the control segment and the one-shot
// region cannot end up half-enabled with respect to each other.
NCCL_PARAM_DECLARE(ShardedRelayModeEnable);

// Calls per published plan. Not a compile-time constant because calls per
// forward is chunk count in the workload this targets -- deployment config, not
// a bounded property -- and about two orders of magnitude larger once attention
// all-to-all is covered. Raising this costs slot bytes, nothing else.
NCCL_PARAM(RelayControlMaxCalls, "RELAY_CONTROL_MAX_CALLS", 128);

// How many forwards the publisher may run ahead. Not load-bearing: the active
// rank self-limits, because its own stream is serialized behind the collectives
// it just enqueued, so it cannot get far ahead regardless.
NCCL_PARAM(RelayControlRingDepth, "RELAY_CONTROL_RING_DEPTH", 4);

uint32_t relayControlConfiguredMaxCalls() {
  const int64_t v = ncclParamRelayControlMaxCalls();
  if (v < 1) {
    return 1;
  }
  if (v > 65536) {
    return 65536;
  }
  return static_cast<uint32_t>(v);
}

uint32_t relayControlConfiguredRingDepth() {
  const int64_t v = ncclParamRelayControlRingDepth();
  // Two is the smallest depth that is a ring at all. Above 1024 the slot array
  // is larger than any plausible run-ahead and almost certainly a typo.
  if (v < 2) {
    return 2;
  }
  if (v > 1024) {
    return 1024;
  }
  return static_cast<uint32_t>(v);
}

size_t RelayControlBlock::segmentBytes(
    uint32_t nRanks,
    uint32_t ringDepth,
    uint32_t maxCalls) {
  return sizeof(Header) + consumedBytes(nRanks) +
      static_cast<size_t>(ringDepth) * slotStride(maxCalls);
}

std::string RelayControlBlock::nameFor(uint64_t commHash) {
  char buf[64];
  snprintf(
      buf,
      sizeof(buf),
      "/rcclx_relay_ctl_%016llx",
      static_cast<unsigned long long>(commHash));
  return std::string(buf);
}

RelayControlBlock::~RelayControlBlock() {
  detach();
}

uint8_t* RelayControlBlock::slotAt(uint64_t epoch) const {
  const size_t stride = slotStride(cfg_.maxCalls);
  const size_t index = static_cast<size_t>(epoch % cfg_.ringDepth);
  return base_ + sizeof(Header) + consumedBytes(cfg_.nRanks) + index * stride;
}

uint64_t* RelayControlBlock::slotSeq(uint64_t epoch) const {
  return reinterpret_cast<uint64_t*>(slotAt(epoch));
}

RelayPlanInfo* RelayControlBlock::slotInfo(uint64_t epoch) const {
  return reinterpret_cast<RelayPlanInfo*>(slotAt(epoch) + sizeof(uint64_t));
}

uint64_t* RelayControlBlock::slotCounts(uint64_t epoch) const {
  return reinterpret_cast<uint64_t*>(slotAt(epoch) + kSlotPrologue);
}

uint64_t* RelayControlBlock::consumedArray() const {
  return reinterpret_cast<uint64_t*>(base_ + sizeof(Header));
}

bool RelayControlBlock::create(const RelayControlConfig& cfg) {
  if (cfg.nRanks == 0 || cfg.ringDepth < 2 || cfg.maxCalls == 0 ||
      cfg.rank >= cfg.nRanks) {
    WARN(
        "Relay control: refusing to create a segment with nRanks=%u rank=%u ringDepth=%u maxCalls=%u",
        cfg.nRanks,
        cfg.rank,
        cfg.ringDepth,
        cfg.maxCalls);
    return false;
  }
  detach();
  cfg_ = cfg;
  name_ = nameFor(cfg.commHash);
  bytes_ = segmentBytes(cfg.nRanks, cfg.ringDepth, cfg.maxCalls);

  int fd = shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0 && errno == EEXIST) {
    // Someone got here first. Either a crashed earlier run left this behind, in
    // which case reclaiming it is right, or a live process owns it, in which
    // case taking it would corrupt that job -- so the creator decides, and we
    // never steal from a live one.
    //
    // An UNSTAMPED header is the trap here. A creator that has just won
    // O_CREAT|O_EXCL but has not yet reached ftruncate/memset/magic presents
    // exactly what an abandoned segment presents: a short read, or a zero
    // magic. Calling that stale would unlink the segment out from under a live
    // creator that still holds a mapping to it. So an unstamped header is
    // retried for a bounded spell, and only declared abandoned if it never gets
    // stamped.
    bool reclaim = false;
    BoundedWait probeWait(200LL * 1000LL * 1000LL);
    for (;;) {
      int probe = shm_open(name_.c_str(), O_RDONLY, 0600);
      if (probe < 0) {
        reclaim = true; // vanished under us; nothing to protect
        break;
      }
      Header probeHdr{};
      const ssize_t got = read(probe, &probeHdr, sizeof(probeHdr));
      close(probe);
      const bool stamped = got == static_cast<ssize_t>(sizeof(probeHdr)) &&
          probeHdr.magic == kRelayControlMagic && probeHdr.ready == 1u;
      if (stamped) {
        reclaim = !creatorAlive(probeHdr.creatorPid, probeHdr.creatorStartTime);
        break;
      }
      if (!probeWait.step()) {
        reclaim = true; // never stamped within the budget: genuinely abandoned
        break;
      }
    }
    if (!reclaim) {
      WARN(
          "Relay control: %s already exists and its creator is still running; not creating a segment for this communicator",
          name_.c_str());
      return false;
    }
    INFO(
        NCCL_INIT, "Relay control: reclaiming stale segment %s", name_.c_str());
    shm_unlink(name_.c_str());
    fd = shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  }
  if (fd < 0) {
    WARN(
        "Relay control: shm_open(%s) for create failed: %s",
        name_.c_str(),
        strerror(errno));
    return false;
  }

  if (ftruncate(fd, static_cast<off_t>(bytes_)) != 0) {
    WARN(
        "Relay control: ftruncate(%s, %zu) failed: %s",
        name_.c_str(),
        bytes_,
        strerror(errno));
    close(fd);
    shm_unlink(name_.c_str());
    return false;
  }

  void* map = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (map == MAP_FAILED) {
    WARN(
        "Relay control: mmap(%s, %zu) failed: %s",
        name_.c_str(),
        bytes_,
        strerror(errno));
    shm_unlink(name_.c_str());
    return false;
  }
  base_ = static_cast<uint8_t*>(map);
  owner_ = true;

  // Zero first: it is what makes every slot's seq 0, i.e. "nothing published
  // yet", and every consumed entry 0, i.e. "no rank has registered as a
  // consumer".
  std::memset(base_, 0, bytes_);

  Header* h = reinterpret_cast<Header*>(base_);
  h->version = kRelayControlVersion;
  h->nRanks = cfg.nRanks;
  h->nActive = cfg.nActive;
  h->commHash = cfg.commHash;
  h->ringDepth = cfg.ringDepth;
  h->maxCalls = cfg.maxCalls;
  h->creatorPid = static_cast<uint32_t>(getpid());
  h->creatorStartTime = processStartTime(getpid());
  h->publisherRank = kNoPublisher;
  // No abort has been attributed yet. A sentinel rather than 0 so a reader that
  // catches the window between the abort reason landing and its rank being
  // stored reports "unattributed" instead of blaming rank 0.
  h->abortRank = kRelayAbortRankUnknown;
  // Magic before ready, both with release ordering, so nothing can observe
  // ready=1 over a header that is not yet valid.
  __atomic_store_n(&h->magic, kRelayControlMagic, __ATOMIC_RELEASE);
  __atomic_store_n(&h->ready, 1u, __ATOMIC_RELEASE);
  return true;
}

bool RelayControlBlock::attach(const RelayControlConfig& cfg) {
  detach();
  if (cfg.nRanks == 0 || cfg.rank >= cfg.nRanks) {
    // consume() indexes consumedArray()[cfg_.rank], which is only nRanks long,
    // so an out-of-range rank would write past it into the first slot's seqlock
    // and plan area and corrupt the ring for every peer. Rejected here (and in
    // create()) so it cannot reach that point at all.
    WARN(
        "Relay control: refusing to attach with rank=%u of nRanks=%u",
        cfg.rank,
        cfg.nRanks);
    return false;
  }
  cfg_ = cfg;
  name_ = nameFor(cfg.commHash);
  const size_t want = segmentBytes(cfg.nRanks, cfg.ringDepth, cfg.maxCalls);

  int fd = shm_open(name_.c_str(), O_RDWR, 0600);
  if (fd < 0) {
    WARN(
        "Relay control: shm_open(%s) for attach failed: %s",
        name_.c_str(),
        strerror(errno));
    return false;
  }
  struct stat st{};
  if (fstat(fd, &st) != 0) {
    WARN("Relay control: fstat(%s) failed: %s", name_.c_str(), strerror(errno));
    close(fd);
    return false;
  }
  const size_t actual = static_cast<size_t>(st.st_size);
  if (actual < sizeof(Header)) {
    WARN(
        "Relay control: %s is %zu bytes, too small to hold a header",
        name_.c_str(),
        actual);
    close(fd);
    return false;
  }
  // Map what is actually there rather than what we expected, so a geometry
  // mismatch can be reported from the header with both values instead of
  // showing up as an unexplained size error.
  void* map = mmap(nullptr, actual, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (map == MAP_FAILED) {
    WARN(
        "Relay control: mmap(%s, %zu) failed: %s",
        name_.c_str(),
        actual,
        strerror(errno));
    return false;
  }
  base_ = static_cast<uint8_t*>(map);
  bytes_ = actual;
  owner_ = false;

  Header* h = reinterpret_cast<Header*>(base_);

  // The creator may have won the shm_open race but not yet stamped the header.
  // Bounded, and short: the caller has already been through a bootstrap
  // all-gather with the creator, so this is a formality rather than a real
  // wait.
  BoundedWait ready(1000LL * 1000LL * 1000LL);
  while (__atomic_load_n(&h->ready, __ATOMIC_ACQUIRE) != 1u) {
    if (!ready.step()) {
      WARN(
          "Relay control: %s was never marked ready by its creator",
          name_.c_str());
      detach();
      return false;
    }
  }

  const uint32_t magic = __atomic_load_n(&h->magic, __ATOMIC_ACQUIRE);
  if (magic != kRelayControlMagic) {
    WARN(
        "Relay control: %s has magic %08x, expected %08x",
        name_.c_str(),
        magic,
        kRelayControlMagic);
    detach();
    return false;
  }
  if (h->version != kRelayControlVersion) {
    WARN(
        "Relay control: %s has version %u but this build speaks version %u; the ranks of this job are not running the same librccl",
        name_.c_str(),
        h->version,
        kRelayControlVersion);
    detach();
    return false;
  }
  if (h->commHash != cfg.commHash) {
    WARN(
        "Relay control: %s belongs to commHash %llx, not %llx",
        name_.c_str(),
        static_cast<unsigned long long>(h->commHash),
        static_cast<unsigned long long>(cfg.commHash));
    detach();
    return false;
  }
  // Geometry comes from environment parameters, so a disagreement here means
  // the ranks of this job were launched with different settings. Failing at
  // init is the entire point of recording it: the alternative is reading a
  // differently-shaped slot at runtime.
  if (h->nRanks != cfg.nRanks || h->ringDepth != cfg.ringDepth ||
      h->maxCalls != cfg.maxCalls || actual != want) {
    WARN(
        "Relay control: %s geometry is nRanks=%u ringDepth=%u maxCalls=%u (%zu bytes) but this rank expects nRanks=%u ringDepth=%u maxCalls=%u (%zu bytes); check that NCCL_RELAY_CONTROL_MAX_CALLS and NCCL_RELAY_CONTROL_RING_DEPTH are set identically on every rank",
        name_.c_str(),
        h->nRanks,
        h->ringDepth,
        h->maxCalls,
        actual,
        cfg.nRanks,
        cfg.ringDepth,
        cfg.maxCalls,
        want);
    detach();
    return false;
  }
  if (!creatorAlive(h->creatorPid, h->creatorStartTime)) {
    WARN(
        "Relay control: %s was created by pid %u, which is gone",
        name_.c_str(),
        h->creatorPid);
    detach();
    return false;
  }
  return true;
}

void RelayControlBlock::detach() {
  if (base_ != nullptr) {
    munmap(base_, bytes_);
    base_ = nullptr;
  }
  if (owner_ && !name_.empty()) {
    shm_unlink(name_.c_str());
  }
  owner_ = false;
  bytes_ = 0;
  name_.clear();
}

uint32_t RelayControlBlock::ringDepth() const {
  return cfg_.ringDepth;
}

uint32_t RelayControlBlock::maxCalls() const {
  return cfg_.maxCalls;
}

uint32_t RelayControlBlock::abortReason() const {
  if (!valid()) {
    return kRelayAbortNone;
  }
  return __atomic_load_n(
      &reinterpret_cast<Header*>(base_)->abortReason, __ATOMIC_ACQUIRE);
}

uint32_t RelayControlBlock::abortRank() const {
  if (!valid()) {
    return 0;
  }
  return __atomic_load_n(
      &reinterpret_cast<Header*>(base_)->abortRank, __ATOMIC_ACQUIRE);
}

uint32_t RelayControlBlock::highWaterCalls() const {
  if (!valid()) {
    return 0;
  }
  return __atomic_load_n(
      &reinterpret_cast<Header*>(base_)->hwmCalls, __ATOMIC_RELAXED);
}

uint64_t RelayControlBlock::consumerProgress(uint32_t rank) const {
  if (!valid() || rank >= cfg_.nRanks) {
    return 0;
  }
  return __atomic_load_n(&consumedArray()[rank], __ATOMIC_ACQUIRE);
}

void RelayControlBlock::setAbort(uint32_t reason) {
  if (!valid() || reason == kRelayAbortNone) {
    return;
  }
  Header* h = reinterpret_cast<Header*>(base_);
  // First abort wins, and ONLY the winner attributes itself. Storing the rank
  // before the exchange would overwrite the first abort's rank every time a
  // later one lost the race, so the segment would report the first reason
  // attributed to an unrelated rank that failed long afterwards.
  uint32_t expected = kRelayAbortNone;
  if (__atomic_compare_exchange_n(
          &h->abortReason,
          &expected,
          reason,
          false,
          __ATOMIC_RELEASE,
          __ATOMIC_RELAXED)) {
    __atomic_store_n(&h->abortRank, cfg_.rank, __ATOMIC_RELEASE);
  }
}

bool RelayControlBlock::waitForSlotDrain(uint64_t epoch, int64_t timeoutNs) {
  // The slot this epoch lands on was last used by epoch - ringDepth. Before
  // that many forwards have happened there is nothing to drain.
  if (epoch < cfg_.ringDepth) {
    return true;
  }
  const uint64_t need = consumedNeededFor(epoch, cfg_.ringDepth);
  const uint64_t* consumed = consumedArray();
  BoundedWait wait(timeoutNs);
  for (;;) {
    uint32_t lagging = UINT32_MAX;
    for (uint32_t r = 0; r < cfg_.nRanks; r++) {
      if (r == cfg_.rank) {
        continue; // the publisher does not consume its own plans
      }
      const uint64_t v = __atomic_load_n(&consumed[r], __ATOMIC_ACQUIRE);
      if (v == kConsumerUnregistered) {
        continue;
      }
      if (v < need) {
        lagging = r;
        break;
      }
    }
    if (lagging == UINT32_MAX) {
      return true;
    }
    const uint32_t ab = abortReason();
    if (ab != kRelayAbortNone) {
      WARN(
          "Relay control: rank %u cannot publish epoch %llu because rank %u aborted (reason %u)",
          cfg_.rank,
          static_cast<unsigned long long>(epoch),
          abortRank(),
          ab);
      return false;
    }
    if (!wait.step()) {
      const uint64_t v = __atomic_load_n(&consumed[lagging], __ATOMIC_ACQUIRE);
      WARN(
          "Relay control: rank %u timed out after %lld ms publishing epoch %llu; rank %u has %s and the ring is %u deep",
          cfg_.rank,
          static_cast<long long>(wait.elapsedNs() / 1000000LL),
          static_cast<unsigned long long>(epoch),
          lagging,
          v == kConsumerRegistered ? "not completed any epoch yet"
                                   : "fallen more than a ring behind",
          cfg_.ringDepth);
      setAbort(kRelayAbortTimeout);
      return false;
    }
  }
}

ncclResult_t RelayControlBlock::publish(
    uint64_t epoch,
    const RelayPlanInfo& info,
    const size_t* counts,
    int64_t timeoutNs) {
  if (!valid()) {
    return ncclInvalidArgument;
  }
  if (info.flags != 0) {
    WARN("Relay control: plan flags must be 0, got %u", info.flags);
    return ncclInvalidArgument;
  }
  if (info.opCode >= kRelayOpCount) {
    WARN("Relay control: plan opCode %u is out of range", info.opCode);
    return ncclInvalidArgument;
  }
  if (info.nCalls > cfg_.maxCalls) {
    WARN(
        "Relay control: plan for epoch %llu names %u calls but this segment holds %u; raise NCCL_RELAY_CONTROL_MAX_CALLS on every rank",
        static_cast<unsigned long long>(epoch),
        info.nCalls,
        cfg_.maxCalls);
    return ncclInvalidArgument;
  }
  if (info.nCalls > 0 && counts == nullptr) {
    WARN("Relay control: plan names %u calls but counts is null", info.nCalls);
    return ncclInvalidArgument;
  }

  // Unconditional, unlike the check inside waitForSlotDrain's wait loop, which
  // is never reached when the slot is already drained (epoch < ringDepth, or
  // every consumer current). Without this a publisher keeps filling slots on a
  // poisoned segment that no consumer will ever read again; consume() checks
  // the flag on every iteration, so this is what makes the poison symmetric.
  const uint32_t aborted = abortReason();
  if (aborted != kRelayAbortNone) {
    WARN(
        "Relay control: rank %u not publishing epoch %llu because rank %u aborted (reason %u)",
        cfg_.rank,
        static_cast<unsigned long long>(epoch),
        abortRank(),
        aborted);
    return ncclInternalError;
  }

  // One ring, so one publisher. Claim it on the first publish and hold every
  // later publish to it: two ranks writing the same slot would drive the
  // seqlock backwards, and because their plans are byte-identical the damage
  // would be invisible in the data and show up only as a spurious desync.
  Header* hdr = reinterpret_cast<Header*>(base_);
  uint32_t owner = kNoPublisher;
  if (!__atomic_compare_exchange_n(
          &hdr->publisherRank,
          &owner,
          cfg_.rank,
          false,
          __ATOMIC_RELEASE,
          __ATOMIC_ACQUIRE) &&
      owner != cfg_.rank) {
    WARN(
        "Relay control: rank %u tried to publish epoch %llu but rank %u already owns publishing on this communicator; exactly one rank may publish",
        cfg_.rank,
        static_cast<unsigned long long>(epoch),
        owner);
    return ncclInvalidArgument;
  }

  if (!waitForSlotDrain(epoch, timeoutNs)) {
    return ncclInternalError;
  }

  uint64_t* seq = slotSeq(epoch);

  // Epochs must advance. The slot's own seqlock word is the record of that: a
  // slot last written by epoch - ringDepth holds 2*(epoch-ringDepth)+2, which
  // is below 2*epoch+2, so anything at or above that means this epoch -- or a
  // later one landing on the same slot -- has already been published.
  //
  // Two things go wrong without the check. Re-publishing an epoch drives the
  // seqlock BACKWARDS from 2e+2 to 2e+1, which a concurrent reader observes as
  // a tear on a slot nobody is actually tearing. And a caller that restarts its
  // epoch counter at 0 walks back over slots holding plans no consumer has
  // taken: waitForSlotDrain returns immediately for any epoch < ringDepth, so
  // the overwrite is silent. The class already enforces a single publisher for
  // this reason; this closes the other half of that invariant.
  const uint64_t published = __atomic_load_n(seq, __ATOMIC_ACQUIRE);
  if (published >= 2 * epoch + 2) {
    WARN(
        "Relay control: rank %u tried to publish epoch %llu but its slot already holds epoch %llu; epochs must advance",
        cfg_.rank,
        static_cast<unsigned long long>(epoch),
        static_cast<unsigned long long>(epochFromSeq(published)));
    return ncclInvalidArgument;
  }

  // Seqlock write. The odd store marks the slot in flux; the release FENCE
  // after it is what stops the body stores below from becoming visible before
  // it, which is what makes a reader's re-read of seq able to detect a tear.
  __atomic_store_n(seq, 2 * epoch + 1, __ATOMIC_RELAXED);
  __atomic_thread_fence(__ATOMIC_RELEASE);

  *slotInfo(epoch) = info;
  uint64_t* dst = slotCounts(epoch);
  for (uint32_t i = 0; i < info.nCalls; i++) {
    dst[i] = static_cast<uint64_t>(counts[i]);
  }

  // Release: everything above is visible to anyone who observes this value.
  __atomic_store_n(seq, 2 * epoch + 2, __ATOMIC_RELEASE);

  uint32_t hwm = __atomic_load_n(&hdr->hwmCalls, __ATOMIC_RELAXED);
  while (info.nCalls > hwm) {
    if (__atomic_compare_exchange_n(
            &hdr->hwmCalls,
            &hwm,
            info.nCalls,
            false,
            __ATOMIC_RELAXED,
            __ATOMIC_RELAXED)) {
      break;
    }
  }
  return ncclSuccess;
}

ncclResult_t RelayControlBlock::publishShutdown(
    uint64_t epoch,
    int64_t timeoutNs) {
  RelayPlanInfo info{};
  info.nCalls = 0;
  info.opCode = kRelayOpShutdown;
  return publish(epoch, info, nullptr, timeoutNs);
}

ncclResult_t RelayControlBlock::consume(
    uint64_t epoch,
    RelayPlanInfo* info,
    size_t* counts,
    uint32_t countsCapacity,
    int64_t timeoutNs) {
  if (!valid() || info == nullptr) {
    return ncclInvalidArgument;
  }
  if (countsCapacity > 0 && counts == nullptr) {
    return ncclInvalidArgument;
  }
  const uint64_t want = 2 * epoch + 2;
  const uint64_t* seq = slotSeq(epoch);
  BoundedWait wait(timeoutNs);

  // Register before waiting, so a publisher cannot run a full ring ahead while
  // we are still working on our first epoch. Only ever moves 0 -> 1; a consumer
  // that has real progress recorded must not be rewound.
  uint64_t* myProgress = &consumedArray()[cfg_.rank];
  uint64_t unregistered = kConsumerUnregistered;
  __atomic_compare_exchange_n(
      myProgress,
      &unregistered,
      kConsumerRegistered,
      false,
      __ATOMIC_RELEASE,
      __ATOMIC_RELAXED);

  for (;;) {
    const uint32_t ab = abortReason();
    if (ab != kRelayAbortNone) {
      WARN(
          "Relay control: rank %u stopping at epoch %llu because rank %u aborted (reason %u)",
          cfg_.rank,
          static_cast<unsigned long long>(epoch),
          abortRank(),
          ab);
      return ncclInternalError;
    }

    const uint64_t s1 = __atomic_load_n(seq, __ATOMIC_ACQUIRE);
    // Both retry paths below fall through to wait.step() rather than looping
    // straight back. Spinning without it would burn a core indefinitely and,
    // far worse, silently drop the bounded-wait guarantee this class exists to
    // keep: a corrupted or geometry-mismatched slot whose seq happens to match
    // `want` never resolves, so the loop would never time out either.
    bool torn = false;
    RelayPlanInfo local{};
    if (s1 == want) {
      local = *slotInfo(epoch);
      // A published plan can never exceed the segment's capacity, so a value
      // that does means we read a slot mid-write. Retry rather than trust it.
      if (local.nCalls > cfg_.maxCalls) {
        torn = true;
      }
    }
    if (s1 == want && !torn) {
      const uint64_t* src = slotCounts(epoch);
      const uint32_t copy = std::min(local.nCalls, countsCapacity);
      for (uint32_t i = 0; i < copy; i++) {
        counts[i] = static_cast<size_t>(src[i]);
      }
      // Acquire fence, then a relaxed re-read: the fence keeps the body reads
      // above from being reordered after this load, which is what makes the
      // comparison meaningful.
      __atomic_thread_fence(__ATOMIC_ACQUIRE);
      if (__atomic_load_n(seq, __ATOMIC_RELAXED) != s1) {
        torn = true;
      }
    }
    if (s1 == want && !torn) {
      *info = local;
      if (local.nCalls > countsCapacity) {
        WARN(
            "Relay control: plan for epoch %llu names %u calls but the caller's buffer holds %u",
            static_cast<unsigned long long>(epoch),
            local.nCalls,
            countsCapacity);
        // Deliberately NOT marked consumed: the plan was not taken, so the
        // publisher must keep treating this slot as occupied.
        return ncclInvalidArgument;
      }
      // Releases the slot ringDepth forwards from now.
      __atomic_store_n(myProgress, consumedEncode(epoch), __ATOMIC_RELEASE);
      return ncclSuccess;
    }

    if (!torn && s1 > want) {
      // The slot has moved a full ring past us. No amount of waiting brings
      // that epoch back, so this is a desync, not slowness.
      WARN(
          "Relay control: rank %u wanted epoch %llu but its slot has already advanced to epoch %llu; the ranks of this communicator are out of step",
          cfg_.rank,
          static_cast<unsigned long long>(epoch),
          static_cast<unsigned long long>(epochFromSeq(s1)));
      return ncclInternalError;
    }

    if (!wait.step()) {
      WARN(
          "Relay control: rank %u timed out after %lld ms waiting for epoch %llu (slot last held epoch %llu)",
          cfg_.rank,
          static_cast<long long>(wait.elapsedNs() / 1000000LL),
          static_cast<unsigned long long>(epoch),
          static_cast<unsigned long long>(epochFromSeq(s1)));
      setAbort(kRelayAbortTimeout);
      return ncclInternalError;
    }
  }
}

namespace {

struct CommEntry {
  // Guards THIS entry's setup. Per-comm rather than process-global because
  // setupForComm() runs two bootstrap all-gathers: holding one global lock
  // across a collective deadlocks as soon as two communicators are initialized
  // concurrently and their peer processes reach the collectives in opposite
  // order -- each process would sit in one comm's all-gather holding the lock
  // the peer it is waiting for needs to enter the other. entryMutex() therefore
  // guards only the map.
  std::mutex mu;
  // Sticky: a retry would have to be collectively agreed, and a rank retrying
  // alone would enter a bootstrap all-gather the others are not in.
  bool tried{false};
  // The map is keyed on the comm POINTER, which the allocator recycles.
  // commHash distinguishes a recycled address from the comm that used to live
  // there, so a missed release cannot be mistaken for a live segment.
  uint64_t commHash{0};
  // shared_ptr, not unique_ptr, so publish/consume can hold the segment mapped
  // for the duration of a call without holding entryMutex() across it.
  std::shared_ptr<RelayControlBlock> block;
};

std::mutex& entryMutex() {
  static std::mutex m;
  return m;
}

std::map<const void*, CommEntry>& entries() {
  static std::map<const void*, CommEntry> e;
  return e;
}

/**
 * Create-or-attach for one communicator.
 *
 * ALWAYS runs both bootstrap all-gathers, even after a local failure, because
 * they are collective: an early return on one rank desynchronizes the bootstrap
 * for every other rank.
 */
std::shared_ptr<RelayControlBlock> setupForComm(ncclComm_t comm) {
  RelayControlConfig cfg;
  cfg.nRanks = static_cast<uint32_t>(comm->nRanks);
  cfg.rank = static_cast<uint32_t>(comm->rank);
  cfg.nActive = 0; // not known at init; diagnostic only
  cfg.commHash = comm->commHash;
  cfg.ringDepth = relayControlConfiguredRingDepth();
  cfg.maxCalls = relayControlConfiguredMaxCalls();

  auto block = std::make_shared<RelayControlBlock>();
  bool ok = true;
  if (comm->rank == 0) {
    ok = block->create(cfg);
  }

  // Barrier plus creator status in one exchange: nobody may attach before the
  // segment exists, and if the creator failed there is nothing to attach to.
  std::vector<uint8_t> created(comm->nRanks, 0);
  created[comm->rank] = ok ? 1 : 0;
  if (bootstrapAllGather(comm->bootstrap, created.data(), sizeof(uint8_t)) !=
      ncclSuccess) {
    ok = false;
  } else if (created[0] == 0) {
    ok = false;
  }

  if (ok && comm->rank != 0) {
    ok = block->attach(cfg);
  }

  // Unanimity. A segment only some ranks have is worse than none: the ranks
  // that have it wait for peers that never publish or consume, which hangs
  // instead of degrading. Same rule, same reason, as the one-shot IPC region.
  std::vector<uint8_t> votes(comm->nRanks, 0);
  votes[comm->rank] = ok ? 1 : 0;
  if (bootstrapAllGather(comm->bootstrap, votes.data(), sizeof(uint8_t)) !=
      ncclSuccess) {
    ok = false;
  } else {
    for (int r = 0; r < comm->nRanks; r++) {
      if (votes[r] == 0) {
        ok = false;
        break;
      }
    }
  }

  if (!ok) {
    block->detach();
    return nullptr;
  }
  INFO(
      NCCL_INIT,
      "Relay control: rank %d attached %s (%u ranks, ring %u, %u calls/plan, %zu bytes)",
      comm->rank,
      RelayControlBlock::nameFor(cfg.commHash).c_str(),
      cfg.nRanks,
      cfg.ringDepth,
      cfg.maxCalls,
      RelayControlBlock::segmentBytes(cfg.nRanks, cfg.ringDepth, cfg.maxCalls));
  return block;
}

// Caller must hold entryMutex(). Returns a REFERENCE-COUNTED handle so the
// caller can drop the lock and still be sure the segment stays mapped for the
// duration of its call, even if a concurrent release erases the entry.
std::shared_ptr<RelayControlBlock> lookupLocked(ncclComm_t comm) {
  if (comm == nullptr) {
    return nullptr;
  }
  auto it = entries().find(static_cast<const void*>(comm));
  if (it == entries().end() || it->second.commHash != comm->commHash) {
    return nullptr;
  }
  const std::shared_ptr<RelayControlBlock>& b = it->second.block;
  return (b != nullptr && b->valid()) ? b : nullptr;
}

} // namespace

void relayControlInit(ncclComm_t comm) {
  if (comm == nullptr || ncclParamShardedRelayModeEnable() != 1) {
    return;
  }
  if (comm->nRanks < 2) {
    return;
  }
  // entryMutex() covers only the map -- the staleness check and the lookup. It
  // is deliberately NOT held across setupForComm(), which runs two bootstrap
  // all-gathers; see CommEntry::mu.
  CommEntry* ep = nullptr;
  {
    std::lock_guard<std::mutex> lock(entryMutex());
    auto it = entries().find(static_cast<const void*>(comm));
    if (it != entries().end() && it->second.tried &&
        it->second.commHash != comm->commHash) {
      // A stale entry can only exist if this comm's release was missed and the
      // allocator handed its address to a new comm. Every rank of the new comm
      // sees the same mismatch, because commHash is agreed across it. The whole
      // node goes, mutex included -- safe because the comm it belonged to is
      // gone, so nobody can be holding that mutex.
      entries().erase(it);
    }
    ep = &entries()[static_cast<const void*>(comm)];
  }
  // std::map is node based, so the reference stays valid once the map lock is
  // dropped; only erasing this key invalidates it, and that happens here or in
  // relayControlRelease(), both only for a comm that is done with.
  CommEntry& e = *ep;

  std::lock_guard<std::mutex> lock(e.mu);
  if (e.tried) {
    return;
  }
  e.tried = true;
  e.commHash = comm->commHash;
  e.block = setupForComm(comm);
}

void relayControlRelease(ncclComm_t comm) {
  if (comm == nullptr) {
    return;
  }
  // Erasing the node destroys CommEntry::mu, so this must not run concurrently
  // with an init/publish/consume for the same comm -- it is called from comm
  // teardown, after the last collective on that comm. Any publish or consume
  // still in flight holds its own shared_ptr, so the segment outlives the
  // erase.
  std::lock_guard<std::mutex> lock(entryMutex());
  auto it = entries().find(static_cast<const void*>(comm));
  if (it == entries().end()) {
    return;
  }
  if (it->second.block != nullptr && it->second.block->valid()) {
    // The one number worth reporting: it turns the capacity parameter from a
    // guess into an observation.
    INFO(
        NCCL_INIT,
        "Relay control: rank %d releasing segment, high-water mark %u calls per plan (capacity %u)",
        comm->rank,
        it->second.block->highWaterCalls(),
        it->second.block->maxCalls());
  }
  entries().erase(it);
}

bool relayControlReady(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(entryMutex());
  return lookupLocked(comm) != nullptr;
}

// entryMutex() is taken only to resolve the comm to a segment, never held
// across publish() or consume(). Both can block for the whole timeoutNs, and
// the mutex is process-wide: holding it there stalls relayControlInit,
// relayControlRelease, relayControlReady and every other communicator's publish
// and consume for that entire budget. The shared_ptr is what the lock used to
// be providing -- it keeps the segment mapped for the duration of the call even
// if a concurrent commFree() erases the entry.
static std::shared_ptr<RelayControlBlock> resolve(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(entryMutex());
  return lookupLocked(comm);
}

ncclResult_t relayControlPublish(
    ncclComm_t comm,
    uint64_t epoch,
    const RelayPlanInfo& info,
    const size_t* counts,
    int64_t timeoutNs) {
  const std::shared_ptr<RelayControlBlock> b = resolve(comm);
  if (b == nullptr) {
    return ncclInvalidArgument;
  }
  return b->publish(epoch, info, counts, timeoutNs);
}

ncclResult_t relayControlConsume(
    ncclComm_t comm,
    uint64_t epoch,
    RelayPlanInfo* info,
    size_t* counts,
    uint32_t countsCapacity,
    int64_t timeoutNs) {
  const std::shared_ptr<RelayControlBlock> b = resolve(comm);
  if (b == nullptr) {
    return ncclInvalidArgument;
  }
  return b->consume(epoch, info, counts, countsCapacity, timeoutNs);
}

} // namespace rcclx::relay
