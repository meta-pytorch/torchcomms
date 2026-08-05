// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>

#include "comms/ctran/algos/perftrace/Record.h"
#include "comms/ctran/algos/perftrace/Tracer.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::allreduce::ring {

inline bool shouldTraceRank(int rank, std::string_view rankFilter) {
  if (rankFilter.empty()) {
    return true;
  }

  while (!rankFilter.empty()) {
    const auto comma = rankFilter.find(',');
    auto token = rankFilter.substr(0, comma);
    const auto first = token.find_first_not_of(' ');
    const auto last = token.find_last_not_of(' ');
    if (first != std::string_view::npos) {
      token = token.substr(first, last - first + 1);
      int configuredRank = -1;
      const auto result = std::from_chars(
          token.data(), token.data() + token.size(), configuredRank);
      if (result.ec == std::errc{} &&
          result.ptr == token.data() + token.size() && configuredRank == rank) {
        return true;
      }
    }
    if (comma == std::string_view::npos) {
      break;
    }
    rankFilter.remove_prefix(comma + 1);
  }
  return false;
}

struct ChunkTraceMetadata {
  int partition;
  int step;
  int round;
  int chunkId;
  int shardId;
  int shardDataChunkId;
  size_t offsetBytes;
  size_t bytes;
  const char* phase;
};

class RingPerfTrace {
 public:
  RingPerfTrace(
      int rank,
      int nRanks,
      size_t messageBytes,
      size_t chunkBytes,
      size_t numChunks,
      int numBlocks,
      uint64_t opCount,
      std::string_view rankFilter = "") {
    if (!NCCL_CTRAN_ENABLE_PERFTRACE || !shouldTraceRank(rank, rankFilter)) {
      return;
    }
    tracer_ = std::make_unique<perftrace::Tracer>(rank);
    if (!tracer_->isTraceEnabled()) {
      return;
    }

    record_ = std::make_unique<perftrace::Record>("AllReduceRing", rank);
    record_->addMetadata("rank", std::to_string(rank));
    record_->addMetadata("nranks", std::to_string(nRanks));
    record_->addMetadata("message_bytes", std::to_string(messageBytes));
    record_->addMetadata("chunk_bytes", std::to_string(chunkBytes));
    record_->addMetadata("num_chunks", std::to_string(numChunks));
    record_->addMetadata("num_blocks", std::to_string(numBlocks));
    record_->addMetadata("op_count", std::to_string(opCount));
    record_->addMetadata(
        "local_flush_mode", std::to_string(NCCL_CTRAN_NET_LOCAL_FLUSH_MODE));
    record_->addMetadata(
        "force_flush", std::to_string(NCCL_CTRAN_NET_FORCE_FLUSH));
    record_->addMetadata(
        "devices_per_rank", std::to_string(NCCL_CTRAN_IB_DEVICES_PER_RANK));
  }

  ~RingPerfTrace() {
    if (record_) {
      tracer_->addRecord(std::move(record_));
    }
  }

  RingPerfTrace(const RingPerfTrace&) = delete;
  RingPerfTrace& operator=(const RingPerfTrace&) = delete;

  bool enabled() const {
    return record_ != nullptr;
  }

  void startChunkStage(
      const std::string& name,
      int sequence,
      int peer,
      const ChunkTraceMetadata& chunk) {
    if (!record_) {
      return;
    }

    record_->startInterval(name, sequence, peer, chunkMetadata(chunk));
  }

  void addChunkPoint(
      const std::string& name,
      int sequence,
      int peer,
      const ChunkTraceMetadata& chunk,
      const std::map<std::string, std::string>& extraMetadata = {}) {
    if (!record_) {
      return;
    }

    auto metadata = chunkMetadata(chunk);
    metadata.insert(extraMetadata.begin(), extraMetadata.end());
    record_->addPoint(name, sequence, peer, metadata);
  }

  void endChunkStage(const std::string& name, int sequence) {
    // Completion callbacks can be observed after the corresponding stage was
    // skipped or already consumed. Tracing must not abort the collective.
    if (record_ && record_->hasInterval(name, sequence)) {
      record_->endInterval(name, sequence);
    }
  }

 private:
  static std::map<std::string, std::string> chunkMetadata(
      const ChunkTraceMetadata& chunk) {
    return {
        {"partition", std::to_string(chunk.partition)},
        {"step", std::to_string(chunk.step)},
        {"round", std::to_string(chunk.round)},
        {"chunk_id", std::to_string(chunk.chunkId)},
        {"shard_id", std::to_string(chunk.shardId)},
        {"shard_chunk_id", std::to_string(chunk.shardDataChunkId)},
        {"offset_bytes", std::to_string(chunk.offsetBytes)},
        {"bytes", std::to_string(chunk.bytes)},
        {"phase", chunk.phase},
    };
  }

  std::unique_ptr<perftrace::Tracer> tracer_;
  std::unique_ptr<perftrace::Record> record_;
};

} // namespace ctran::allreduce::ring
