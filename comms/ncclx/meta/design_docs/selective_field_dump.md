# Selective Field Dump via requestFields Hint

## Context

`ncclCommDumpAll` and `ncclCommDump` dump everything for every communicator —
comm info, colltrace, process global errors, memory,
and global info. This is expensive when callers only need a subset (e.g., just
the aggregated iteration time). We add a `requestFields` hint that specifies
which individual output keys to include. Each dump function checks which of its
keys are requested and only produces those, minimizing overhead.

## API

```cpp
std::unordered_map<std::string, std::unordered_map<std::string, std::string>> map;
ncclCommDumpAll(map, "CT_pastColls;rank;GlobalInfo::totalCommDurPerIterationUs");
```

From Python (via TorchComms):
```python
dump = comm_dump_all(request_fields="GlobalInfo::totalCommDurPerIterationUs")
total_us = int(dump["GlobalInfo"]["totalCommDurPerIterationUs"])
```

The `requestFields` parameter is `std::optional<std::string>` (default:
`std::nullopt`). When `std::nullopt`, all fields are dumped — fully backward
compatible.

Values are semicolon-separated key names.

## Key-Level Filtering

Each dump function takes `const DumpFieldSet& requestFields` and only produces
keys that are in the set (or all keys if the set is empty). A function is
skipped entirely if none of its keys are requested (`anyKeyRequested` check).

### Per-Communicator Keys

| Source function | Output keys |
|----------------|------------|
| `dumpCommInfo()` | commHash, rank, localRank, node, nRanks, localRanks, nNodes, commDesc |
| `dumpNewCollTrace()` → `commDumpToMap()` | CT_pastColls, CT_currentColls, CT_pendingColls, CT_currentIteration, CT_currentIterationCommTimeUs |
| `dumpMemoryTrace()` | memory |

### GlobalInfo Keys (in outer map under "GlobalInfo")

| requestFields value | What it produces |
|--------------------|-----------------|
| `GlobalInfo::NetworkPerfInfo` | NetworkPerfMonitor stats |
| `GlobalInfo::totalCommDurPerIterationUs` | Aggregated iteration comm time across all communicators |

## Architecture

```
ncclCommDumpAll(map, requestFields)
  │
  ├─ parseRequestFields(requestFields)  // Split by ";" → unordered_set<string>
  │                                      // std::nullopt → empty set = dump all
  │
  └─ CommsMonitor::commDumpAll(requestFields)
       │
       └─ commDumpAllImpl(requestFields)
            │
            ├─ per communicator:
            │    commDumpByMonitorInfo(info, requestFields)
            │      ├─ anyKeyRequested({commHash,rank,...})?  → dumpCommInfo(requestFields)
            │      ├─ anyKeyRequested({CT_*})?               → dumpNewCollTrace(requestFields)
            │      │                                           → commDumpToMap(requestFields)
            │      └─ isKeyRequested(memory)?                → dumpMemoryTrace(requestFields)
            │
            ├─ isKeyRequested(GlobalInfo::NetworkPerfInfo)?
            │    → NetworkPerfMonitor::reportPerfStatsAsMap()
            │
            └─ isKeyRequested(GlobalInfo::totalCommDurPerIterationUs)?
                 → aggregate getCurrentIterationCommTime() across all comms
```

The `DumpFieldSet` (alias for `std::unordered_set<std::string>`) is parsed once
at the entry point and threaded through the entire call chain. Helper functions:
- `isKeyRequested(fields, key)` — true if set is empty or contains the key
- `anyKeyRequested(fields, {key1, key2, ...})` — true if set is empty or
  contains any of the keys (used to skip entire dump functions)

## Key Files

| File | Role |
|------|------|
| `v2_27/src/nccl.h.in`, `v2_29/src/nccl.h.in`, `v2_30/src/nccl.h.in` | `ncclCommDumpAll` with `std::optional<std::string> requestFields` parameter |
| `meta/commDump.h` | `DumpFieldSet`, `parseRequestFields()`, `isKeyRequested()`, `anyKeyRequested()` |
| `meta/commDump.cc` | Parsing, per-key gating in all dump functions, `commDumpByMonitorInfo` |
| `meta/comms-monitor/CommsMonitor.cc` | `commDumpAllImpl` with GlobalInfo key gating |
| `meta/tests/CommDumpTest.cc` | Integration tests for selective dump |
| `comms/utils/colltrace/plugins/CommDumpPlugin.cc` | `commDumpToMap(dump, requestFields)` with per-key filtering |
