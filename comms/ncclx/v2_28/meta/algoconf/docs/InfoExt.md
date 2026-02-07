# InfoExt: Per-Collective Algorithm Override

## Overview

`ncclInfoExt` provides a mechanism to override algorithm selection on a per-collective basis. Instead of relying on NCCL's automatic algorithm selection (based on message size, topology, etc.), callers can specify exactly which algorithm, protocol, and parameters to use.

## Data Flow

The override information flows through the NCCL enqueue path as follows:

```
ncclInfo.ext (caller sets)
    │
    ▼
collTaskAppend()  ──►  ncclTaskColl.ext = info->ext  (value copy)
    │
    ▼
ncclPrepareTasks()
    │
    ▼
if (task.ext.hasOverride())
    │
    ├──► infoExtOverride()    // Apply override
    │
    └──► getAlgoInfo()        // Normal auto-selection (fallback)
```

### Key Steps

1. **Caller sets `info.ext`**: When constructing `ncclInfo`, the caller populates the `ext` field with override parameters.

2. **`collTaskAppend()`** (enqueue.cc): Copies the value of `ext` from `ncclInfo` to `ncclTaskColl`:
   ```cpp
   t->ext = info->ext;
   ```

3. **`ncclPrepareTasks()`** (enqueue.cc): Checks if override is requested and applies it:
   ```cpp
   if (agg.ext.hasOverride()) {
     const auto isGrouped = (aggBeg->next != nullptr);
     NCCLCHECK(ncclx::algoconf::infoExtOverride(&agg, isGrouped));
   } else {
     NCCLCHECK(getAlgoInfo(comm, &agg, ...));
   }
   ```

4. **`infoExtOverride()`** (InfoExtOverride.h): Copies override values to task fields:
   ```cpp
   task->algorithm = ext.algorithm;
   task->protocol = ext.protocol;
   task->nMaxChannels = ext.nMaxChannels;
   task->nWarps = ext.nWarps;
   ```

## ncclInfoExt Structure

Defined in `meta/algoconf/InfoExt.h`:

```cpp
struct ncclInfoExt {
  int algorithm = NCCL_ALGO_UNDEF;    // NCCL_ALGO_RING, NCCL_ALGO_TREE, etc.
  int protocol = NCCL_PROTO_UNDEF;    // NCCL_PROTO_LL, NCCL_PROTO_SIMPLE, etc.
  ncclDevRedOpFull opDev = {};        // Optional: device reduction operation
  bool opDevSet = false;              // Whether opDev is set
  int nMaxChannels = 0;               // Number of channels to use
  int nWarps = 0;                     // Number of warps per channel

  bool hasOverride() const;  // True if ANY field is set
  bool isComplete() const;   // True if ALL required fields are set
};
```

### Required Fields for Override

A complete override requires ALL of:
- `algorithm` (e.g., `NCCL_ALGO_RING`, `NCCL_ALGO_TREE`)
- `protocol` (e.g., `NCCL_PROTO_SIMPLE`, `NCCL_PROTO_LL`)
- `nMaxChannels` (> 0)
- `nWarps` (> 0)

**Why all fields are required:** When using `ncclInfoExt`, we bypass NCCL's default algorithm selection logic (`getAlgoInfo()`). That logic normally:
1. Evaluates available algorithms based on topology and message size
2. Selects the optimal algorithm/protocol combination
3. Dynamically computes `nChannels` and `nWarps` based on the selected algorithm

Since we skip this entire computation path, the caller must provide all parameters that would have been computed. Partial override (e.g., specifying only algorithm) is not supported because there is no fallback logic to fill in the missing fields.

The `opDev` field is optional and only applied if `opDevSet = true`.

### Validation

- **Partial override**: Returns `ncclInvalidUsage`. All required fields must be set.
- **Grouped collectives**: Returns `ncclInvalidUsage`. Override is not supported within `ncclGroupStart()`/`ncclGroupEnd()` blocks.

## Usage Example

See `meta/tests/InfoExtAllReduceTest.cc` for a complete integration test.

## Limitations

1. **No partial override**: All required fields must be set. You cannot override just the algorithm while letting NCCL auto-select the protocol.

2. **No grouped collectives**: Override does not work within `ncclGroupStart()`/`ncclGroupEnd()` blocks.

3. **No validation of compatibility**: The caller is responsible for ensuring the algorithm/protocol combination is valid for the given collective and topology.
