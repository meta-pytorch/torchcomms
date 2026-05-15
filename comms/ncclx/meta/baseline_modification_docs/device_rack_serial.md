# Device Rack Serial: Baseline Modifications

## Background

`NCCL_MNNVL_TRUNK_DISABLE` is a Meta-custom feature that disables multi-NVLink trunk connections and instead uses `DEVICE_RACK_SERIAL` (read from the topology file at `NCCL_TOPO_FILE_PATH`) to decide P2P connectivity — ranks on the same rack get P2P enabled, ranks on different racks do not. This feature was originally added to all three NCCLX versions (v2_27, v2_29, v2_30) with the following baseline modifications:

- `transport.h`: Added `int rackSerial` field to `ncclPeerInfo`
- `init.cc`: Added `ncclxGetDeviceRackSerial()` to parse `DEVICE_RACK_SERIAL` from the topology file as an integer via `folly::tryTo<int>()`
- `paths.cc`: Added a block in `ncclTopoCheckP2p()` to compare `rackSerial` values and set `*p2p` accordingly

Each version had its own copy of `ncclxGetDeviceRackSerial()` with identical logic.

## Bug Fixes

GCP GB300 hosts have alphanumeric `DEVICE_RACK_SERIAL` values (e.g. `C1507842765072`). The original `folly::tryTo<int>()` conversion crashed with a `CHECK(maybeRackSerial.hasValue())` failure on these hosts. This diff fixes the crash by switching `rackSerial` from `int` to `char[]` (string-based), and deduplicates the three copies of parsing logic into a shared `meta/DeviceRackSerial.h/cc`.

## Versions Affected

v2_27, v2_29, v2_30

## Baseline Files Modified

### 1. `src/include/transport.h` — rackSerial type change

**Change**: Added `constexpr auto kMaxRackSerialLen = 63` and changed `ncclPeerInfo.rackSerial` from `int` to `char[kMaxRackSerialLen+1]`.

```cpp
constexpr auto kMaxRackSerialLen = 63; // [META] for DeviceRackSerial string support
// ...
struct ncclPeerInfo {
  // ...
  char rackSerial[kMaxRackSerialLen+1]; // [META] string-based rack serial (was int)
};
```

**Why in baseline**: `ncclPeerInfo` is a core NCCL struct exchanged between ranks during init. The field type must match across all ranks for correct bootstrapping.

### 2. `src/init.cc` — Rack serial loading

**Change**: Removed the per-version `ncclxGetDeviceRackSerial()` function (which used `folly::tryTo<int>()` and crashed on alphanumeric serials). Replaced with a call to shared `ncclx::loadRackSerial()` that stores the value as a string.

```cpp
// [META] Load rack serial for MNNVL trunk disable (string-based, supports alphanumeric serials)
if(NCCL_MNNVL_TRUNK_DISABLE) {
  if (ncclx::loadRackSerial(NCCL_TOPO_FILE_PATH, info->rackSerial, sizeof(info->rackSerial))) {
    INFO(NCCL_INIT, "Loaded rack serial: %s", info->rackSerial);
  } else {
    WARN("No rack serial information available, skipping rack serial check");
  }
}
```

**Why in baseline**: `fillInfo()` populates `ncclPeerInfo` during communicator init. The rack serial must be loaded here so it is available for the P2P path decision in `paths.cc`.

### 3. `src/graph/paths.cc` — Rack serial comparison

**Change**: Updated the `NCCL_MNNVL_TRUNK_DISABLE` block to use string-based comparison via `ncclx::isSameRackSerial()` instead of integer equality. Changed format specifiers from `%d` to `%s`. Block is tagged with `[META]` comment.

```cpp
// [META] Check if multi-NVLink P2P is disabled and handle rack serial matching
if (NCCL_MNNVL_TRUNK_DISABLE && mnnvl) {
  if (comm->peerInfo[rank1].rackSerial[0] != '\0' && comm->peerInfo[rank2].rackSerial[0] != '\0') {
    *p2p = ncclx::isSameRackSerial(comm->peerInfo[rank1].rackSerial, comm->peerInfo[rank2].rackSerial);
  } else {
    WARN("No rack serial information available, skipping rack serial check");
  }
}
```

**Why in baseline**: `ncclTopoCheckP2p()` is the core P2P path decision function. The rack serial comparison must happen here alongside other P2P topology checks.

## NCCLX meta/ Files (not baseline)

| File | Purpose |
|------|---------|
| `meta/DeviceRackSerial.h` | `loadRackSerial()` and `isSameRackSerial()` declarations |
| `meta/DeviceRackSerial.cc` | `loadRackSerial()` implementation — parses `DEVICE_RACK_SERIAL` from topology file as string |
| `meta/tests/DeviceRackSerialTest.cc` | Unit tests for load/compare functions |

## Build Integration

Each version's `def_build.bzl` and `src/Makefile` include `meta/DeviceRackSerial.cc` in the source list for both buck and conda builds.

## Revert Checklist

To remove rack serial support from the baseline:

1. `src/include/transport.h`: Revert `rackSerial` from `char[]` to `int`; remove `kMaxRackSerialLen`
2. `src/init.cc`: Remove `meta/DeviceRackSerial.h` include; remove `loadRackSerial()` call block; restore original `ncclxGetDeviceRackSerial()` if integer-only serials are acceptable
3. `src/graph/paths.cc`: Remove `meta/DeviceRackSerial.h` include; revert to integer comparison (`==`); revert format specifiers from `%s` to `%d`; remove `[META]` tag
4. `def_build.bzl` / `src/Makefile`: Remove `meta/DeviceRackSerial.cc` from source lists
