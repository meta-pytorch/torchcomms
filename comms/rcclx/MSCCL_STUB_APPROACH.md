# MSCCL/MSCCL++ Pure Stub Approach

## Strategy

**Pure Stub Approach**: Minimize changes to existing RCCL code by providing stub
implementations for all MSCCL functions. This approach:

- Keeps all original RCCL source code unchanged
- Removes MSCCL source files from build configuration
- Provides stub file with safe no-op implementations
- Easiest to maintain during upstream RCCL syncs

## Implementation

### What Was Changed

**Build Configuration Only:**

1. `/data/users/dmwu/fbsource/fbcode/comms/rcclx/develop/def_build.bzl`
   - Commented out MSCCL source files
   - Added stub file to build: `"misc/msccl_stub.cc"`

**New Files Created:**

1. `/data/users/dmwu/fbsource/fbcode/comms/rcclx/develop/src/misc/msccl_stub.cc`
   - Provides stub implementations for all MSCCL functions
   - All functions return `false` or `ncclSuccess` (no-ops)
   - Uses forward declarations (no MSCCL header dependencies)

**Documentation:**

1. `/data/users/dmwu/fbsource/fbcode/comms/rcclx/patches/msccl_stub.patch`
   - Documents the stub implementation
2. `/data/users/dmwu/fbsource/fbcode/comms/rcclx/MSCCL_REMOVAL_SUMMARY.md`
   - Original summary from initial implementation

### What Was NOT Changed

**All RCCL Source Files Remain Unchanged:**

- `init.cc` - Keeps all MSCCL headers and initialization code
- `group.cc` - Keeps all MSCCL headers and conditional blocks
- `collectives.cc` - Keeps all MSCCL headers and conditional checks
- `transport/net.cc` - Unchanged
- `graph/connect.cc` - Unchanged
- All other source files - Unchanged

**MSCCL Implementation Files (Excluded from Build):**

- All files in
  `/data/users/dmwu/fbsource/fbcode/comms/rcclx/develop/src/misc/msccl/`
- All files in
  `/data/users/dmwu/fbsource/fbcode/comms/rcclx/develop/src/include/msccl/`
- All files in
  `/data/users/dmwu/fbsource/fbcode/comms/rcclx/develop/src/misc/mscclpp/`

## How It Works

1. **Compilation**: RCCL source files include MSCCL headers normally
2. **Linking**: Stub file provides all MSCCL symbols
3. **Runtime**: MSCCL checks always return `false`, so MSCCL code paths never
   execute
4. **Result**: MSCCL functionality completely disabled with minimal code changes

## Stub Functions Provided

All stub functions return safe defaults:

```cpp
bool mscclEnabled() { return false; }
bool mscclForceEnabled() { return false; }
bool mscclIsCaller() { return false; }
bool mscclAvailable(const ncclComm_t comm) { return false; }
ncclResult_t mscclGroupStart() { return ncclSuccess; }
ncclResult_t mscclGroupEnd() { return ncclSuccess; }
ncclResult_t mscclInit(ncclComm_t comm) { return ncclSuccess; }
ncclResult_t mscclTeardown(ncclComm_t comm) { return ncclSuccess; }
ncclResult_t mscclEnqueueCheck(...) { return ncclInternalError; }
// ... and others
```

## Benefits

1. **Minimal Changes**: Only 2 files changed (def_build.bzl + stub file)
2. **Safe Runtime**: All MSCCL checks return false immediately
3. **Easy Maintenance**: Future RCCL syncs won't conflict
4. **No Risk**: Original RCCL code logic preserved
5. **Clean Build**: Linker satisfied with all required symbols

## Validation

- ✅ No changes to RCCL source files
- ✅ Stub file compiles without MSCCL headers
- ✅ All MSCCL symbols provided by stub
- ✅ Build configuration updated correctly
- ✅ No new compilation errors
- ✅ All MSCCL link errors resolved
- ✅ Function signatures match exactly (including mscclFunc_t enum)

## Files Modified Summary

### Changed:

- `develop/def_build.bzl` - Build configuration
- `develop/src/misc/msccl_stub.cc` - Stub implementation (NEW)

### Created:

- `patches/msccl_stub.patch` - Patch documentation
- `MSCCL_REMOVAL_SUMMARY.md` - Original documentation
- `MSCCL_STUB_APPROACH.md` - This file

### Unchanged:

- All RCCL source files (_.cc, _.h)
- All MSCCL implementation files (ignored by build)

## Result

**Pure stub approach successfully implemented!**

The RCCL codebase remains unchanged, MSCCL functionality is completely disabled,
and the build is stable. This is the cleanest approach with minimal maintenance
overhead.
