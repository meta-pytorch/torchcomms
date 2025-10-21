# IBVerbX Library

IBVerbX is a C++ wrapper library for InfiniBand Verbs that provides both dynamic loading and direct linking options.

## Build Options

### Default Build (Dynamic Loading)
```bash
buck build //comms/ctran/ibverbx:ibverbx
```

This builds the library with dynamic loading of InfiniBand verbs functions using `dlopen`/`dlsym`. This is the default behavior and allows the library to work without requiring InfiniBand libraries to be linked at compile time.

### Direct Linking Build
```bash
buck build //comms/ctran/ibverbx:ibverbx-rdma-core
```

This builds the library with the `-DIBVERBX_BUILD_RDMA_CORE` compiler flag, which:
- Includes `<infiniband/verbs.h>` directly
- Bypasses the `dlopen` path in `buildIbvSymbols()`
- Links directly against the `rdma-core` library
- Provides better performance by avoiding function pointer indirection

## Compiler Flag: `-DIBVERBX_BUILD_RDMA_CORE`

When the `-DIBVERBX_BUILD_RDMA_CORE` flag is set:

1. **Header files** (`Ibverbx.h` and `Ibvcore.h`):
   - Include `<infiniband/verbs.h>` directly

2. **Implementation** (`Ibverbx.cc`):
   - The `buildIbvSymbols()` function assigns function pointers directly to the real InfiniBand functions
   - Skips the dynamic loading code path entirely

3. **Build configuration** (`BUCK`):
   - Links against the `rdma-core` external dependency
   - Adds the compiler flag to enable conditional compilation

## Usage

Both build variants provide the same API and can be used interchangeably. Choose the appropriate variant based on your deployment requirements:

- Use the default build when you need runtime flexibility and don't want to require InfiniBand libraries at link time
- Use the linked build when you want maximum performance and know InfiniBand libraries will be available

## Implementation Details

The library always uses `ibverbx::` struct definitions that are identical to the real InfiniBand types, regardless of build configuration. All types are defined in the `ibverbx` namespace (e.g., `ibverbx::ibv_device`, `ibverbx::ibv_context`, `ibverbx::ibv_qp`, etc.) to avoid namespace conflicts with system InfiniBand headers.

The conditional compilation (`#ifdef IBVERBX_BUILD_RDMA_CORE`) only affects:
- Whether to include `<infiniband/verbs.h>` directly or use dynamic loading
- Function pointer assignment in `buildIbvSymbols()` (direct assignment vs dlsym)
- Build dependencies and linking against rdma-core libraries

This design ensures type compatibility across all build variants while maintaining clear namespace separation and avoiding conflicts with system InfiniBand installations.
