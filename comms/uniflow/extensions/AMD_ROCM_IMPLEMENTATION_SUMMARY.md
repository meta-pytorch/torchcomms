# AMD ROCm implementation

AMD builds shared CUDA sources through existing hipify rules and small HIP
guards. The source move adds the two transport hooks and their shared-link
initializer fallback. Namespace and public APIs stay unchanged.

| Component | Owner and behavior |
|---|---|
| CUDA operations and device adapter | Shared core sources, hipified |
| RDMA/copy engine/NIC selection | Shared core sources and existing dependency selectors |
| NVML stub | Extensions; NVML queries return NotImplemented |
| P2P/XGMI | Extensions; existing capability gate/HIP IPC, slot 0 |
| TCP data/host staging | Extensions; existing registration and failure behavior |
| TCP control | Core |
| MTIA adapter/discovery | Internal extensions `fb/`, existing link-time functions |

Factories sort by transport type. Supported AMD GPU defaults are P2P slot 0,
RDMA and TCP. Existing global preferences, per-case overrides and capability/
failure checks remain. NVIDIA NVLink transport is deleted.

Shared validation source bodies stay in core. Extension variants compile
them through narrow exports and standard source aliases. P2P/TCP tests and
AMD benchmarks live here. Core retains NVIDIA reference benchmarks and
GPU-free dependency guards.

See [build and validation](AMD_BUILD.md) and `comms/uniflow/SOURCE_MOVE.md`
for package variants, public layout and release gates. Only fresh complete
hardware runs count toward hardware qualification.
