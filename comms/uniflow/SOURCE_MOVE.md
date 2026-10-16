# UniFlow source ownership and release

## Ownership and hooks

Core contains NVIDIA support, RDMA and TCP control messaging. AMD-owned files,
P2P/XGMI, TCP data and internal MTIA implementations live in
`fbcode/neteng/network_ai/uniflow_extensions`. Public export nests that package
at `comms/uniflow/extensions`. Namespace `uniflow` and Python imports are
unchanged. Core has no extension implementation dependency.

NVIDIA NVLink transport, tests, benchmarks and script are deleted. Keep NVML
topology discovery, shared CUDA operations needed by AMD, and enum values
`NVLink=0`, `RDMA=1`, `TCP=2`. Slot 0 serves AMD P2P.

`addExtensionTransportFactories` and `extensionTransportSupported` are the
two transport hooks. Core provides weak no-op/NotImplemented defaults;
extensions provide existing P2P/TCP setup and checks. A retained static
initializer sets two private function pointers for shared linkers that bind
the weak core functions first. Factories sort by transport type for the
topology merge.

MTIA keeps its existing link-time adapter/discovery functions. Shared CUDA
sources still hipify, with the small HIP guards and dependency helpers.
Selection and failure behavior are preserved. NVIDIA same-host transfers use
RDMA, including when the caller prefers NVLink.

## Consumers and packages

NVIDIA consumers keep their core dependency. Applicable AMD/MTIA consumers
also link `fbcode//neteng/network_ai/uniflow_extensions:default_transports`.
Both NIXL versions, vLLM, supported oil/TorchStore and benchmark variants use
that bundle.

AMD/MTIA validation compiles exported core source bodies. P2P/TCP tests and
AMD benchmarks are extension-owned. Core keeps NVIDIA reference benchmarks.

The existing `torchcomms-uniflow` package produces NVIDIA core, NVIDIA with
TCP and AMD extension-enabled wheels. See [build instructions](python/README.md).
Extensions build in tree through `UNIFLOW_BUILD_EXTENSIONS`.

Versions are preserved: native `VERSION` is `0.1.0.dev0` and Python project
metadata is `0.0.1`. Release owners must reconcile this existing mismatch
before assigning a public release.

## Release gates

The ShipIt mapping in
`source/opensource/shipit_config/meta-pytorch/torchcomms.cconf` must land before
public export of extension code. It maps the public extension folder and
excludes MTIA/internal files. Local exported-layout builds publish no code.

Core CI covers NVIDIA; extensions own AMD/MTIA coverage. Shared-source changes
must select extension AMD targets. Actual NVIDIA-only selection requires the
project rollout of `ci_efficiency/citadel:enable_configured_target_filter`
for `fbcode`.

Retain source revisions, commands, test IDs and artifacts for:

- Static/shared hooks and exact registered transport sets.
- Single/two-host RDMA/TCP/P2P and old/new NVIDIA peer compatibility.
- Topology/NIC parity and complete bandwidth/latency size/chunk matrices.
- All three installed wheels and native installed consumers.
- Both NIXL versions and applicable downstream consumers.
- Actual extension-only, NVIDIA-only and shared-source CI selection.

Record the intentional NVIDIA NVLink-to-RDMA performance change separately.
A build, mock, skipped test, historical run or partial matrix is not a
hardware pass. List unavailable resources and external approvals explicitly.
