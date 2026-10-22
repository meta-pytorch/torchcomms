# AMD build and validation

AMD-owned files and targets live here. Shared CUDA/RDMA sources remain in core
and are translated by the existing hipify helpers.

## Internal builds

Consumers link core and `:default_transports`. The bundle supplies the NVML
stub and strong hooks. Supported AMD GPU defaults are P2P/XGMI slot 0, RDMA
and TCP, selected with the existing capability checks.

~~~bash
buck2 build @fbcode//mode/opt-amd-gpu -m rocm70   fbcode//neteng/network_ai/uniflow_extensions/benchmarks:uniflow_bench
buck2 test @fbcode//mode/opt-amd-gpu -m rocm70   fbcode//neteng/network_ai/uniflow_extensions:multi_transport_test_amd   fbcode//neteng/network_ai/uniflow_extensions:rdma_transport_test_amd
~~~

P2P/TCP implementations, tests and benchmark targets are owned by this
package. Narrow core exports supply shared validation sources; local aliases
let the existing GPU macros hipify them.

~~~bash
buck2 test @fbcode//mode/opt-amd-gpu -m rocm70   fbcode//neteng/network_ai/uniflow_extensions/transport/p2p/tests/integration:p2p_single_host_test   fbcode//neteng/network_ai/uniflow_extensions/transport/p2p/tests/integration:peer_to_peer_transfer_test
~~~

Distributed AMD tests use an approved host pair or scheduled qualification
resource. Build `*_binary` targets and run two ranks on matching runtimes.
TCP data uses frontend NICs. MPI metadata can use compatible separate
interfaces. Preserve host networking and record the launch configuration.

## Public in-tree build

This folder exports to `comms/uniflow/extensions`:

~~~bash
cmake -S comms/uniflow -B build-amd   -DUNIFLOW_GPU_PLATFORM=HIP   -DUNIFLOW_BUILD_EXTENSIONS=ON   -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-amd -j
~~~

Inside fbsource, set `UNIFLOW_EXTENSIONS_DIR` to this physical directory.
Python uses the existing distribution; see `comms/uniflow/python/README.md`.
Keep hipify-perl, shared HIP guards and core dependency helpers. The NVML stub
reports unsupported queries; shared discovery still provides GPU/NIC topology.

## Benchmarks and CI

~~~bash
python3 fbcode/neteng/network_ai/uniflow/benchmarks/scripts/compare_rdma_bandwidth.py   --gpu mi350 --hosts <host0>,<host1>   --bench-target fbcode//neteng/network_ai/uniflow_extensions/benchmarks:uniflow_bench
~~~

The MI350 serving package is
`fbcode//neteng/network_ai/uniflow_extensions/benchmarks:uniflow_disagg_bench_mast_mi350`.
NVIDIA reference packages remain in core.

This package owns AMD CI. Shared sources receive AMD coverage through these
targets. Core `amd/` retains the neutral-zone dependency checks.
Only fresh complete hardware runs qualify a release; list resource and
configured CI rollout gaps.
