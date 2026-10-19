# UniFlow transport benchmarks

## Targets

| Benchmark | NVIDIA core | NVIDIA with TCP | AMD extensions |
|---|---|---|---|
| `rdma_bandwidth` | yes | yes | yes |
| `sendrecv_bandwidth` | yes | yes | yes |
| `nccl_sendrecv` | reference | reference | no |
| `tcp_bandwidth` | no | yes | yes |
| `xgmi_bandwidth` | no | no | yes |

Core owns the harness and shared RDMA/sendrecv sources. Extensions compile
exported `main.cpp` and own TCP/XGMI sources. NVIDIA NVLink benchmarks and
launcher are deleted.

~~~bash
buck2 build @fbcode//mode/opt -c hpc_comms.use_ncclx=stable   fbcode//comms/uniflow/benchmarks:uniflow_bench
buck2 build @fbcode//mode/opt-amd-gpu -m rocm70   fbcode//neteng/network_ai/uniflow_extensions/benchmarks:uniflow_bench
~~~

The Python KV source stays in core. Its AMD target is
`fbcode//neteng/network_ai/uniflow_extensions/benchmarks:kv_transfer_bench`.
The MI350 serving package moves to the same extensions package. NVIDIA
serving/reference packages remain in core.

## Execution

The harness reads `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE` and
`LOCAL_RANK`. Rendezvous establishes TCP control channels. Each benchmark
creates its own transport, registers buffers, exchanges handles and
synchronizes timed loops over the control channel.

BenchmarkRunner registers implementations and dispatches them. Each source
owns its sweep and warmup. Stats computes latency statistics; Reporter writes
rank-0 tables/CSV. SegmentHelper keeps the friend name SegmentTest required
by core.

RDMA tests put/get, NIC/chunk count, queue depth and GPU selection.
Sendrecv tests slab staging/pipeline depth. TCP tests frontend lane striping
and GPU host staging; XGMI tests AMD P2P. C++ microbenchmarks drive transports
directly; Python KV and serving benchmarks exercise the public API.

## Launchers

Core scripts accept `--bench-target`:

~~~bash
python3 fbcode/comms/uniflow/benchmarks/scripts/compare_rdma_bandwidth.py   --hosts <host0>,<host1> --gpu mi350   --bench-target fbcode//neteng/network_ai/uniflow_extensions/benchmarks:uniflow_bench
~~~

Local caches and remote target stamps keep core/extension selections distinct.
The existing linkage choice remains independent.
Native builds enable `UNIFLOW_BUILD_BENCHMARKS`; extension sources are added
in tree when `UNIFLOW_BUILD_EXTENSIONS=ON`.

## Evidence

Record revisions, toolchain, NIC/GPU mapping, sizes/chunks, warmup, iterations,
queue depth and correctness checks. Compare complete curves at matching
settings and repeat changes outside noise.

Record NVIDIA same-host NVLink-to-RDMA separately. Topology discovery and NIC
selection remain. Builds, skipped tests, historical numbers and incomplete
curves are not hardware performance passes.
