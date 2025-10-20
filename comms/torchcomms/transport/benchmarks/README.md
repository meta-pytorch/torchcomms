# RDMA Transport Benchmark

## Running the Benchmark

```bash
# Run all benchmarks
buck run @//mode/opt //comms/torchcomms/transport/benchmarks:rdma_transport_bench

# Run specific benchmarks
buck run @//mode/opt //comms/torchcomms/transport/benchmarks:rdma_transport_bench -- --benchmark_filter="BM_RdmaTransport_Write"

# Generate JSON output
buck run @//mode/opt //comms/torchcomms/transport/benchmarks:rdma_transport_bench -- --benchmark_format=json --benchmark_out=rdma_bench_results.json
```
