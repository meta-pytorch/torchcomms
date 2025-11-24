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

## Reference Run
```
[...]port/benchmarks/RdmaTransportBench.cc     relative  time/iter   iters/s  bytes_per_second  message_size
============================================================================================================
RdmaMemory_Register_Deregister(8192)                       54.54us    18.34K               NaN           NaN
RdmaTransport_Write(8192)                                 160.05us     6.25K           130.03M         8.19K
RdmaTransport_Write(16384)                                175.75us     5.69K           268.59M        16.38K
RdmaTransport_Write(32768)                                177.49us     5.63K           504.12M        32.77K
RdmaTransport_Write(65536)                                179.14us     5.58K             1.07G        65.54K
RdmaTransport_Write(131072)                               191.55us     5.22K             2.08G       131.07K
RdmaTransport_Write(262144)                               181.81us     5.50K             4.30G       262.14K
RdmaTransport_Write(524288)                               163.74us     6.11K             8.46G       524.29K
RdmaTransport_Write(1048576)                              171.10us     5.84K            15.42G         1.05M
RdmaTransport_Write(2097152)                              186.61us     5.36K            23.83G         2.10M
RdmaTransport_Write(4194304)                              230.06us     4.35K            32.51G         4.19M
RdmaTransport_Write(8388608)                              314.55us     3.18K            40.33G         8.39M
RdmaTransport_Write(16777216)                             483.04us     2.07K            43.24G        16.78M
RdmaTransport_Write(33554432)                             823.78us     1.21K            45.90G        33.55M
RdmaTransport_Write(67108864)                               1.54ms    650.69            47.19G        67.11M
RdmaTransport_Write(134217728)                              2.90ms    345.26            47.88G       134.22M
RdmaTransport_Write(268435456)                              5.68ms    176.18            48.19G       268.44M
RdmaTransport_Read(8192)                                  146.80us     6.81K           146.29M         8.19K
RdmaTransport_Read(16384)                                 155.91us     6.41K           309.13M        16.38K
RdmaTransport_Read(32768)                                 145.62us     6.87K           668.73M        32.77K
RdmaTransport_Read(65536)                                 156.39us     6.39K             1.21G        65.54K
RdmaTransport_Read(131072)                                182.26us     5.49K             1.96G       131.07K
RdmaTransport_Read(262144)                                178.20us     5.61K             3.97G       262.14K
RdmaTransport_Read(524288)                                160.20us     6.24K             8.32G       524.29K
RdmaTransport_Read(1048576)                               149.55us     6.69K            16.13G         1.05M
RdmaTransport_Read(2097152)                               172.71us     5.79K            25.89G         2.10M
RdmaTransport_Read(4194304)                               209.88us     4.76K            34.38G         4.19M
RdmaTransport_Read(8388608)                               318.93us     3.14K            39.02G         8.39M
RdmaTransport_Read(16777216)                              468.73us     2.13K            43.58G        16.78M
RdmaTransport_Read(33554432)                              818.68us     1.22K            45.96G        33.55M
RdmaTransport_Read(67108864)                                1.51ms    660.78            47.16G        67.11M
RdmaTransport_Read(134217728)                               2.89ms    345.50            47.88G       134.22M
RdmaTransport_Read(268435456)                               5.66ms    176.59            48.23G       268.44M
============================================================================================================
```
