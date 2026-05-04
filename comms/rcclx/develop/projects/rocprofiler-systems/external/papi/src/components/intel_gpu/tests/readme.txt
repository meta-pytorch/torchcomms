Test examples:
To collect metrics, set ZET_ENABLE_METRICS=1
To collect metrics in query mode, set ZET_ENABLE_API_TRACING_EXP=1

---------------------------
gpu_metric_list:
	list all available metrics with descriptions
	usage:  gpu_metric_list [-a] [-g <metricGroupName>] [-m <metricName>]
	option:
		-a		          list all metrics groups
		-g <group_name>:  list all metrics in the given group name
		-m <metric_name>: check if the given metric is supported.

---------------------------
gpu_metric_read:
	collect metrics in a certain time interval (second).
	usage: usage: gpu_metric_read -d <duration> [ -l <loops>][-s][-m metric[:device=0][:tile=0]]
	option: 
		-d <duration>   collect {duration} second, independent on workload execution.
		-l <loops>      collect {loops} time, each with {duration} second.
		-s              reset after each collection, and get delta between collections.
		-m <metric1,metric2,...,>   metrics list. 
                        Using modify ":device=<n>" for selecting a device in a multi-devices system
                        Using modify ":tile=<m>" for selecting a tile(subdevice) in a multi-tiles device
	example:
		gpu_metric_read -d 1 -l 2 ComputeBasic.GpuTime,ComputeBasic.AvgGpuCoreFrequencyMHz
		gpu_metric_read -d 1 ComputeBasic.GpuTime:device=0,ComputeBasic.AvgGpuCoreFrequencyMHz:device=0
		gpu_metric_read -d 1 ComputeBasic.GpuTime:device=0:tile=0,ComputeBasic.GpuTime:device=0:tile=1

---------------------------
gpu_query_gemm
	collect metrics for kernel gemm execution
	usage: gpu_query_gemm <size> <repeats> [-m metric[:device=0][:tile=0]]
	option:
		size:    matrix size for calculation.
		repeats: execution repeat tiems
		-m <metric1,metric2,...,>   metrics list.
	example:
		export ZET_ENABLE_API_TRACING_EXP=1
		gpu_query_gemm 4096 1 -m ComputeBasic.GpuTime,ComputeBasic.ComputeBasic.GpuCoreClocks,ComputeBasic.AvgGpuCoreFrequencyMHz

