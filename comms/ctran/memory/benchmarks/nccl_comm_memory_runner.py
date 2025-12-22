from comms.testinfra.multiproc_benchmark_runner import (  # pyre-ignore
    run_nccl_distributed_test,
)
from windtunnel.benchmarks.python_benchmark_runner.benchmark import (  # pyre-ignore
    main as servicelab_main,
    register_benchmark,
    UserCounters,
    UserMetric,
)


TARGET_PREFIX = "fbcode//comms/ctran/memory/benchmarks:nccl_comm_memory_bench_ppn"
SEARCH_PATTERN = r"NCCL Comm Memory:\s*([\d.]+)\s*MB"


@register_benchmark(use_counters=True)
def ncclCommMemoryMB(counters: UserCounters) -> None:  # pyre-ignore
    for ppn in [1, 4]:
        target = f"{TARGET_PREFIX}{ppn}"
        memory_mb = run_nccl_distributed_test(target, SEARCH_PATTERN)
        counters[f"ncclCommMemoryMB_ppn{ppn}"] = UserMetric(value=int(memory_mb))


def run_local() -> None:
    results = {}
    ncclCommMemoryMB(results)
    for key, metric in results.items():
        print(f"  {key}: {metric.value:.2f} MB")


def main() -> None:
    servicelab_main()


if __name__ == "__main__":
    main()
