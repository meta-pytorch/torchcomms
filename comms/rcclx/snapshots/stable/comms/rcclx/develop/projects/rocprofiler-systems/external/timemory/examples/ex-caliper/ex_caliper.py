#!@PYTHON_EXECUTABLE@

import os
import sys
import argparse
import timemory
from timemory.bundle import auto_tuple
from timemory.component import CaliperConfig


def fib(n):
    return n if n < 2 else (fib(n - 1) + fib(n - 2))


def get_tools(extra=[]):
    """
    Return the caliper component
    """
    components = ["caliper_marker"] + extra
    return [getattr(timemory.component, x) for x in components]


@auto_tuple(get_tools(), add_args=True)
def run_fib(n):
    return fib(n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--nfib",
        help="Fibonacci value",
        type=int,
        default=34,
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Caliper configuration",
        type=str,
        nargs="*",
        default=["runtime-report"],
    )
    parser.add_argument(
        "-C",
        "--components",
        help="timemory component types",
        default=[],
        choices=timemory.component.get_available_types(),
        nargs="*",
        type=str,
    )

    args = parser.parse_args()
    nfib = args.nfib

    cfg = CaliperConfig()
    cfg.configure(",".join(args.config))
    cfg.start()

    with auto_tuple(
        get_tools(args.components),
        "%s(%i)" % (os.path.basename(sys.argv[0]), nfib),
    ):
        ans = fib(nfib)
        ans += run_fib(nfib + 1)

    cfg.stop()
    print("fibonacci({}) = {}".format(nfib, ans))
    timemory.finalize()
