#!@PYTHON_EXECUTABLE@

import sys
import timemory
from timemory.profiler import profile


def fib(n):
    return n if n < 2 else (fib(n - 1) + fib(n - 2))


@profile(["wall_clock", "peak_rss"])
def run_profile(nfib):
    return fib(nfib)


if __name__ == "__main__":
    nfib = int(sys.argv[1]) if len(sys.argv) > 1 else 23
    ans = run_profile(nfib)
    print("Answer = {}".format(ans))
    timemory.finalize()
