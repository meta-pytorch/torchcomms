#!@PYTHON_EXECUTABLE@

import sys
import timemory
from timemory.trace import trace


def fib(n):
    return n if n < 2 else (fib(n - 1) + fib(n - 2))


@trace(["wall_clock", "peak_rss"])
def run(nfib):
    return fib(nfib)


if __name__ == "__main__":
    nfib = int(sys.argv[1]) if len(sys.argv) > 1 else 23
    ans = run(nfib)
    print("Answer = {}".format(ans))
    timemory.finalize()
