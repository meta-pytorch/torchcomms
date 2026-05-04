#!@PYTHON_EXECUTABLE@

"""Example
@PYTHON_EXECUTABLE@ -m timemory.profiler -b -m 10 -- ./@FILENAME@
@PYTHON_EXECUTABLE@ -m timemory.trace -b -- ./@FILENAME@
"""

import sys
import numpy as np


@profile  # noqa: F821
def fib(n):
    return n if n < 2 else (fib(n - 1) + fib(n - 2))


@profile  # noqa: F821
def inefficient(n):
    a = 0
    for i in range(n):
        a += i
        for j in range(n):
            a += j
    arr = np.arange(a * n * n * n, dtype=np.double)
    return arr.sum()


def run(nfib):
    ret = 0
    ret += fib(nfib)
    ret += fib(nfib % 5 + 1)
    ret += inefficient(nfib)
    return ret


if __name__ == "__main__":
    nfib = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ans = run(nfib)
    print("Answer = {}".format(ans))
