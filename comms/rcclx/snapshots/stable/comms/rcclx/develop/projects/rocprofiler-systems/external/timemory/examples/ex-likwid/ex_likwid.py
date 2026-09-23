#!@PYTHON_EXECUTABLE@

import timemory
from timemory.util import auto_tuple
import sys


def fib(n):
    return n if n < 2 else (fib(n - 1) + fib(n - 2))


def get_tools(extra=[]):
    """
    Return the LIKWID components
    """
    components = ["likwid_marker", "likwid_nvmarker"] + extra
    return [getattr(timemory.component, x) for x in components]


@auto_tuple(get_tools(), add_args=True)
def run_fib(n):
    return fib(n)


if __name__ == "__main__":
    nfib = int(sys.argv[1]) if len(sys.argv) > 1 else 34

    with auto_tuple(get_tools(), "%s(%i)" % (sys.argv[0], nfib)):
        ans = fib(nfib)
        ans += run_fib(nfib + 1)

    print("fibonacci(34) = {}".format(ans))
