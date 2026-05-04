#!@PYTHON_EXECUTABLE@

import os
import json
import time

nfib = 33

#
#   CONFIG ENV BEFORE LOADING TIMEMORY
#
if int(os.environ.get("EX_SAMPLE_NO_ENV", "0")) < 1:
    clinit = os.environ.get("TIMEMORY_COMPONENT_LIST_INIT", "").split()
    events = os.environ.get("TIMEMORY_CUPTI_EVENTS", "").split()
    metrics = os.environ.get("TIMEMORY_CUPTI_METRICS", "").split()

    clinit += ["cupti_counters"]
    events += ["active_warps", "global_load"]
    metrics += ["achieved_occupancy", "global_store"]

    os.environ["TIMEMORY_COMPONENT_LIST_INIT"] = "{}".format(",".join(clinit))
    os.environ["TIMEMORY_CUPTI_EVENTS"] = "{}".format(",".join(events))
    os.environ["TIMEMORY_CUPTI_METRICS"] = "{}".format(",".join(metrics))

import timemory  # noqa: E402


#
#    RUN FIBONACCI CALCULATION
#
def fibonacci(n):
    """
    Use this function to get CPU usage
    """
    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)


#
#   CONFIG
#
def get_timemory_config(config=""):
    """
    Provides auto-tuple configurations
    """
    configs = {}
    configs[None] = ["wall_clock", "cpu_clock"]
    items = configs[None] if config not in configs else configs[config]
    #
    # generate the list of enums
    #
    return [getattr(timemory.component, x) for x in items]


#
#   AUTO-TIMER
#
@timemory.bundle.auto_timer()
def foo():
    """
    Demonstrate decorator and context-manager with auto_timer
    Sleep for 2 seconds then run fibonacci calculation within context-manager
    """
    time.sleep(2)
    with timemory.bundle.auto_timer(key="[fibonacci]"):
        print("fibonacci({}) = {}".format(nfib, fibonacci(nfib)))


#
#   AUTO-TUPLE
#
@timemory.bundle.auto_tuple(get_timemory_config(), key="")
def bar():
    """
    Demonstrate decorator and context-manager with auto_tuple
    Run fibonacci calculation and then sleep for 2 seconds within context-manager
    """
    print("fibonacci({}) = {}".format(nfib, fibonacci(nfib)))
    with timemory.bundle.auto_tuple(get_timemory_config(), key="[sleep]"):
        time.sleep(2)


#
#   EXECUTE
#
if __name__ == "__main__":
    # do some work
    foo()
    bar()

    # get the dictionary
    data = timemory.get()
    # print the dictionary
    print(
        "TIMEMORY_DATA:\n{}".format(json.dumps(data, indent=4, sort_keys=True))
    )
    timemory.finalize()
