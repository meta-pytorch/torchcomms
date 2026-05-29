#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2018, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division

# flake8: noqa

import os
import sys

__author__ = "Jonathan Madsen"
__copyright__ = "Copyright 2020, The Regents of the University of California"
__credits__ = ["Jonathan Madsen"]
__license__ = "MIT"
__version__ = "3.2.0"
__maintainer__ = "Jonathan Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"

from .libpytimemory import (
    report,
    toggle,
    enable,
    disable,
    is_enabled,
    enabled,
    has_mpi_support,
    set_rusage_children,
    set_rusage_self,
    timemory_init,
    timemory_finalize,
    initialize,
    finalize,
    timemory_argparse,
    add_arguments,
    get,
    get_text,
    clear,
    size,
    get_hash,
    get_hash_identifier,
    add_hash_id,
    mpi_init,
    mpi_finalize,
    upcxx_init,
    upcxx_finalize,
    dmp_init,
    dmp_finalize,
    start_mpip,
    stop_mpip,
    start_ompt,
    stop_ompt,
    start_ncclp,
    stop_ncclp,
    start_mallocp,
    stop_mallocp,
    start_function_wrappers,
    stop_function_wrappers,
    enable_signal_detection,
    disable_signal_detection,
    set_exit_action,
    api,
    # papi,
    # cupti,
    # cuda,
    component,
    hardware_counters,
    manager,
    # profiler,
    # trace,
    units,
    scope,
    settings,
    storage,
    signals,
    region,
    options,
    timer_decorator,
    component_decorator,
    component_bundle,
    component_tuple,
    timer,
    rss_usage,
)

"""
from .libpytimemory import profiler as profile

__all__ = [
    "report",
    "toggle",
    "enable",
    "disable",
    "is_enabled",
    "enabled",
    "has_mpi_support",
    "set_rusage_children",
    "set_rusage_self",
    "timemory_init",
    "timemory_finalize",
    "initialize",
    "finalize",
    "timemory_argparse",
    "add_arguments",
    "get",
    "get_text",
    "clear",
    "size",
    "mpi_init",
    "mpi_finalize",
    "upcxx_init",
    "upcxx_finalize",
    "dmp_init",
    "dmp_finalize",
    "start_mpip",
    "stop_mpip",
    "start_ompt",
    "stop_ompt",
    "start_ncclp",
    "stop_ncclp",
    "start_mallocp",
    "stop_mallocp",
    "start_function_wrappers",
    "stop_function_wrappers",
    "enable_signal_detection",
    "disable_signal_detection",
    "set_exit_action",
    "api",
    # papi,
    # cupti,
    # cuda,
    "component",
    "hardware_counters",
    "manager",
    "profiler",
    "trace",
    "units",
    "scope",
    "settings",
    "storage",
    "signals",
    "region",
    "options",
    "timer_decorator",
    "component_decorator",
    "component_bundle",
    "component_tuple",
    "timer",
    "rss_usage",
    "profile",
]
"""
