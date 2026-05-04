#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2020, The Regents of the University of California,
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
#

from __future__ import absolute_import
import os
import re
import sys
import math
from math import log, pi

import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import numpy


GIGABYTE = 1.0e9
FONT_SIZE = 16
FONT_COLOR = "black"
FONT_WEIGHT = "bold"
WARP_SIZE = 32
DEBUG = (
    True
    if re.search(
        "^(t|y|on|[0-9])", os.environ.get("TIMEMORY_DEBUG", "off").lower()
    )
    is not None
    else False
)
VERBOSE = int(os.environ.get("TIMEMORY_VERBOSE", "0")) if not DEBUG else 255
RANK_INDEX = 0

__all__ = [
    "echo_dart_tag",
    "get_json_entry",
    "ert_params",
    "ert_counter",
    "ert_data",
    "smooth",
    "read_ert",
    "get_peak_ops",
    "get_peak_int_theo",
    "get_peak_bandwidth",
    "get_theo_bandwidth_txns",
    "get_hotspots",
    "get_hotspots_integer",
    "get_color",
    "plot_parameters",
    "plot_roofline",
    "FONT_SIZE",
    "VERBOSE",
    "DEBUG",
    "RANK_INDEX",
]


# -------------------------------------------------------------------------------------- #
#
def echo_dart_tag(name, filepath, img_type):
    """
    Printing this string will upload the results to CDash when running CTest
    """
    print(
        '<DartMeasurementFile name="{}" '.format(name)
        + 'type="image/{}">{}</DartMeasurementFile>'.format(img_type, filepath)
    )


# -------------------------------------------------------------------------------------- #
#
def get_json_entry(inp, key):
    for _key in [key, key.replace("_", "-"), key.replace("-", "_")]:
        if _key in inp:
            return inp[_key]
    return None


# -------------------------------------------------------------------------------------- #
#
def flatten_list(datalist):
    return [item for sublist in datalist for item in sublist]


# -------------------------------------------------------------------------------------- #
#
def get_font():
    return {
        "size": FONT_SIZE,
        "color": FONT_COLOR,
        "weight": FONT_WEIGHT,
        "family": "serif",
    }


# -------------------------------------------------------------------------------------- #
#
class ert_params:
    """
    ERT execution parameters
    """

    def __init__(self, inp):
        self.working_set_min = get_json_entry(inp, "working_set_min")
        self.memory_max = get_json_entry(inp, "memory_max")
        self.nthreads = get_json_entry(inp, "nthreads")
        self.nrank = get_json_entry(inp, "nrank")
        self.nproc = get_json_entry(inp, "nproc")
        self.nstreams = get_json_entry(inp, "nstreams")
        self.grid_size = get_json_entry(inp, "grid_size")
        self.block_size = get_json_entry(inp, "block_size")
        self.shmem_size = get_json_entry(inp, "shmem_size")

    def __str__(self):
        ret = []
        for itr in (
            "working_set_min",
            "memory_max",
            "nthreads",
            "nrank",
            "nproc",
            "nstreams",
            "grid_size",
            "block_size",
            "shmem_size",
        ):
            ret += ["=".join([itr, "{}".format(getattr(self, itr))])]
        return ", ".join(ret)


# -------------------------------------------------------------------------------------- #
#
class ert_counter:
    """
    ERT counter data
    """

    def __init__(self, inp):
        _data = get_json_entry(inp, "repr_data")
        self.units = get_json_entry(inp, "units")
        self.display_units = get_json_entry(inp, "display_units")
        if not isinstance(_data, list) and not isinstance(_data, tuple):
            self.data = [_data]
        else:
            self.data = _data
        if self.units is not None:
            for i in range(len(self.data)):
                self.data[i] /= self.units

    def __str__(self):
        ret = []
        for itr in ("data", "units", "display_units"):
            ret += ["=".join([itr, "{}".format(getattr(self, itr))])]
        return ", ".join(ret)

    def get(self, total):
        _data = self.data
        _list = []
        for i in range(len(_data)):
            _list.append(total / _data[i])
        return _list

    def get_warp_ops(
        self, total
    ):  # for int ops, scale the peak ops by 32 for warp-based measurements
        _data = (
            self.data
        )  # we can use the same function for transaction bandwidth
        _list = []
        for i in range(len(_data)):
            _list.append((total / _data[i]) / WARP_SIZE)
        return _list


# -------------------------------------------------------------------------------------- #
#
class ert_data:
    """
    ERT instance data
    """

    def __init__(self, inp):
        self.label = get_json_entry(inp, "label")
        self.working_set = get_json_entry(inp, "working-set")
        self.trials = get_json_entry(inp, "trials")
        self.total_bytes = get_json_entry(inp, "total-bytes")
        self.total_ops = get_json_entry(inp, "total-ops")
        self.ops_per_set = get_json_entry(inp, "ops-per-set")
        self.device = get_json_entry(inp, "device")
        self.dtype = get_json_entry(inp, "dtype")
        self.counter = ert_counter(get_json_entry(inp, "counter"))
        self.exec_params = ert_params(get_json_entry(inp, "exec-params"))

    def __str__(self):
        ret = []
        for itr in (
            "label",
            "working_set",
            "trials",
            "total_bytes",
            "total_ops",
            "ops_per_set",
            "device",
            "dtype",
        ):
            ret += ["=".join([itr, "{}".format(getattr(self, itr))])]
        for itr in ("counter", "exec_params"):
            ret += [
                "".join(
                    [
                        "\n    {:>12}".format(itr),
                        ": ({})".format(getattr(self, itr)),
                    ]
                )
            ]
        return ", ".join(ret)


# -------------------------------------------------------------------------------------- #
#
def smooth(x, y):
    """
    Smooth a curve
    """
    xs = x[:]
    ys = y[:]
    d = 0
    for i in range(0, len(ys)):
        num = min(len(ys), i + d + 1) - max(0, i - d)
        _beg = max(0, i - d)
        _end = min(len(ys), i + d + 1)
        _ys = ys[_beg:_end]
        total = sum(_ys)
        ys[i] = total / float(num)
    return xs, ys


# -------------------------------------------------------------------------------------- #
#
def read_ert(inp):
    data = inp

    def _read_ert(_data):
        if "ert" not in _data:
            return None
        else:
            inst = []
            _ert = _data["ert"]
            for element in _ert:
                inst.append(ert_data(element))
            return inst

    inst = _read_ert(data)
    if inst is None:
        if "roofline" in data:
            data = data["roofline"]
        inst = _read_ert(data)

    if VERBOSE > 2:
        for entry in inst:
            print("{}\n".format(entry))

    return inst


# -------------------------------------------------------------------------------------- #
#
def get_peak_ops(roof_data, flop_info=None):
    """
    Get the peak operations / sec
    """
    peak = {}

    for element in roof_data:
        total_ops = element.total_ops / GIGABYTE
        _label = element.label
        _data = element.counter.get(
            total_ops
        )  # this gives total_ops/repr_data (time) ops/sec
        if VERBOSE > 2:
            print("LABEL: {}, DATA: {}".format(_label, _data))
        if _label not in peak:
            peak[_label] = _data
        else:
            _peak = peak[_label]
            for i in range(len(_data)):
                if i >= len(_peak):
                    _peak += [_data[i]]
                else:
                    _peak[i] = max([_peak[i], _data[i]])

    info = "GFLOPs/sec"
    if flop_info is not None:
        try:
            info_list = re.sub(r"[^\w]", " ", flop_info).split()
            if len(info_list) > 0:
                info = info_list[0] + " GFLOPs/sec"
        except Exception:
            pass

    print("PEAK: {}".format(peak))

    peak_ops = [peak, info]
    return peak_ops


# -------------------------------------------------------------------------------------- #
#
def get_peak_int_theo(_inst_peak):
    peak = {}
    peak["Theoretical Peak"] = _inst_peak
    info = "warps GIPS"
    peak_ops = [peak, info]
    print(peak)
    return peak_ops


# -------------------------------------------------------------------------------------- #
#
def get_peak_bandwidth(roof_data, band_labels):
    """
    Get multi-level bandwidth peaks - Implementation from ERT:
    https://bitbucket.org/berkeleylab/cs-roofline-toolkit
    """
    ref_intensity = 0
    work_set = []
    bandwidth_data = []

    # Read bandwidth raw data
    for element in roof_data:
        intensity = element.ops_per_set
        if ref_intensity == 0:
            ref_intensity = intensity
        if intensity != ref_intensity:
            continue
        work_set.append(element.working_set)
        total_bytes = element.total_bytes / GIGABYTE
        bandwidth_data.append(element.counter.get(total_bytes))

    fraction = 1.05
    samples = 10000
    bandwidth_data = flatten_list(bandwidth_data)

    max_bandwidth = max(bandwidth_data)
    begin = bandwidth_data.index(max_bandwidth)

    work_set = work_set[begin:]
    bandwidth_data = bandwidth_data[begin:]
    # min_bandwidth = min(bandwidth_data)

    dband = max_bandwidth / float(samples - 1)

    counts = samples * [0]
    totals = samples * [0.0]

    work_set, bandwidth_data = smooth(work_set, bandwidth_data)

    for i in range(0, samples):
        cband = i * dband
        for bandwidth in bandwidth_data:
            if bandwidth >= cband / fraction and bandwidth <= cband * fraction:
                totals[i] += bandwidth
                counts[i] += 1

    band_list = [[1000 * max_bandwidth, 1000]]

    maxc = -1
    maxi = -1

    for i in range(samples - 3, 1, -1):
        if counts[i] > 10:
            if counts[i] > maxc:
                maxc = counts[i]
                maxi = i
        else:
            if maxc > 1:
                value = float(totals[maxi]) / max(1, counts[maxi])
                if 1.20 * value < float(band_list[-1][0]) / band_list[-1][1]:
                    band_list.append([totals[maxi], counts[maxi]])
                else:
                    band_list[-1][0] += totals[maxi]
                    band_list[-1][1] += counts[maxi]
            maxc = -1
            maxi = -1

    band_info_list = ["DRAM"]
    cache_num = len(band_list) - 1

    for cache in range(1, cache_num + 1):
        band_info_list = ["L%d" % (cache_num + 1 - cache)] + band_info_list

    peak_bandwidths = []
    band_info_list = [e for e in band_info_list if e in band_labels]

    for band, band_info in zip(band_list, band_info_list):
        band_info = band_info + " GB/s"
        peak_bandwidths.append([float(band[0] / band[1]), band_info])

    return peak_bandwidths


# -------------------------------------------------------------------------------------- #
#
def get_theo_bandwidth_txns(txn_bandwidth):
    peak_bandwidths = []
    peak_bandwidths.append([float(txn_bandwidth[0]), "L1 GTXNs/s"])
    peak_bandwidths.append([float(txn_bandwidth[1]), "L2 GTXNs/s"])
    peak_bandwidths.append([float(txn_bandwidth[2]), "DRAM GTXNs/s"])
    return peak_bandwidths


# -------------------------------------------------------------------------------------- #
#
def get_hotspots(op_data, ai_data, index=None):
    """
    Get the hotspots information
    """
    import itertools

    marker = itertools.cycle(("o", ",", "^", "+", "*", ">", "<"))

    if "type" not in op_data or "type" not in ai_data:
        return []

    op_data_type = op_data["type"]
    ai_data_type = ai_data["type"]

    op_type = None
    ai_type = None
    if "cpu_roofline" in op_data_type:
        op_type = "cpu"
    if "cpu_roofline" in ai_data_type:
        ai_type = "cpu"
    if "gpu_roofline" in op_data_type:
        op_type = "gpu"
    if "gpu_roofline" in ai_data_type:
        ai_type = "gpu"

    if index is None:
        index = RANK_INDEX

    if "ranks" in op_data:
        op_data = op_data["ranks"][index]
    if "ranks" in ai_data:
        ai_data = ai_data["ranks"][index]

    op_graph_data = op_data["graph"]
    ai_graph_data = ai_data["graph"]
    hotspots = []

    avg_runtime = 0.0
    max_runtime = 0.0
    max_length = min([len(op_graph_data), len(ai_graph_data)])
    all_runtime = []

    def get_runtime(_data, extra=[]):
        opts = ["runtime", "elapsed"] + extra
        for opt in opts:
            if opt in _data.keys():
                return float(_data[opt])
            elif opt.title() in _data.keys():
                return float(_data[opt.title()])
        # print("Could not find runtime field in {}. Searched: {}".format(_data, opts))
        return None

    def get_flops(_data, extra=[]):
        opts = [
            "flops",
            "counted_ops",
            "flop_count",
            "flop_count_sp",
            "flop_count_dp",
            "flop_count_hp",
            "DP_operations",
            "SP_operations",
        ] + extra
        for opt in opts:
            if opt in _data.keys():
                value = float(_data[opt])
                if value > 0.0:
                    return value
            elif opt.title() in _data.keys():
                value = float(_data[opt.title()])
                if value > 0.0:
                    return value
        # print("Could not find flops field in {}. Searched: {}".format(_data, opts))
        return None

    def get_bandwidth(_data, extra=[]):
        opts = [
            "bandwidth",
            "counted_ins",
            "ldst_executed",
            "L/S completed",
            "Loads_Stores_completed",
        ] + extra
        for opt in opts:
            if opt in _data.keys():
                return float(_data[opt])
            elif opt.title() in _data.keys():
                return float(_data[opt.title()])
        # print("Could not find bandwidth field in {}. Searched: {}".format(_data, opts))
        return None

    def check_ignore(_ai, _op):
        if (
            "cuptiOverhead" in _ai
            or "cudaRuntime" in _ai
            or "cuptiOverhead" in _op
            or "cudaRuntime" in _op
        ):
            return True
        return False

    for i in range(0, max_length):
        if check_ignore(ai_graph_data[i]["prefix"], op_graph_data[i]["prefix"]):
            continue

        ai_repr = ai_graph_data[i]["entry"]["repr_data"]
        op_repr = op_graph_data[i]["entry"]["repr_data"]
        all_runtime += filter(
            None, [get_runtime(ai_repr), get_runtime(op_repr)]
        )

    for rt in all_runtime:
        avg_runtime += rt

    if len(all_runtime) > 1:
        max_runtime = max(all_runtime)
        avg_runtime -= max_runtime
        avg_runtime /= len(all_runtime) - 1.0

    for i in range(0, max_length):
        if check_ignore(ai_graph_data[i]["prefix"], op_graph_data[i]["prefix"]):
            continue

        runtimes = []
        flop = None
        bandwidth = None

        ai_repr = ai_graph_data[i]["entry"]["repr_data"]
        op_repr = op_graph_data[i]["entry"]["repr_data"]

        label = op_graph_data[i]["prefix"]
        runtimes += filter(None, [get_runtime(ai_repr), get_runtime(op_repr)])

        flop = get_flops(op_repr)

        if flop is None:
            print("No flops found in {}!".format(op_repr))
            continue

        if op_type == "gpu" or ai_type == "gpu":
            bandwidth = get_bandwidth(op_repr)
        elif ai_type == "cpu" or op_type == "cpu":
            bandwidth = get_bandwidth(ai_repr)

        if bandwidth is None:
            bandwidth = get_bandwidth(op_repr)
            if bandwidth is None:
                bandwidth = get_bandwidth(ai_repr)

        runtime = 0.0
        for rt in runtimes:
            runtime += rt
        runtime /= len(runtimes)

        intensity = flop / bandwidth if bandwidth != 0.0 else 0.0
        flop = flop / GIGABYTE / runtime
        proportion = runtime / avg_runtime
        label = (
            re.sub(r"^[|0-9]+", " ", label)
            .replace("> [cxx] ", "")
            .replace("> [_c_] ", "")
            .replace("> [pyc] ", "")
        )

        if VERBOSE > 1:
            print(
                "intensity: {}, flop: {}, proportion: {}, label: {}".format(
                    intensity, flop, proportion, label
                )
            )

        if VERBOSE > 0:
            print(
                "{} : runtime = {}, avg = {}, proportion = {}, flop = {}".format(
                    label, runtime, avg_runtime, proportion, flop
                )
            )

        # this can arise from overflow
        if flop <= 1.0e-3 or bandwidth <= 0.0:
            continue

        my_marker = next(marker)
        hotspots.append([intensity, flop, proportion, my_marker, label])
    return hotspots


# -------------------------------------------------------------------------------------- #
#
def get_hotspots_integer(op_data, ai_data, index=None):
    """
    Get the hotspots information
    """
    import itertools

    marker = itertools.cycle(("o", ",", "^", "+", "*", ">", "<"))
    if "type" not in op_data or "type" not in ai_data:
        return []

    op_data_type = op_data["type"]

    op_type = None
    if "cpu_roofline" in op_data_type:
        op_type = "cpu"
    if "gpu_roofline" in op_data_type:
        op_type = "gpu"

    if index is None:
        index = RANK_INDEX

    if "ranks" in op_data:
        op_data = op_data["ranks"][index]
    if "ranks" in ai_data:
        ai_data = ai_data["ranks"][index]

    op_graph_data = op_data["graph"]
    ai_graph_data = ai_data["graph"]
    hotspots = []

    # avg_runtime = 1.0
    # max_runtime = 0.0
    # max_length = min([len(op_graph_data), len(ai_graph_data)])
    all_runtime = []

    def get_runtime(_data, extra=[]):
        opts = ["runtime", "elapsed"] + extra
        for opt in opts:
            if opt in _data:
                return float(_data[opt])
            elif opt.title() in _data:
                return float(_data[opt.title()])
        return None

    def get_int_ops_scaled(_data, extra=[]):
        opts = ["inst_integer"]
        for opt in opts:
            if opt in _data:
                value = float(_data[opt])
                if value > 0.0:
                    return (
                        value / WARP_SIZE
                    )  # scale it by 32 because its a thread level counter
        return None

    def get_HBM_transactions(
        _data, extra=[]
    ):  # each transaction is 32 bytes long
        _HBM_transactions = 0
        opts = ["dram_read_transactions", "dram_write_transactions"] + extra
        for opt in opts:
            _HBM_transactions += float(_data[opt])
        if _HBM_transactions > 0.0:
            return _HBM_transactions

        else:
            print("No Memory Transactions counted")
            return 0

    def get_L2_transactions(_data, extra=[]):
        _L2_transactions = 0
        opts = ["l2_read_transactions", "l2_write_transactions"] + extra
        for opt in opts:
            _L2_transactions += float(_data[opt])
        if _L2_transactions > 0.0:
            return _L2_transactions
        else:
            return 0

    def get_L1_transactions(_data, extra=[]):
        _L1_transactions = 0
        opts = [
            "gld_transactions",
            "gst_transactions",
            "shared_store_transactions",
            "shared_load_transactions",
        ] + extra
        for opt in opts:
            if "shared" in opt:
                _L1_transactions += float(_data[opt]) * 4
            else:
                _L1_transactions += float(_data[opt])
        if _L1_transactions > 0.0:
            return _L1_transactions
        else:
            return 0

    def check_ignore(_ai, _op):
        if (
            "cuptiOverhead" in _ai
            or "cudaRuntime" in _ai
            or "cuptiOverhead" in _op
            or "cudaRuntime" in _op
        ):
            return True
        return False

    all_runtime = {}
    for i in range(0, len(ai_graph_data)):
        ai_repr = ai_graph_data[i]["entry"]["repr_data"]
        all_runtime[ai_graph_data[i]["hash"]] = get_runtime(ai_repr)

    for i in range(0, len(op_graph_data)):
        Iops = None
        HBM_transactions = None
        L1_transactions = None
        L2_transactions = None

        op_repr = op_graph_data[i]["entry"]["repr_data"]

        label = op_graph_data[i]["prefix"]
        if "thrust" in label:
            label = "thrust kernel"
        else:
            label = label.split("(")[0]

        if op_type == "gpu":
            Iops = get_int_ops_scaled(op_repr)
            HBM_transactions = get_HBM_transactions(op_repr)
            if HBM_transactions == 0:
                continue

            L1_transactions = get_L1_transactions(op_repr)
            if L1_transactions == 0:
                continue
            L2_transactions = get_L2_transactions(op_repr)
            if L2_transactions == 0:
                continue
        else:
            print("no GPU data available")

        runtime = 0.0
        runtime = all_runtime[op_graph_data[i]["hash"]]

        # hotspot for HBM
        HBM_intensity = Iops / HBM_transactions  # if bandwidth != 0.0 else 0.0
        HBM_GIPS = Iops / GIGABYTE / runtime
        HBM_proportion = 0.1  # runtime / avg_runtime

        # hotspot for L1
        L1_intensity = Iops / L1_transactions  # if bandwidth != 0.0 else 0.0
        L1_GIPS = Iops / GIGABYTE / runtime
        L1_proportion = 0.5  # runtime / avg_runtime

        # hotspot for L2
        L2_intensity = Iops / L2_transactions  # if bandwidth != 0.0 else 0.0
        L2_GIPS = Iops / GIGABYTE / runtime
        L2_proportion = 1  # runtime / avg_runtime

        # label = re.sub(r'^[|0-9]+', ' ', label).replace("> [cxx] ", "").replace(
        #    "> [_c_] ", "").replace("> [pyc] ", "")

        # if VERBOSE > 1:
        #     print("intensity: {}, flop: {}, proportion: {}, label: {}".format(
        #         intensity, flop, proportion, label))
        #
        # if VERBOSE > 0:
        #     print("{} : runtime = {}, avg = {}, proportion = {}, flop = {}".format(
        #         label, runtime, avg_runtime, proportion, flop))
        #
        # # this can arise from overflow
        # if flop <= 1.0e-3 or bandwidth <= 0.0:
        #     continue

        my_marker = next(marker)
        hotspots.append(
            [
                HBM_intensity,
                HBM_GIPS,
                HBM_proportion,
                my_marker,
                "HBM: " + label,
            ]
        )
        hotspots.append(
            [L1_intensity, L1_GIPS, L1_proportion, my_marker, "L1: " + label]
        )
        hotspots.append(
            [L2_intensity, L2_GIPS, L2_proportion, my_marker, "L2: " + label]
        )
    return hotspots


# -------------------------------------------------------------------------------------- #
#
def get_color(proportion):
    if proportion < 0.2:
        color = "green"
    elif proportion < 0.6:
        color = "blue"
    else:
        color = "red"
    return color


# -------------------------------------------------------------------------------------- #
#
class plot_parameters:
    def __init__(self, peak_flops, hotspots, _inst_roofline):
        _peak = peak_flops[0]
        if isinstance(_peak, list):
            _peak = max(_peak)
        elif isinstance(_peak, dict):
            tmp = 0.0
            for key, entry in _peak.items():
                tmp = max([tmp] + entry)
            _peak = tmp

        y_digits = int(math.log10(_peak)) + 1
        self.xmin = 0.01
        self.xmax = 1000
        self.ymin = 1
        self.ymax = 10**y_digits

        if _inst_roofline:
            self.xlabel = "Instruction Intensity [Warp Inst. Per Transaction]"
            self.ylabel = "Performance [Warp GIPS]"
        else:
            self.xlabel = "Arithmetic Intensity [FLOPs/Byte]"
            self.ylabel = "Performance [GFLOPs/sec]"

        for element in hotspots:
            intensity = element[0]
            flop = element[1]
            if flop > self.ymax:
                self.ymax = 10 ** int(log(flop) / log(10) + 1)
            if flop < self.ymin:
                self.ymin = 10 ** int(log(flop) / log(10) - 1)
            if intensity > self.xmax:
                self.xmax = 100 ** int(log(intensity) / log(10) + 1)
            if intensity < self.xmin:
                self.xmin = 10 ** int(log(intensity) / log(10) - 1)

        print(
            "X (min, max) = {}, {}, Y (min, max) = {}, {}".format(
                self.xmin, self.xmax, self.ymin, self.ymax
            )
        )


#
# -------------------------------------------------------------------------------------- #
#
def plot_roofline(
    ai_data,
    op_data,
    *_args,
    **_kwargs,
):
    _ai = []
    _op = []

    if "timemory" in ai_data:
        for key, data in ai_data["timemory"].items():
            _ai.append(data)
    else:
        _ai.append(ai_data)

    if "timemory" in op_data:
        for key, data in op_data["timemory"].items():
            _op.append(data)
    else:
        _op.append(op_data)

    for ai, op in zip(_ai, _op):
        plot_roofline_impl(ai, op, *_args, **_kwargs)


# -------------------------------------------------------------------------------------- #
#
def plot_roofline_impl(
    ai_data,
    op_data,
    band_labels,
    txn_bandwidths,
    inst_peak,
    _rtype,
    display=False,
    fname="roofline",
    image_type="png",
    output_dir=os.getcwd(),
    title="Roofline Plot",
    width=1600,
    height=1200,
    dpi=100,
    echo_dart=False,
    index=None,
):
    """Plot the roofline"""

    if index is None:
        index = RANK_INDEX

    inst_roofline = False
    if "gpu_roofline_inst" in _rtype:
        inst_roofline = True
        print("GPU INST ROOFLINE ON")
    else:
        print("GPU INST ROOFLINE OFF")

    band_data = read_ert(ai_data)
    peak_data = read_ert(op_data)

    info = op_data["unit_repr"] if "unit_repr" in op_data else None

    if inst_roofline:
        peak_flop = get_peak_int_theo(inst_peak)
    else:
        peak_flop = get_peak_ops(peak_data, info)

    if inst_roofline:
        peak_band = get_theo_bandwidth_txns(txn_bandwidths)
    else:
        peak_band = get_peak_bandwidth(band_data, band_labels)

    if inst_roofline:
        hotspots = get_hotspots_integer(op_data, ai_data, index)
    else:
        hotspots = get_hotspots(op_data, ai_data, index)

    print("peak_flop = {}, peak_band = {}".format(peak_flop, peak_band))

    plot_params = plot_parameters(peak_flop, hotspots, inst_roofline)

    f = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    f.add_subplot(111)

    _title_font = get_font()
    _title_font["size"] += 8
    plt.title(title.title(), **_title_font)
    plt.grid(True, which="major", ls="--", lw=1)
    plt.grid(True, which="minor", ls="--", lw=0.5)
    plt.yscale("log")
    plt.xscale("log")

    plt.xlabel(plot_params.xlabel, **get_font())
    plt.ylabel(plot_params.ylabel, **get_font())

    axes = plt.gca()
    axes.set_xlim([plot_params.xmin, plot_params.xmax])
    axes.set_ylim([plot_params.ymin, plot_params.ymax])

    # _peak_flop = peak_flop[0]
    # if isinstance(_peak_flop, list):
    #    _peak_flop = max(_peak_flop)

    # plot bandwidth roof
    _nitr = 0
    for _label, _peak_flop in peak_flop[0].items():
        x0 = plot_params.xmax
        print("Label: {}, Peak: {}".format(_label, _peak_flop))
        _peakop = max(_peak_flop)
        for band in peak_band:
            x1 = plot_params.xmin
            y1 = band[0] * plot_params.xmin
            if y1 < plot_params.ymin:
                x1 = plot_params.ymin / band[0]
                y1 = plot_params.ymin
            x2 = _peakop / band[0]
            y2 = _peakop
            if x2 < x0:
                x0 = x2

            x1log = log(x1) / log(10)
            x2log = log(x2) / log(10)
            y1log = log(y1) / log(10)
            y2log = log(y2) / log(10)
            x_text = 10 ** ((x1log + x2log) / 2)
            y_text = 10 ** ((y1log + y2log) / 2)

            fig = plt.gcf()
            size = fig.get_size_inches() * fig.dpi
            fig_x, fig_y = size

            dx = log(x2) - log(x1)
            dy = log(y2) - log(y1)
            x_min, x_max = plt.xlim()
            y_min, y_max = plt.ylim()
            Dx = dx * fig_x / (log(x_max) - log(x_min))
            Dy = dy * fig_y / (log(y_max) - log(y_min))
            fdiv = 1.0
            angle = (180.0 / pi) * numpy.arctan(Dy / Dx / fdiv)

            if _nitr == 0:
                text(
                    x_text,
                    y_text,
                    "%.2f %s" % (band[0], band[1]),
                    rotation=angle,
                    rotation_mode="anchor",
                    **get_font(),
                )
            plt.plot([x1, x2], [y1, y2], color="magenta")

            # plot computing roof
            temp_label = _label.replace("_", "-")
            temp_label = temp_label.upper()
            text(
                plot_params.xmax,
                _peakop + 2,
                "%.2f %s %s" % (_peakop, temp_label, peak_flop[1]),
                horizontalalignment="right",
                **get_font(),
            )

            plt.plot([x0, plot_params.xmax], [_peakop, _peakop], color="b")
        # ensure bandwidth labels are not duplicated
        _nitr += 1

    # from random import random
    import pylab

    labels = []
    # plot hotspots
    plotted_spots = []
    for element in hotspots:
        if VERBOSE > 0:
            print(element[0], element[1])
        c = get_color(element[2])
        myScatter = plt.scatter(element[0], element[1], c=c, marker=element[3])
        plotted_spots.append(myScatter)
        # factor = 10.0 * random() - 5.0
        label = "{}".format(element[4]).replace(">>>", "")

        # plt.annotate(label, (element[0], element[1] + factor), **get_font())
        labels.append(label)
        # text(element[0], element[1], "%s" %
        #     element[3], rotation=0, rotation_mode='anchor')

    # try:
    #    from adjustText import adjust_text
    print(labels)
    try:
        pylab.legend(
            plotted_spots,
            labels,
            prop={"size": (FONT_SIZE - 2)},
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
        )
    except Exception as e:
        sys.stderr.write(f"{e}\n")
        pylab.legend(
            plotted_spots,
            labels,
            prop={"size": (FONT_SIZE - 2)},
        )

    # plt.legend(labels)
    if display:
        print("Displaying plot...")
        plt.show()
    else:
        imgfname = os.path.join(output_dir, os.path.basename(fname))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not ".{}".format(image_type) in imgfname:
            imgfname += ".{}".format(image_type)
        print('Saving plot: "{}"...'.format(imgfname))
        plt.savefig(imgfname, bbox_inches="tight")
        plt.close()
        if echo_dart:
            bfname = os.path.basename(fname)
            echo_dart_tag(bfname, imgfname, image_type)
