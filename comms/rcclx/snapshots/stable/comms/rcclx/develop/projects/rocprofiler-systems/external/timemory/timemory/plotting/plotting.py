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

""" @file plotting.py
Plotting routines for timemory module
"""

from __future__ import absolute_import
from __future__ import division

__author__ = "Jonathan Madsen"
__copyright__ = "Copyright 2020, The Regents of the University of California"
__credits__ = ["Jonathan Madsen"]
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "Jonathan Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"
__all__ = [
    "plot",
    "plot_maximums",
    "plot_all",
    "plot_generic",
    "read",
    "plot_data",
    "timemory_data",
    "echo_dart_tag",
    "add_plotted_files",
    "make_output_directory",
    "nested_dict",
    "plot_parameters",
    "plotted_files",
]

import sys
import os
import copy
import json
import warnings
import traceback
import collections

__dir__ = os.path.realpath(os.path.dirname(__file__))

if os.environ.get("DISPLAY") is None and os.environ.get("MPLBACKEND") is None:
    os.environ.setdefault("MPLBACKEND", "agg")

_matplotlib_backend = None
# a previous bug inserted wrong value for levels, this gets around it
_excess_levels = 64

# check with timemory.options
#
try:
    import timemory.options

    if timemory.options.matplotlib_backend != "default":
        _matplotlib_backend = timemory.options.matplotlib_backend
except ImportError:
    pass


# if not display variable we probably want to use agg
#
if (
    os.environ.get("DISPLAY") is None
    and os.environ.get("MPLBACKEND") is None
    and _matplotlib_backend is None
):
    os.environ.setdefault("MPLBACKEND", "agg")


# tornado helps set the matplotlib backend but is not necessary
#
try:
    import tornado  # noqa: F401
except ImportError:
    pass


# import matplotlib and pyplot but don't fail
#
try:
    import matplotlib
    import matplotlib.pyplot as plt_default_backend  # noqa: F401

    _matplotlib_backend = matplotlib.get_backend()
except ImportError:
    try:
        import matplotlib

        matplotlib.use("agg", warn=False)
        import matplotlib.pyplot as plt_try_agg_backend  # noqa: F401

        _matplotlib_backend = matplotlib.get_backend()
    except ImportError:
        pass


#
_is_ci = (
    False if os.environ.get("CONTINUOUS_INTEGRATION", None) is None else True
)

#
_default_scale = 1.0 if not _is_ci else 0.6
""" Scale the image down during continuous integration """

#
_default_min_percent = 0.05  # 5% of max
""" Default minimum percent of max when reducing # of timing functions plotted """

_default_img_dpi = 60
""" Default image dots-per-square inch """

_default_img_size = {
    "w": int(1000 / _default_scale),
    "h": int(600 / _default_scale),
}
""" Default image size """

_default_img_type = "png"
""" Default image type """

_default_log_x = False
"""Log scaled X axis"""

_default_font_size = 11 if not _is_ci else 8
"""Font size for y-axis labels"""

plotted_files = []
""" A list of all files that have been plotted """

verbosity = int(os.environ.get("TIMEMORY_VERBOSE", 0))

# -------------------------------------------------------------------------------------- #


class plot_parameters:
    """
    A class for reducing the amount of data in plot by specifying a minimum
    percentage of the max value and the fields to check against
    """

    min_percent = copy.copy(_default_min_percent)
    """ Global plotting params (these should be modified instead of _default_*) """

    img_dpi = copy.copy(_default_img_dpi)
    """ Global image plotting params (these should be modified instead of _default_*) """
    img_size = copy.copy(_default_img_size)
    """ Global image plotting params (these should be modified instead of _default_*) """
    img_type = copy.copy(_default_img_type)
    """ Global image plotting params (these should be modified instead of _default_*) """
    log_xaxis = copy.copy(_default_log_x)
    """ Global image plotting params (these should be modified instead of _default_*) """
    font_size = copy.copy(_default_font_size)
    """ Global image plotting params (these should be modified instead of _default_*) """

    def __init__(
        self,
        min_percent=min_percent,
        img_dpi=img_dpi,
        img_size=img_size,
        img_type=img_type,
        log_xaxis=log_xaxis,
        font_size=font_size,
    ):
        self.min_percent = min_percent
        self.img_dpi = img_dpi
        self.img_size = img_size
        self.img_type = img_type
        # max values
        self.max_value = 0.0
        self.log_xaxis = log_xaxis
        self.font_size = font_size

    def __str__(self):
        _c = "Image: {} = {}%, {} = {}, {} = {}, {} = {}".format(
            "min %",
            self.min_percent,
            "dpi",
            self.img_dpi,
            "size",
            self.img_size,
            "type",
            self.img_type,
        )
        return "[{}]".format(_c)


# -------------------------------------------------------------------------------------- #
def echo_dart_tag(name, filepath, img_type=plot_parameters.img_type):
    """
    Printing this string will upload the results to CDash when running CTest
    """
    print(
        '<DartMeasurementFile name="{}" '.format(name)
        + 'type="image/{}">{}</DartMeasurementFile>'.format(img_type, filepath)
    )


# -------------------------------------------------------------------------------------- #
def add_plotted_files(name, filepath, echo_dart):
    """Adds a file to the plotted file list and print CDash dart string"""
    global plotted_files
    if echo_dart:
        filerealpath = os.path.realpath(filepath)
        echo_dart_tag(name, filerealpath)
    found = False
    for p in plotted_files:
        if p[0] == name and p[1] == filepath:
            found = True
            break
    if found is False:
        plotted_files.append([name, filepath])


# -------------------------------------------------------------------------------------- #
def make_output_directory(directory):
    """mkdir -p"""
    if not os.path.exists(directory) and directory != "":
        os.makedirs(directory)


# -------------------------------------------------------------------------------------- #
def nested_dict():
    return collections.defaultdict(nested_dict)


# -------------------------------------------------------------------------------------- #
class timemory_data:
    """This class is for internal usage. It holds the JSON data"""

    # ------------------------------------------------------------------------ #

    def __init__(self, func, obj, stats=None):
        """initialize data from JSON object"""
        self.func = func

        def process(inp, key, fallbacks=[]):
            inst = None
            if inp is None:
                return inst
            if key not in inp:
                if len(fallbacks) == 0:
                    return inst
                else:
                    return process(inp, fallbacks[0], fallbacks[1:])

            if isinstance(inp[key], dict):
                inst = []
                for key, item in inp[key].items():
                    inst.append(item)
            elif isinstance(inp[key], list):
                inst = inp[key]
            else:
                inst = inp[key]
            return inst

        self.laps = process(obj, "laps")
        self.data = process(obj, "repr_data", ["repr"])
        self.value = process(obj, "value")
        self.accum = process(obj, "accum", ["value"])
        if stats is not None:
            self.sum = process(stats, "sum")
            self.sqr = process(stats, "sqr")
            self.min = process(stats, "min")
            self.max = process(stats, "max")

    # ------------------------------------------------------------------------ #
    def plottable(self, params):
        """valid data above minimum"""
        # compute the minimum values
        _min = (0.01 * params.min_percent) * params.max_value

        # function for checking passes test
        return abs(self.accum) > _min

    # ------------------------------------------------------------------------ #
    def __str__(self):
        """printing data"""
        return '"{}" : laps = {}, value = {}, accum = {}, data = {}'.format(
            self.func, self.laps, self.value, self.accum, self.data
        )

    # ------------------------------------------------------------------------ #
    def __repr__(self):
        """printing data"""
        print("{}".format(self))

    # ------------------------------------------------------------------------ #
    def __add__(self, rhs):
        """for combining results (typically from different MPI processes)"""
        self.laps += rhs.laps
        self.value += rhs.value
        self.accum += rhs.accum
        self.data += rhs.data
        return self

    # ------------------------------------------------------------------------ #
    def __sub__(self, rhs):
        """for differencing results (typically from two different runs)"""
        self.value -= rhs.value
        self.accum -= rhs.accum
        self.data -= rhs.data
        # this is a weird situation
        if self.laps != rhs.laps:
            self.laps = max(self.laps, rhs.laps)
        return self

    # ------------------------------------------------------------------------ #
    def reset(self):
        """clear all the data"""
        self.laps = 0
        self.value = 0
        self.accum = 0
        self.data = 0.0

    # ------------------------------------------------------------------------ #
    def __getitem__(self, key):
        """array indexing"""
        return getattr(self, key)


# -------------------------------------------------------------------------------------- #
class plot_data:
    """A custom configuration for the data to be plotted

    Args:
        filename (str):
            the input filename (JSON file) to be processed
            also used to generate the output image name unless "output_name"
            is specified
        concurrency (int):
            the concurrency used
        mpi_size (int):
            the number of MPI processes
        title (str):
            title for plot
        plot_params (plot_parameters instance):
            the plotting parameters
        output_name (str):
    """

    def __init__(
        self,
        filename="output",
        concurrency=1,
        mpi_size=1,
        timemory_functions=[],
        title="",
        units="",
        ctype="",
        cid="",
        description="",
        plot_params=plot_parameters(),
        output_name=None,
    ):
        def _convert(_obj):
            if isinstance(_obj, dict):
                _ret = []
                for key, item in _obj.items():
                    _ret.append(item)
                return _ret
            else:
                return _obj

        self.cid = cid
        self.units = _convert(units)
        self.ctype = _convert(ctype)
        self.description = _convert(description)

        self.nitems = 1
        if isinstance(self.ctype, list):
            self.nitems = len(self.ctype)

        self.timemory_functions = timemory_functions
        self.concurrency = concurrency
        self.mpi_size = mpi_size
        self.title = title
        self.plot_params = plot_params

        outname = output_name if output_name is not None else filename
        if self.nitems == 1:
            self.filename = filename
            self.output_name = outname
        else:
            self.filename = []
            self.output_name = []
            for n in range(0, self.nitems):
                tag = "_".join(self.ctype[n].lower().split())
                self.filename.append("_".join([filename, tag]))
                self.output_name.append("_".join([outname, tag]))

    # ------------------------------------------------------------------------ #
    def update_parameters(self, params=None):
        """Update plot parameters (i.e. recalculate maxes)"""
        if params is not None:
            self.plot_params = params
        self.plot_params.max_value = 0.0
        # calc max values
        for key, obj in self.timemory_functions.items():
            _max = self.plot_params.max_value
            _val = obj.accum
            if isinstance(_val, list):
                self.plot_params.max_value = max([_max] + _val)
            else:
                self.plot_params.max_value = max([_max, _val])

    # ------------------------------------------------------------------------ #
    def __len__(self):
        """Get the length"""
        return len(self.timemory_functions)

    # ------------------------------------------------------------------------ #
    def keys(self):
        """Get the dictionary keys"""
        return self.timemory_functions.keys()

    # ------------------------------------------------------------------------ #
    def items(self):
        """Get the dictionary items"""
        return self.timemory_functions.items()

    # ------------------------------------------------------------------------ #
    def __str__(self):
        """String repr"""
        _list = [
            ("Filename", self.filename),
            ("Concurrency", self.concurrency),
            ("MPI ranks", self.mpi_size),
            ("# functions", len(self)),
            ("Title", self.title),
            ("Parameters", self.plot_params),
        ]
        _str = "\n"
        for entry in _list:
            _str = '{}\t{} : "{}"\n'.format(_str, entry[0], entry[1])
        return _str

    # ------------------------------------------------------------------------ #
    def get_title(self):
        """Construct the title for the plot"""
        return "{} @ Processes = {}, Threads/process = {}".format(
            self.title, self.mpi_size, int(self.concurrency)
        )


# -------------------------------------------------------------------------------------- #
def read(data, **_kwargs):
    """Read the graph data"""

    # some fields
    timemory_functions = nested_dict()

    if "graph_size" in data:
        # loop over ranks
        ngraph = int(data["graph_size"])
        for i in range(0, ngraph):
            _data = data["graph"][i]
            tag = _data["prefix"]

            _stats = _data["stats"] if "stats" in _data else None
            tfunc = timemory_data(tag, _data["entry"], _stats)

            if tfunc.laps == 0:
                continue

            if tag not in timemory_functions:
                # create timemory_data object if doesn't exist yet
                timemory_functions[tag] = tfunc
            else:
                # append
                timemory_functions[tag] += tfunc
    else:
        for tag, itr in data.items():
            _stats = itr["stats"] if "stats" in itr else None
            tfunc = timemory_data(tag, itr, _stats)

            if tfunc.laps == 0:
                continue

            if tag not in timemory_functions:
                # create timemory_data object if doesn't exist yet
                timemory_functions[tag] = tfunc
            else:
                # append
                timemory_functions[tag] += tfunc

    # return a plot_data object
    return plot_data(**_kwargs, timemory_functions=timemory_functions)


# -------------------------------------------------------------------------------------- #
def plot_generic(_plot_data, _type_min, _type_unit, idx=0):
    """Generic plotting routine"""
    if _matplotlib_backend is None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            warnings.warn(
                "Matplotlib could not find a suitable backend. Skipping plotting..."
            )
            return

    import numpy as np
    import matplotlib.pyplot as plt

    def get_obj_idx(_obj, _idx):
        if isinstance(_obj, list):
            return _obj[_idx]
        else:
            return _obj

    font = {
        "family": "serif",
        "color": "black",
        "weight": "bold",
        "size": _plot_data.plot_params.font_size,
    }

    filename = [get_obj_idx(_plot_data.filename, idx)]
    plot_params = _plot_data.plot_params
    nitem = len(_plot_data)
    ntics = len(_plot_data)
    ytics = []
    types = [get_obj_idx(_plot_data.ctype, idx)]

    if len(types) == 0 or (len(types) == 1 and len(types[0]) == 0):
        return False

    # print("Plot types: {}".format(types))

    if ntics == 0:
        print(
            "{} had no data less than the minimum time ({} {})".format(
                filename, _type_min, _type_unit
            )
        )
        return False

    avgs = []
    stds = []

    for func, obj in _plot_data.items():
        func = func.strip(">")
        _len = len(func)
        if _len > 30:
            func = "{}...{}".format(
                func[:10], func[(_len - 20) :]  # noqa: E203
            )
        ytics.append("{} x [ {} counts ]".format(func, obj.laps))
        avgs.append(get_obj_idx(obj.data, idx))
        stds.append(0.0)

    # the x locations for the groups
    ind = np.arange(ntics)
    # the thickness of the bars: can also be len(x) sequence
    thickness = 1.0

    f = plt.figure(
        figsize=(
            plot_params.img_size["w"] / plot_params.img_dpi,
            plot_params.img_size["h"] / plot_params.img_dpi,
        ),
        dpi=plot_params.img_dpi,
    )
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    f.subplots_adjust(left=0.05, right=0.75, bottom=0.10, top=0.90)

    # put largest at top
    ytics.reverse()
    avgs.reverse()
    stds.reverse()

    # construct the plots
    _colors = None
    if len(types) == 1:
        _colors = ["grey", "darkblue"]
    plt.barh(
        ind,
        avgs,
        thickness,
        xerr=stds,
        alpha=1.0,
        antialiased=False,
        color=_colors,
    )

    grid_lines = []
    for i in range(0, ntics):
        if i % nitem == 0:
            grid_lines.append(i)

    plt.yticks(ind, ytics, ha="left", **font)
    plt.setp(ax.get_yticklabels(), **font)
    plt.setp(ax.get_xticklabels(), **font)
    # plt.setp(ax.get_yticklabels(), fontsize='small')

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.90, box.height])

    # Add grid
    ax.xaxis.grid(which="both")
    ax.yaxis.grid()
    if plot_params.log_xaxis:
        ax.set_xscale("log")
    return True


# -------------------------------------------------------------------------------------- #
def plot_all(_plot_data, disp=False, output_dir=".", echo_dart=False):
    #
    if _matplotlib_backend is None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            warnings.warn(
                "Matplotlib could not find a suitable backend. Skipping plotting..."
            )
            return

    import matplotlib.pyplot as plt

    def get_obj_idx(_obj, _idx):
        if isinstance(_obj, list):
            return _obj[_idx]
        else:
            return _obj

    for idx in range(0, _plot_data.nitems):
        title = _plot_data.get_title()
        params = _plot_data.plot_params

        _fname = get_obj_idx(_plot_data.filename, idx)
        _ctype = get_obj_idx(_plot_data.ctype, idx)
        if _plot_data.nitems > 1:
            _fname = "{}_{}".format(_fname, "_".join(_ctype.lower().split()))
        _units = get_obj_idx(_plot_data.units, idx)
        _desc = get_obj_idx(_plot_data.description, idx)
        _ctype = get_obj_idx(_plot_data.ctype, idx)
        _plot_min = params.max_value * (0.01 * params.min_percent)

        _do_plot = plot_generic(_plot_data, _plot_min, _units, idx)

        if not _do_plot:
            return

        _xlabel = "{}".format(_ctype.title())
        if _units:
            _xlabel = "{} [{}]".format(_xlabel, _units)

        #
        subfont = {
            "family": "serif",
            "color": "black",
            "size": _plot_data.plot_params.font_size + 2,
        }

        font = {
            "family": "serif",
            "color": "black",
            "weight": "bold",
            "size": _plot_data.plot_params.font_size + 5,
        }

        plt.xlabel(_xlabel, **font)
        title = title.replace("_", " ").title()
        plt.suptitle("{}".format(title), **font)
        if _desc:
            # just the first sentence
            if "." in _desc:
                _desc = _desc.split(".")[0]
            if "(" in _desc:
                _desc = _desc.split("(")[0]
            plt.title("{}".format(_desc.title()), **subfont)

        if disp:
            print("Displaying plot...")
            plt.show()
        else:
            make_output_directory(output_dir)
            imgfname = os.path.basename(_fname)
            imgfname = imgfname.replace(".json", ".{}".format(params.img_type))
            imgfname = imgfname.replace(".py", ".{}".format(params.img_type))
            if not ".{}".format(params.img_type) in imgfname:
                imgfname += ".{}".format(params.img_type)
            while ".." in imgfname:
                imgfname = imgfname.replace("..", ".")

            imgfname = os.path.join(output_dir, imgfname)
            if verbosity > 0:
                print("Opening '{}' for output...".format(imgfname))
            else:
                lbl = os.path.basename(imgfname).replace(
                    ".{}".format(params.img_type), ""
                )
                print("[{}]|0> Outputting '{}'...".format(lbl, imgfname))
            plt.savefig(imgfname, dpi=params.img_dpi)
            plt.close()
            if verbosity > 0:
                print("Closed '{}'...".format(imgfname))

            add_plotted_files(imgfname, imgfname, echo_dart)


def plot_maximums(
    output_name,
    title,
    data,
    plot_params=plot_parameters(),
    display=False,
    output_dir=".",
    echo_dart=None,
):
    """
    A function to plot JSON data

    Args:
        - data (list):
            - list of "plot_data" objects
            - should contain their own plot_parameters object
        - files (list):
            - list of JSON files
            - "plot_params" argument object will be applied to these files
        - combine (bool):
            - if specified, the plot_data objects from "data" and "files"
              will be combined into one "plot_data" object
            - the plot_params object will be used for this
    """
    try:
        # try here in case running as main on C++ output
        import timemory.options as options

        if echo_dart is None and options.echo_dart:
            echo_dart = True
        elif echo_dart is None:
            echo_dart = False
    except ImportError:
        pass

    _combined = None
    for _data in data:
        if _combined is None:
            _combined = plot_data(
                filename=output_name,
                output_name=output_name,
                concurrency=_data.concurrency,
                mpi_size=_data.mpi_size,
                timemory_functions={},
                title=title,
                plot_params=plot_params,
            )

        _key = list(_data.timemory_functions.keys())[0]
        _obj = _data.timemory_functions[_key]
        _obj_name = "{}".format(_data.filename)
        _obj.func = _obj_name
        _combined.timemory_functions[_obj_name] = _obj

    try:
        # print('Plotting {}...'.format(_combined.filename))
        plot_all(_combined, display, output_dir, echo_dart)

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print("Exception - {}".format(e))
        print('Error! Unable to plot "{}"...'.format(_combined.filename))


def plot(
    data=[],
    files=[],
    plot_params=plot_parameters(),
    combine=False,
    display=False,
    output_dir=".",
    echo_dart=None,
):
    """
    A function to plot JSON data

    Args:
        - data (list):
            - list of "plot_data" objects
            - should contain their own plot_parameters object
        - files (list):
            - list of JSON files
            - "plot_params" argument object will be applied to these files
        - combine (bool):
            - if specified, the plot_data objects from "data" and "files"
              will be combined into one "plot_data" object
            - the plot_params object will be used for this
    """
    try:
        # try here in case running as main on C++ output
        import timemory.options as options

        if echo_dart is None and options.echo_dart:
            echo_dart = True
        elif echo_dart is None:
            echo_dart = False
    except ImportError:
        pass

    if len(files) > 0:
        for filename in files:
            # print('Reading {}...'.format(filename))
            f = open(filename, "r")
            _data = read(json.load(f))
            f.close()
            _data.filename = filename
            _data.title = filename
            data.append(_data)

    data_sum = None
    if combine:
        for _data in data:
            if data_sum is None:
                data_sum = _data
            else:
                data_sum += _data
        data_sum.update_parameters(plot_params)
        data = [data_sum]

    for _data in data:
        try:
            # print('Plotting {}...'.format(_data.filename))
            plot_all(_data, display, output_dir, echo_dart)

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, limit=5
            )
            print("Exception - {}".format(e))
            print('Error! Unable to plot "{}"...'.format(_data.filename))
