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

""" @file plotting.py
Table generation routines for timemory package
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
    "timem",
]

import os
import sys
from .plotting import make_output_directory, add_plotted_files

# use the matplotlib import stuff from plotting
try:
    import matplotlib.pyplot as plt

except ImportError as e:
    print(f"{e}")
    pass


verbosity = os.environ.get("TIMEMORY_VERBOSE", 0)


def timem(data, fname, args):
    cmd = "unknown"
    if "command_line" in data:
        cmd = data["command_line"][1]

    nranks = len(data["timem"])
    row_labels = []
    col_labels = []
    table_vals = []

    min_data = {}
    max_data = {}
    sum_data = {}
    int_data = {}

    for rank in range(nranks):
        col_labels.append("Rank {}".format(rank))
        i = 0
        for key, entry in data["timem"][rank].items():
            if isinstance(entry["value"], (list, dict)):
                sys.stderr.write(
                    f"Skipping multi-dimensional entry for {key}\n"
                )
                continue
            if rank == 0:
                unit = entry["unit_repr"]
                if unit:
                    unit = " [{}]".format(unit)
                lbl = "{}{}".format(key.replace("_", " ").title(), unit)
                row_labels.append(lbl)
                min_data[i] = entry["repr"]
                max_data[i] = entry["repr"]
                sum_data[i] = 0
            if len(table_vals) < i + 1:
                table_vals.append([])
            value = entry["repr"]
            strval = "{}".format(value)
            min_data[i] = min([min_data[i], value])
            max_data[i] = max([max_data[i], value])
            sum_data[i] += value
            if "." in strval:
                table_vals[i].append("{:.3f}".format(value))
                int_data[i] = False
            else:
                table_vals[i].append("{}".format(value))
                if i not in int_data:
                    int_data[i] = True
            i += 1

    if nranks > 1:
        col_labels = ["Sum", "Mean"] + col_labels + ["Min", "Max"]
        i = 0
        for i in range(len(row_labels)):
            _mean = sum_data[i] / nranks
            if int_data[i]:
                table_vals[i] = (
                    ["{}".format(sum_data[i]), "{:.3f}".format(_mean)]
                    + table_vals[i]
                    + ["{}".format(min_data[i]), "{}".format(max_data[i])]
                )
            else:
                table_vals[i] = (
                    ["{:.3f}".format(sum_data[i]), "{:.3f}".format(_mean)]
                    + table_vals[i]
                    + [
                        "{:.3f}".format(min_data[i]),
                        "{:.3f}".format(max_data[i]),
                    ]
                )
            i += 1

    # print(f"cols: {col_labels}")
    # print(f"rows: {row_labels}")
    # print(f"vals: {table_vals}")
    #
    font = {
        "family": "serif",
        "color": "black",
        "weight": "bold",
        "size": 20,
    }

    if args.titles is not None:
        plt.suptitle("{}".format(args.titles[0].title()), **font)
    else:
        plt.suptitle(f"{cmd}", **font)

    # Draw table
    the_table = plt.table(
        cellText=table_vals,
        colWidths=[0.2] * len(col_labels),
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(16)
    the_table.scale(2, 2)

    # Removing ticks and spines enables you to get the figure only with table
    plt.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    plt.tick_params(
        axis="y", which="both", right=False, left=False, labelleft=False
    )
    for pos in ["right", "top", "bottom", "left"]:
        plt.gca().spines[pos].set_visible(False)

    output_dir = "."
    try:
        output_dir = args.output_dir
    except AttributeError:
        pass

    if args.display_plot:
        print("Displaying plot...")
        plt.show()
    else:
        make_output_directory(output_dir)
        imgfname = (
            os.path.basename(fname)
            .replace(".json", ".{}".format(args.img_type))
            .replace(".py", ".{}".format(args.img_type))
        )
        if not ".{}".format(args.img_type) in imgfname:
            imgfname += ".{}".format(args.img_type)
        while ".." in imgfname:
            imgfname = imgfname.replace("..", ".")
        add_plotted_files(
            imgfname, os.path.join(output_dir, imgfname), args.echo_dart
        )

        imgfname = os.path.join(output_dir, imgfname)
        if verbosity > 0:
            print("Opening '{}' for output...".format(imgfname))
        else:
            lbl = os.path.basename(imgfname).strip(".{}".format(args.img_type))
            print("[{}]|0> Outputting '{}'...".format(lbl, imgfname))
        plt.savefig(imgfname, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        if verbosity > 0:
            print("Closed '{}'...".format(imgfname))

    # fig, ax = plt.subplots(figsize=(16, 8))

    # for step, group in df.groupby('step', observed=True):
    #     d = group['ipc'].rolling('30s').agg(['mean', 'std']).resample('10s').mean()
    #     mean = d['mean']
    #     std = d['std']
    #     mean.plot(ax=ax, label=step, color=color)
    #     std = d['std']
    #     ax.fill_between(std.index, mean - 0.5*std, mean + 0.5*std, alpha=0.2,
    #                     color=color)

    # _ = ax.set_title(f'Instructions Per Cycle (IPC) by step jobid = {jobid}')
    # _ = ax.set_ylabel('IPC')
    # _ = ax.legend(ncol=2, title='job step')

    # buf = io.BytesIO()
    # plt.savefig(buf,  format='png')
    # buf.seek(0)
    # return base64.b64encode(buf.read())
