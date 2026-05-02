# -*- coding: utf-8 -*-

# Copyright (c) 2014-2018, Joe Rickerby and contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from ..version import __version__

import pandas as pd
import numpy as np
import warnings
from ..util.colormaps import ColorMaps


class ConsoleRenderer:
    def __init__(self, unicode=False, color=False):
        self.unicode = unicode
        self.color = color
        self.visited = []

    def render(self, roots, dataframe, **kwargs):
        result = self.render_preamble()

        if roots is None:
            result += "The graph is empty.\n\n"
            return result

        self.metric_columns = kwargs["metric_column"]
        self.precision = kwargs["precision"]
        self.name = kwargs["name_column"]
        self.expand = kwargs["expand_name"]
        self.context = kwargs["context_column"]
        self.rank = kwargs["rank"]
        self.thread = kwargs["thread"]
        self.depth = kwargs["depth"]
        self.highlight = kwargs["highlight_name"]
        self.colormap = kwargs["colormap"]
        self.invert_colormap = kwargs["invert_colormap"]

        if self.color:
            self.colors = self.colors_enabled
            # set the colormap based on user input
            self.colors.colormap = ColorMaps().get_colors(
                self.colormap, self.invert_colormap
            )
        else:
            self.colors = self.colors_disabled

        if isinstance(self.metric_columns, str):
            self.primary_metric = self.metric_columns
            self.second_metric = None
        elif isinstance(self.metric_columns, list):
            self.primary_metric = self.metric_columns[0]
            if len(self.metric_columns) > 2:
                warnings.warn(
                    "More than 2 metrics specified in metric_column=. Tree() will only show 2 metrics at a time. The remaining metrics will not be shown.",
                    SyntaxWarning,
                )
                self.second_metric = self.metric_columns[1]
            elif len(self.metric_columns) == 2:
                self.second_metric = self.metric_columns[1]
            elif len(self.metric_columns) == 1:
                self.second_metric = None

        if self.primary_metric not in dataframe.columns:
            raise KeyError(
                "metric_column={} does not exist in the dataframe, please select a valid column. See a list of the available metrics with GraphFrame.show_metric_columns().".format(
                    self.primary_metric
                )
            )
        if (
            self.second_metric is not None
            and self.second_metric not in dataframe.columns
        ):
            raise KeyError(
                "metric_column={} does not exist in the dataframe, please select a valid column. See a list of the available metrics with GraphFrame.show_metric_columns().".format(
                    self.second_metric
                )
            )

        # grab the min and max value of the primary metric of a given rank
        # across all the nodes in the CCT, ignoring inf and nan values
        if "rank" in dataframe.index.names:
            metric_series = (dataframe.xs(self.rank, level=1))[self.primary_metric]
        else:
            metric_series = dataframe[self.primary_metric]
        isfinite_mask = np.isfinite(metric_series.values)
        filtered_series = pd.Series(
            metric_series.values[isfinite_mask], metric_series.index[isfinite_mask]
        )

        self.max_metric = filtered_series.max()
        self.min_metric = filtered_series.min()

        if self.unicode:
            self.lr_arrows = {"◀": "◀ ", "▶": "▶ "}
        else:
            self.lr_arrows = {"◀": "< ", "▶": "> "}

        # TODO: probably better to sort by time
        for root in sorted(roots, key=lambda n: n.frame):
            result += self.render_frame(root, dataframe)

        if self.color is True:
            result += self.render_legend()

        if self.unicode:
            return result
        else:
            return result.encode("utf-8")

    # pylint: disable=W1401
    def render_preamble(self):
        lines = [
            r"    __          __       __         __ ",
            r"   / /_  ____ _/ /______/ /_  ___  / /_",
            r"  / __ \/ __ `/ __/ ___/ __ \/ _ \/ __/",
            r" / / / / /_/ / /_/ /__/ / / /  __/ /_  ",
            r"/_/ /_/\__,_/\__/\___/_/ /_/\___/\__/  {:>2}".format("v" + __version__),
            r"",
            r"",
        ]

        return "\n".join(lines)

    def render_legend(self):
        def render_label(index, low, high):
            metric_range = self.max_metric - self.min_metric

            return (
                self.colors.colormap[index]
                + "█ "
                + self.colors.end
                + "{:.2f}".format(low * metric_range + self.min_metric)
                + " - "
                + "{:.2f}".format(high * metric_range + self.min_metric)
                + "\n"
            )

        legend = (
            "\n"
            + "\033[4m"
            + "Legend"
            + self.colors.end
            + " (Metric: "
            + self.primary_metric
            + " Min: {:.2f}".format(self.min_metric)
            + " Max: {:.2f}".format(self.max_metric)
            + ")\n"
        )

        legend += render_label(0, 0.9, 1.0)
        legend += render_label(1, 0.7, 0.9)
        legend += render_label(2, 0.5, 0.7)
        legend += render_label(3, 0.3, 0.5)
        legend += render_label(4, 0.1, 0.3)
        legend += render_label(5, 0.0, 0.1)

        legend += "\n" + self._ansi_color_for_name("name") + "name" + self.colors.end
        legend += " User code    "

        legend += self.colors.left + self.lr_arrows["◀"] + self.colors.end
        legend += " Only in left graph    "
        legend += self.colors.right + self.lr_arrows["▶"] + self.colors.end
        legend += " Only in right graph\n"

        return legend

    def render_frame(self, node, dataframe, indent="", child_indent=""):
        node_depth = node._depth
        if node_depth <= self.depth:
            # set dataframe index based on whether rank and thread are part of
            # the MultiIndex
            if "rank" in dataframe.index.names and "thread" in dataframe.index.names:
                df_index = (node, self.rank, self.thread)
            elif "rank" in dataframe.index.names:
                df_index = (node, self.rank)
            elif "thread" in dataframe.index.names:
                df_index = (node, self.thread)
            else:
                df_index = node

            node_metric = dataframe.loc[df_index, self.primary_metric]

            metric_precision = "{:." + str(self.precision) + "f}"
            metric_str = (
                self._ansi_color_for_metric(node_metric)
                + metric_precision.format(node_metric)
                + self.colors.end
            )

            if self.second_metric is not None:
                metric_str += " {c.faint}{second_metric:.{precision}f}{c.end}".format(
                    second_metric=dataframe.loc[df_index, self.second_metric],
                    precision=self.precision,
                    c=self.colors,
                )

            node_name = dataframe.loc[df_index, self.name]
            if self.expand is False:
                if len(node_name) > 39:
                    node_name = (
                        node_name[:18] + "..." + node_name[(len(node_name) - 18) :]
                    )
            name_str = (
                self._ansi_color_for_name(node_name) + node_name + self.colors.end
            )

            # 0 is "", 1 is "L", and 2 is "R"
            if "_missing_node" in dataframe.columns:
                left_or_right = dataframe.loc[df_index, "_missing_node"]
                if left_or_right == 0:
                    lr_decorator = ""
                elif left_or_right == 1:
                    lr_decorator = " {c.left}{decorator}{c.end}".format(
                        decorator=self.lr_arrows["◀"], c=self.colors
                    )
                elif left_or_right == 2:
                    lr_decorator = " {c.right}{decorator}{c.end}".format(
                        decorator=self.lr_arrows["▶"], c=self.colors
                    )

            result = "{indent}{metric_str} {name_str}".format(
                indent=indent, metric_str=metric_str, name_str=name_str
            )
            if "_missing_node" in dataframe.columns:
                result += lr_decorator
            if self.context in dataframe.columns:
                result += " {c.faint}{context}{c.end}\n".format(
                    context=dataframe.loc[df_index, self.context], c=self.colors
                )
            else:
                result += "\n"

            if self.unicode:
                indents = {"├": "├─ ", "│": "│  ", "└": "└─ ", " ": "   "}
            else:
                indents = {"├": "|- ", "│": "|  ", "└": "`- ", " ": "   "}

            # ensures that we never revisit nodes in the case of
            # large complex graphs
            if node not in self.visited:
                self.visited.append(node)
                # TODO: probably better to sort by time
                sorted_children = sorted(node.children, key=lambda n: n.frame)
                if sorted_children:
                    last_child = sorted_children[-1]

                for child in sorted_children:
                    if child is not last_child:
                        c_indent = child_indent + indents["├"]
                        cc_indent = child_indent + indents["│"]
                    else:
                        c_indent = child_indent + indents["└"]
                        cc_indent = child_indent + indents[" "]
                    result += self.render_frame(
                        child, dataframe, indent=c_indent, child_indent=cc_indent
                    )
        else:
            result = ""
            indents = {"├": "", "│": "", "└": "", " ": ""}

        return result

    def _ansi_color_for_metric(self, metric):
        metric_range = self.max_metric - self.min_metric

        if metric_range != 0:
            proportion_of_total = (metric - self.min_metric) / metric_range
        else:
            proportion_of_total = metric / 1

        if proportion_of_total > 0.9:
            return self.colors.colormap[0]
        elif proportion_of_total > 0.7:
            return self.colors.colormap[1]
        elif proportion_of_total > 0.5:
            return self.colors.colormap[2]
        elif proportion_of_total > 0.3:
            return self.colors.colormap[3]
        elif proportion_of_total > 0.1:
            return self.colors.colormap[4]
        elif proportion_of_total >= 0:
            return self.colors.colormap[5]
        else:
            return self.colors.blue

    def _ansi_color_for_name(self, node_name):
        if self.highlight is False:
            return ""

        if "<unknown procedure>" in node_name or "<unknown file>" in node_name:
            return ""
        else:
            return self.colors.bg_white_255 + self.colors.dark_gray_255

    class colors_enabled:
        colormap = []

        blue = "\033[34m"
        cyan = "\033[36m"

        bg_white_255 = "\033[48;5;246m"
        dark_gray_255 = "\033[38;5;232m"

        left = "\033[38;5;160m"
        right = "\033[38;5;28m"

        faint = "\033[2m"
        end = "\033[0m"

    class colors_disabled:
        colormap = ["", "", "", "", "", "", ""]

        def __getattr__(self, key):
            return ""

    colors_disabled = colors_disabled()
