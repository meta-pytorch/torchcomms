# Copyright 2022-2023 The University of Arizona and other Hatchet Project
# Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import inspect
import os
import getpass
import json
from datetime import datetime
import hatchet


def isJsonable(var):
    try:
        json.dumps(var)
        return True
    except TypeError:
        return False


class Log(object):
    def __init__(self, filename="hatchet.log", active=False):
        self._log_file = filename
        self._active = active

        # ensures we only log api calls made explicitly by
        # a user
        self._nested = False

    def set_output_file(self, filename=""):
        self._log_file = filename

    def set_active(self):
        self._active = True

    def set_inactive(self):
        self._active = False

    def append_to_file(self, log):
        """Manages the opening and writing of log information to a file."""

        with open(self._log_file, "a") as f:
            try:
                f.write(json.dumps(log) + "\n")
            except TypeError as e:
                raise e

    def loggable(self, function):
        """A decrator which logs calls to hatchet functions"""

        def inner(*args, **kwargs):
            try:
                # turn on logger
                if (
                    "logging" in kwargs
                    and kwargs["logging"] is True
                    and not self._active
                ):
                    self._active = True

                    self._log_file = "hatchet_{}.log".format(
                        datetime.now().replace(microsecond=0).isoformat()
                    )
                    print(
                        """Notification: Local logging of Hatchet function calls has been enabled. Output will be dumped to {}""".format(
                            self._log_file
                        )
                    )

                elif "logging" in kwargs and kwargs["logging"] is False:
                    self._active = False
                    print(
                        """Notification: Local logging of Hatchet function calls has been disabled."""
                    )

                # if logger is on and user called function
                if self._active and not self._nested:
                    log_dict = {}
                    arg_list = []
                    serlizable_kwargs = {}

                    # Get a user id
                    try:
                        log_dict["user_id"] = os.getuid()
                    except AttributeError:
                        # for windows machines
                        log_dict["user_id"] = getpass.getuser()

                    for i, arg in enumerate(args):
                        if inspect.isfunction(arg):
                            # log functions passed (eg. lambda passed to filter) as source code
                            arg_source = inspect.getsource(arg)
                            arg_list.append(arg_source)

                        elif isinstance(arg, hatchet.GraphFrame):
                            # log a graphframe as a dictionary of metadata
                            graphframe_metadata = {}

                            graphframe_metadata["object"] = arg.__class__.__name__
                            graphframe_metadata["id"] = id(arg)
                            graphframe_metadata["rows"] = arg.dataframe.shape[0]
                            graphframe_metadata["nodes"] = len(arg.graph)

                            arg_list.append(graphframe_metadata)
                        else:
                            # log everything else
                            if isJsonable(arg):
                                arg_list.append(arg)
                            else:
                                arg_list.append(arg.__repr__())

                    log_dict["start"] = datetime.now().isoformat()

                    self._nested = True
                    holder = function(*args, **kwargs)
                    self._nested = False

                    log_dict["end"] = datetime.now().isoformat()
                    log_dict["function"] = function.__name__
                    log_dict["args"] = tuple(arg_list)

                    for key in kwargs:
                        if isJsonable(kwargs[key]):
                            serlizable_kwargs[key] = kwargs[key]
                        else:
                            serlizable_kwargs[key] = kwargs[key].__repr__()

                    log_dict["kwargs"] = serlizable_kwargs

                    self.append_to_file(log_dict)

                    return holder
                else:
                    return function(*args, **kwargs)

            # If there is a file io error when logging
            # we run function as normal and abandon log
            except IOError:
                return function(*args, **kwargs)

        return inner

    # TODO: add log reader


Logger = Log()
