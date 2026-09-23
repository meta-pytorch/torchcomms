# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import functools


def deprecated_params(**old_to_new):
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            rename_kwargs(f.__name__, old_to_new, kwargs)
            return f(*args, **kwargs)

        return wrapper

    return deco


def rename_kwargs(fname, old_to_new, kwargs):
    for old, new in old_to_new.items():
        if old in kwargs:
            if new in kwargs:
                raise TypeError(
                    '{}() received both "{}=" and "{}=".'.format(fname, old, new)
                )

            # if parameter has been removed
            if not new:
                raise ValueError(
                    '{}() parameter "{}=" will be removed in a future release. Please remove it from this script.'.format(
                        fname, old
                    )
                )
            # if parameter has been deprecated and renamed
            else:
                raise ValueError(
                    '{}() parameter "{}=" has been deprecated, please use "{}=".'.format(
                        fname, old, new
                    )
                )
                kwargs[new] = kwargs.pop(old)
