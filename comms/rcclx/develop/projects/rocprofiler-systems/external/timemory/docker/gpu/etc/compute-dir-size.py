#!/usr/bin/env python3

import os
import sys

BYTE = (1, "B")
KILOBYTE = (1024 * BYTE[0], "KB")
MEGABYTE = (1024 * KILOBYTE[0], "MB")
GIGABYTE = (1024 * MEGABYTE[0], "GB")
TERABYTE = (1024 * GIGABYTE[0], "TB")


def get_file_size(path):
    return os.path.getsize(path) if os.path.isfile(path) else 0


def compute_units(cum_fsize):
    if os.environ.get("DEBUG") is not None:
        print("Compute units for {}".format(cum_fsize))

    for unit in [TERABYTE, GIGABYTE, MEGABYTE, KILOBYTE, BYTE]:
        if os.environ.get("DEBUG") is not None:
            print("Checking {}".format(unit))
        result = cum_fsize / unit[0]
        if os.environ.get("DEBUG") is not None:
            print("file size: {} {}".format(result, unit[1]))
        if result > 1.0:
            return unit
    return BYTE


def main():
    cum_filesize = 0
    args = sys.argv[1:]
    if len(args) == 0:
        args.append(os.getcwd())
    if os.environ.get("DEBUG") is not None:
        print("Directories: {}".format(args))

    for arg in args:
        if os.path.exists(arg):
            files = os.listdir(arg)
            for f in files:
                f = os.path.join(arg, f)
                fsize = get_file_size(f)
                if os.environ.get("DEBUG") is not None:
                    print("file '{}' is {} bytes".format(f, fsize))
                cum_filesize += fsize
    units = compute_units(cum_filesize)
    if os.environ.get("DEBUG") is not None:
        print("Units: {} {}".format(units[0], units[1]))
    fsize = cum_filesize / units[0]
    print("{} {}".format(int(fsize), units[1]))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if os.environ.get("DEBUG") is not None:
            import traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, limit=5)
            print('Exception - {}'.format(e))
        else:
            print("")
