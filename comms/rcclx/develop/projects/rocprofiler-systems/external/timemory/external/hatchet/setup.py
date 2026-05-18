# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from setuptools import setup
from setuptools import Extension
from codecs import open
from os import path
from os import environ
import sysconfig

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get the version in a safe way which does not refrence hatchet `__init__` file
# per python docs: https://packaging.python.org/guides/single-sourcing-package-version/
version = {}
with open("./hatchet/version.py") as fp:
    exec(fp.read(), version)

# compiler configured from cmake
c_compiler = "@CMAKE_C_COMPILER@"
if c_compiler != "".join(["@", "CMAKE_C_COMPILER", "@"]):
    environ["CC"] = c_compiler

# Common flags for both release and debug builds.
if sysconfig.get_config_var("CFLAGS") is not None:
    extra_compile_args = (
        sysconfig.get_config_var("CFLAGS")
        .replace("-arch arm64", "")
        .replace("-flto", "")
        .replace("-ffat-lto-objects", "")
        .split()
    )
else:
    extra_compile_args = []
extra_compile_args += environ.get("CFLAGS", "").split()
config_compiler_args = "@CFLAGS@"
if config_compiler_args and config_compiler_args != "".join(["@", "CFLAGS", "@"]):
    extra_compile_args += config_compiler_args.split()

setup(
    name="hatchet",
    version=version["__version__"],
    description="A Python library for analyzing hierarchical performance data",
    url="https://github.com/hatchet/hatchet",
    author="Abhinav Bhatele",
    author_email="bhatele@cs.umd.edu",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="",
    packages=[
        "hatchet",
        "hatchet.readers",
        "hatchet.util",
        "hatchet.external",
        "hatchet.tests",
        "hatchet.cython_modules.libs",
    ],
    install_requires=[
        "pydot",
        "PyYAML",
        "matplotlib",
        "numpy",
        "pandas",
        "textX",
        "multiprocess",
        "caliper-reader",
        "pycubexr; python_version >= '3.6'",
    ],
    ext_modules=[
        Extension(
            "hatchet.cython_modules.libs.reader_modules",
            ["hatchet/cython_modules/reader_modules.c"],
            extra_compile_args=extra_compile_args,
        ),
        Extension(
            "hatchet.cython_modules.libs.graphframe_modules",
            ["hatchet/cython_modules/graphframe_modules.c"],
            extra_compile_args=extra_compile_args,
        ),
    ],
)
