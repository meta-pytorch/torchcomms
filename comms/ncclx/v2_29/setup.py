# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from distutils.command.build import build as _build

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
from setuptools.command.egg_info import egg_info as _egg_info

BUILDDIR = os.environ.get("BUILDDIR", None)
if BUILDDIR is not None:
    LIBDIR = os.path.join(BUILDDIR, "lib")
    BUILDDIR = os.path.join(BUILDDIR, "pybind")
    os.makedirs(BUILDDIR, exist_ok=True)


class build(_build):
    def initialize_options(self):
        super().initialize_options()
        self.build_base = BUILDDIR


class egg_info(_egg_info):
    def initialize_options(self):
        super().initialize_options()
        self.egg_base = BUILDDIR


def get_cmdclass():
    from pybind11.setup_helpers import build_ext

    return {
        "build": build,
        "egg_info": egg_info,
        "build_ext": build_ext,
    }


ext_modules = [
    Pybind11Extension(
        "ncclx_trainer_context",
        ["meta/py/wrapper.cc"],
        library_dirs=[LIBDIR],
        libraries=["nccl"],
    ),
]

setup(
    name="ncclx_trainer_context",
    ext_modules=ext_modules,
    cmdclass=get_cmdclass(),
)
