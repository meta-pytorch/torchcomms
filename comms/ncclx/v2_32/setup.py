# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
from setuptools.command.build import build as _build
from setuptools.command.egg_info import egg_info as _egg_info

BUILD_ROOT = os.environ.get("BUILDDIR", os.path.abspath("build"))
LIBDIR = os.path.join(BUILD_ROOT, "lib")
BUILDDIR = os.path.join(BUILD_ROOT, "pybind")
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
        ["../meta/py/wrapper.cc"],
        library_dirs=[LIBDIR],
        libraries=["nccl"],
    ),
]

setup(
    name="ncclx_trainer_context",
    ext_modules=ext_modules,
    cmdclass=get_cmdclass(),
)
