# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from distutils.command.build import build as _build

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
    return {
        "build": build,
        "egg_info": egg_info,
    }


# For now, RCCLX doesn't have pybind extensions like NCCLX
# This can be expanded later when pybind wrappers are added
ext_modules = []

setup(
    name="rcclx_trainer_context",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass=get_cmdclass(),
)
