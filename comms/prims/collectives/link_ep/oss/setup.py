# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Self-contained, conda-installable build of the in-house link_ep INTRANODE
dispatch/combine kernels, packaged as the top-level `link_ep` module.

See the design notes in this directory. The intranode path is folly/comms-free
under `-DLINK_EP_OSS_INTRANODE` (bootstrap ctor + LL/internode compiled out);
the ONLY external header it needs is transport/amd/HipHostCompat.h.

Build (inside the activated conda env, from this directory):
    export PYTORCH_ROCM_ARCH=gfx950
    python setup.py build_ext --inplace      # dev build (_cpp.so next to link_ep/)
  or:
    pip install --no-build-isolation -v .

IMPORTANT: torch's ROCm auto-hipify scans the extension's include dirs. Pointing
it at the source root is prohibitively slow (it recursively walks the whole
tree, which on a virtual filesystem can hang outright). So we STAGE a
minimal self-contained source tree (link_ep/cpp + HipHostCompat.h, mirroring the
`comms/prims/...` include layout) and point both sources and -I at the stage,
so hipify only ever scans this small tree.
"""

import os
import shutil

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Source root = the directory that contains `comms/`. This file lives at
# <SRC_ROOT>/comms/prims/collectives/link_ep/oss/, hence five levels up.
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", "..", ".."))
LINK_EP_REL = "comms/prims/collectives/link_ep"
HIPHOSTCOMPAT_REL = "comms/prims/transport/amd/HipHostCompat.h"

# ---------------------------------------------------------------------------
# Stage a minimal self-contained source tree under _gen/ so hipify's scan is
# confined to it (NOT the whole source tree). Mirror the include layout so
# `#include "comms/prims/..."` resolves with -I<stage>.
# ---------------------------------------------------------------------------
STAGE = os.path.join(THIS_DIR, "_gen")


def stage_sources() -> None:
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    # link_ep/cpp subtree (intranode + shared + Buffer/PyBindings/EventHandle/Config).
    shutil.copytree(
        os.path.join(SRC_ROOT, LINK_EP_REL, "cpp"),
        os.path.join(STAGE, LINK_EP_REL, "cpp"),
    )
    # The single external header the OSS intranode path includes.
    dst_hhc = os.path.join(STAGE, HIPHOSTCOMPAT_REL)
    os.makedirs(os.path.dirname(dst_hhc), exist_ok=True)
    shutil.copy(os.path.join(SRC_ROOT, HIPHOSTCOMPAT_REL), dst_hhc)


stage_sources()

CPP = os.path.join(STAGE, LINK_EP_REL, "cpp")
SOURCES = [
    os.path.join(CPP, "PyBindings.cc"),
    os.path.join(CPP, "Buffer.cc"),
    os.path.join(CPP, "shared", "EventHandle.cc"),
    os.path.join(CPP, "intranode", "Runtime.cc"),
    os.path.join(CPP, "intranode", "kernels", "Dispatch.cu"),
    os.path.join(CPP, "intranode", "kernels", "Combine.cu"),
    os.path.join(CPP, "intranode", "kernels", "Notify.cu"),
    os.path.join(CPP, "intranode", "kernels", "Layout.cu"),
    os.path.join(CPP, "shared", "kernels", "_anchor.cu"),
]

OSS_DEFINE = "-DLINK_EP_OSS_INTRANODE"
extra_compile_args = {
    "cxx": ["-O3", OSS_DEFINE, "-Wno-unused-parameter", "-Wno-unused-variable"],
    # BuildExtension routes this to hipcc on a ROCm torch and hipifies the
    # staged sources (cuda_runtime.h -> hip_runtime.h, cudaXxx -> hipXxx).
    "nvcc": ["-O3", OSS_DEFINE],
}

setup(
    name="link_ep",
    version="0.1.0",
    packages=["link_ep"],
    package_dir={"link_ep": os.path.join(SRC_ROOT, LINK_EP_REL, "link_ep")},
    ext_modules=[
        CUDAExtension(
            name="link_ep._cpp",
            sources=SOURCES,
            include_dirs=[STAGE],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
