#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

# Method with genctran.py is used to pararalye compilation. Without it compiler thread would need
# to compile many instantiations for one source file (someAlgo.cu) but with genctran.py we generate
# many source files like someAlgoInt.cu, someAlgoFlot.cu so compiler threads could compile
# intantiations separately. Introducing this approach got speed up ~ 20x, see D59433968.
# TODO: move all ctran algorithm compilation to genctran.py approach T240136045

import os
import sys

types = [
    "__nv_bfloat16",
    "__nv_fp8_e4m3",
    "__nv_fp8_e5m2",
    "int8_t",
    "double",
    "float",
    "half",
    "int32_t",
    "int64_t",
    "uint32_t",
    "uint64_t",
    "uint8_t",
]
ops = ["avg", "max", "min", "prod", "sum"]

header = "// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary."


def gen_algo_files(gensrc, srcs, rules, algo_info):
    """
    Generic function to generate kernel instantiation files.

    Args:
        algo_info: dict with keys:
            - 'bases': list of algorithm base names (e.g., ["AllReduceDirect"])
            - 'dir': subdirectory under algos (e.g., "AllReduce")
            - 'has_ops': whether algorithm needs reduction operations
    """
    for base in algo_info["bases"]:
        for type in types:
            if algo_info["has_ops"]:
                # Generate files with operations (e.g., AllReduceDirect_float_sum.cu)
                for op in ops:
                    file = base + "_" + type + "_" + op
                    f = open(os.path.join(gensrc, file + ".cu"), "w")

                    f.write(header)
                    f.write("\n\n")
                    f.write(
                        f'#include "comms/ctran/algos/{algo_info["dir"]}/{base}.cuh"'
                    )
                    f.write("\n\n")

                    if type == "__nv_bfloat16":
                        f.write("#if defined(__CUDA_BF16_TYPES_EXIST__)")
                        f.write("\n")
                    elif type == "__nv_fp8_e4m3" or type == "__nv_fp8_e5m2":
                        f.write(
                            "#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)"
                        )
                        f.write("\n")

                    f.write(
                        f"DECL_CTRAN_{base.upper()}_KERN("
                        + type
                        + ", comm"
                        + op.capitalize()
                        + ");"
                    )
                    f.write("\n")

                    if (
                        type == "__nv_bfloat16"
                        or type == "__nv_fp8_e4m3"
                        or type == "__nv_fp8_e5m2"
                    ):
                        f.write("#endif")
                        f.write("\n")

                    f.close()
                    srcs += [file + ".cu"]
            else:
                # Generate files without operations (e.g., AllGatherDirect_float.cu)
                file = base + "_" + type
                f = open(os.path.join(gensrc, file + ".cu"), "w")

                f.write(header)
                f.write("\n\n")
                f.write(f'#include "comms/ctran/algos/{algo_info["dir"]}/{base}.cuh"')
                f.write("\n\n")

                if type == "__nv_bfloat16":
                    f.write("#if defined(__CUDA_BF16_TYPES_EXIST__)")
                    f.write("\n")
                elif type == "__nv_fp8_e4m3" or type == "__nv_fp8_e5m2":
                    f.write(
                        "#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)"
                    )
                    f.write("\n")

                f.write(f"DECL_CTRAN_{base.upper()}_KERN(" + type + ");")
                f.write("\n")

                if (
                    type == "__nv_bfloat16"
                    or type == "__nv_fp8_e4m3"
                    or type == "__nv_fp8_e5m2"
                ):
                    f.write("#endif")
                    f.write("\n")

                f.close()
                srcs += [file + ".cu"]


def gen_allreduce_files(gensrc, srcs, rules):
    gen_algo_files(
        gensrc,
        srcs,
        rules,
        {"bases": ["AllReduceDirect"], "dir": "AllReduce", "has_ops": True},
    )


def gen_allgather_files(gensrc, srcs, rules):
    gen_algo_files(
        gensrc,
        srcs,
        rules,
        {"bases": ["AllGatherDirect"], "dir": "AllGather", "has_ops": False},
    )


def gen_reduce_scatter_files(gensrc, srcs, rules):
    gen_algo_files(
        gensrc,
        srcs,
        rules,
        {
            "bases": ["ReduceScatterDirect", "ReduceScatterRing", "ReduceScatterRHD"],
            "dir": "ReduceScatter",
            "has_ops": True,
        },
    )


def gen_alltoall_files(gensrc, srcs, rules):
    gen_algo_files(
        gensrc,
        srcs,
        rules,
        {"bases": ["AllToAll"], "dir": "AllToAll", "has_ops": False},
    )


def genalgos(gensrc):
    srcs = []
    rules = open(os.path.join(gensrc, "ctran_rules.mk"), "w")
    gen_allreduce_files(gensrc, srcs, rules)
    gen_allgather_files(gensrc, srcs, rules)
    gen_reduce_scatter_files(gensrc, srcs, rules)
    gen_alltoall_files(gensrc, srcs, rules)

    rules.write("CTRAN_GEN_SRCS = ")
    for src in srcs:
        rules.write("$(OBJDIR)/gensrc/" + src + " ")
    rules.write("\n")
    rules.close()


if __name__ == "__main__":
    gensrc = sys.argv[1]

    if os.path.exists(gensrc):
        for name in os.listdir(gensrc):
            os.remove(os.path.join(gensrc, name))
    else:
        os.mkdir(gensrc)

    genalgos(gensrc)
